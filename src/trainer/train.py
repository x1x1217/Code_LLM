import os, sys, json, random, gc
from datasets import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from generator import CodeGenerator, build_prompt, post_process, evaluate_single, remove_prompt_function_declaration
from data_loader import KodCodeDataset
from memory.memory_manager import MemoryManager
from scheduler.scheduler import Scheduler
from trainer import Trainer, Config
from multiprocessing import Pool

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
import torch
import torch_npu

'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
'''

class DPOData_Collector:
    
    def __init__(self):
        self.preference_data = []
        
    def add_candidates(self, prompt, candidates_with_results):
        passed = [code for code, res in candidates_with_results if res]
        failed = [code for code, res in candidates_with_results if not res]
        
        if not passed or not failed:
            return 0
        
        pairs_added = 0
        for i, chosen in enumerate(passed):
            rejected = failed[i % len(failed)]
            self.preference_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
            pairs_added += 1
            
        # max_pairs = min(len(passed), len(failed), 3)  
        # for i in range(max_pairs):
        #     self.preference_data.append({
        #         "prompt": prompt,
        #         "chosen": passed[i][0],
        #         "rejected": failed[i][0],
        #     })
        #     pairs_added += 1
        
        return pairs_added
    
    def to_hf_dataset(self):
        if not self.preference_data:
            return None
        
        return Dataset.from_list(self.preference_data)


class Validation:
    
    def __init__(self, kodcode_dataset, val_pool, code_generator, config, trainer):
        self.kodcode_dataset = kodcode_dataset
        self.val_pool = val_pool
        self.code_generator = code_generator
        self.config = config
        self.trainer = trainer
        
        self.best_pass_rate = 0.0
        self.patience_cnt = 0
        
        self.val_used = self.val_pool[:100]
        self.val_samples = [self.kodcode_dataset.get_by_index(i) for i in self.val_used]
        # self.val_samples = [self.kodcode_dataset.get_by_index(i) for i in self.val_pool]
                
    def validate(self):
        print("\n====== Validation ======")    
        
        prompts = []
        for sample in self.val_samples:
            ques = self.kodcode_dataset.get_question(sample)
            info = self.kodcode_dataset.get_test_info(sample)
            prompts.append(build_prompt(ques, info, self.config.model_type))
            
        batch_size = 25
        num_candidates = 5
        all_completions = []
            
        for batch_idx in range(0, len(prompts), batch_size):
            batch_end = min(batch_idx + batch_size, len(prompts))
            batch_prompts = prompts[batch_idx:batch_end]
                        
            batch_completions = self.code_generator.generate(batch_prompts, num_candidates=num_candidates)
            all_completions.extend(batch_completions)

            gc.collect()
            torch.npu.empty_cache()
        
        completions = all_completions
            
        from multiprocessing import Pool
        eval_tasks = []
        
        for i, cands in enumerate(completions):
            sample = self.val_samples[i]
            test_code = self.kodcode_dataset.get_test(sample)
            test_info = self.kodcode_dataset.get_test_info(sample)
            
            for cand in cands:
                processed = post_process(cand, test_info)
                eval_tasks.append((processed, test_code))
        
        with Pool(self.config.num_eval_workers) as pool:
            results = pool.map(evaluate_single, eval_tasks)
            
        pass_5_cnt = 0
        for i in range(len(self.val_samples)):
            task_results = results[i * 5 : (i + 1) * 5]
            if any(task_results):
                pass_5_cnt += 1
        
        pass_5 = pass_5_cnt / len(self.val_samples)
        print(f"Validation pass@5: {pass_5:.2%}")
        return pass_5
    
    def check_early_stopping(self, val_pass_rate):
        if val_pass_rate > self.best_pass_rate:
            self.best_pass_rate = val_pass_rate
            self.patience_cnt = 0
            
            adapter_path = os.path.join(self.config.checkpoint_dir, "best_adapter")
            os.makedirs(adapter_path, exist_ok=True)
            self.trainer.save_adapter(adapter_path)
            print(f"New best model saved. pass@5: {val_pass_rate:.2%}")
            
            return False
        else:
            self.patience_cnt += 1
            print(f"No improvement. Patience: {self.patience_cnt}. Best pass@5: {self.best_pass_rate:.2%}")
            
            if self.patience_cnt >= self.config.patience:
                print(f"Early stopping triggered. Best pass@5: {self.best_pass_rate:.2%}")
                return True
        return False

def train_sft(config, kodcode_dataset, memory_manager, scheduler, code_generator, trainer, validation, dpo_collector):
    seen_all = False
    train_tasks = int(config.total_tasks * config.train_ratio)
    
    for round_idx in range(1, config.max_rounds + 1):
        status = memory_manager.get_status(train_tasks)
        
        if status["graduation_rate"] > 0.95:
            print("Graduation Rate > 95%. Training complete. Exiting.")
            break
        
        if status['total_seen'] == train_tasks:
            seen_all = True
            
        task_selected = scheduler.select_tasks(round_idx, config.task_per_round)
        if not task_selected:
            break
        
        print(f"\n====== Round {round_idx} [SFT] ======")
        print(f"Seen: {status['total_seen']} | Graduated: {status['graduated']} ({status['graduation_rate']:.1%})")
        
        current_samples = [kodcode_dataset.get_by_index(i) for i in task_selected]
        print(f"Selected {len(task_selected)} tasks: {task_selected} for training.")

        prompt = [
            build_prompt(
                kodcode_dataset.get_question(sample),
                kodcode_dataset.get_test_info(sample),
                config.model_type
            )
            for sample in current_samples
        ]
        
        batch_size = 25
        all_completions = []
        
        for batch_idx in range(0, len(prompt), batch_size):
            batch_end = min(batch_idx + batch_size, len(prompt))
            batch_prompts = prompt[batch_idx:batch_end]
            
            gc.collect()
            torch.npu.empty_cache()

            batch_completions = code_generator.generate(batch_prompts, num_candidates=config.num_candidates)
            all_completions.extend(batch_completions)

            gc.collect()
            torch.npu.empty_cache()
        
        completions = all_completions
                
        eval_tasks = []
        for i, candidates in enumerate(completions):
            test_code = kodcode_dataset.get_test(current_samples[i])
            test_info = kodcode_dataset.get_test_info(current_samples[i])
            
            for cand in candidates:
                processed_code = post_process(cand, test_info)
                eval_tasks.append((processed_code, test_code))
                
                # ====== output completions ======
                debug_log_path = "./completion.jsonl"
                
                log_entry = {
                    "task_id": i,
                    "completion": processed_code,
                    # "prompt": prompt[i // config.num_candidates]
                }
                
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                # ====== output completions ======
                
                
        with Pool(config.num_eval_workers) as pool:
            results = pool.map(evaluate_single, eval_tasks)
            
        sft_samples = []
        res_idx = 0
        
        dpo_pairs_added = 0
        
        for i, task_id in enumerate(task_selected):
            task_res = results[res_idx : res_idx + config.num_candidates]
            task_cands = [eval_tasks[res_idx + j][0] for j in range(config.num_candidates)]
            res_idx += config.num_candidates
            
            pass_rate = sum(task_res) / config.num_candidates
            
            # ====== output pass_rate ======
            pr_log_path = "./pass_rate_log.jsonl"
            
            log_entry = {
                    "task_id": i,
                    "pass_rate": pass_rate,
                }
                
            with open(pr_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            # ====== output pass_rate ======
            
            memory_manager.update(task_id, pass_rate, round_idx)
            
            dpo_pairs_added += dpo_collector.add_candidates(prompt[i], list(zip(task_cands, task_res)))
            
            passed = [code for code, res in zip(task_cands, task_res) if res]
            solution = passed[0] if passed else kodcode_dataset.get_solution(current_samples[i])
            sft_samples.append({
                "prompt": prompt[i],
                "solution": solution
            })
            
        print(f"DPO pairs added: {dpo_pairs_added}")
        print(f"DPO total pairs: {len(dpo_collector.preference_data)}")
        
        if sft_samples:
            print(f"[SFT] Training on {len(sft_samples)} samples...")
            sft_loss = trainer.train_step_sft(sft_samples)
            
        trainer.save_adapter()
        code_generator.update_lora(config.lora_adapter_path)
        
        # ==========del cache==========
        del current_samples, prompt, completions, eval_tasks, results
        gc.collect()
        torch.npu.empty_cache()
        # ==========del cache==========
        
        if seen_all and round_idx % config.val_interval == 0:
            val_pass_rate = validation.validate()
            if validation.check_early_stopping(val_pass_rate):
                break
        
        if round_idx % 50 == 0:
            adapter_path = os.path.join(config.checkpoint_dir, f"lora_adapter_{round_idx}")
            os.makedirs(adapter_path, exist_ok=True)
            trainer.save_adapter(adapter_path)
            print(f"Checkpoint saved at round {round_idx}")
            
        memory_manager.save()

def train_dpo(config, dpo_collector, validation, code_generator, trainer):
    print(f"\n====== [DPO] ======")
    
    train_dataset = dpo_collector.to_hf_dataset()
    print(f"DPO Dataset size: {len(train_dataset)}")
    
    # del code_generator.llm
    # del code_generator
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # import time
    # time.sleep(3)
    
    policy_model = trainer.model
    tokenizer = trainer.tokenizer
    policy_model.train()
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
        # device_map={"": 1}
    )
    adapter_path = os.path.join(config.checkpoint_dir, "best_adapter")
    ref_model = PeftModel.from_pretrained(ref_model, adapter_path)
    ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    
    for param in ref_model.parameters():
        param.requires_grad = False
        
    training_args = DPOConfig(
        output_dir=os.path.join(config.checkpoint_dir, "dpo_output"),
        num_train_epochs=config.dpo_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.dpo_lr,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        remove_unused_columns=False,
        beta=config.dpo_beta,
        max_length=config.max_tokens,
        max_prompt_length=config.max_tokens // 2,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,  
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    dpo_trainer.train()
    dpo_path = os.path.join(config.checkpoint_dir, "final_dpo")
    dpo_trainer.save_model(dpo_path)

def train():
    config = Config()
    
    print("Initializing System...")
    
    kodcode_dataset = KodCodeDataset(config.kodcode_path)
    memory_manager = MemoryManager(config.memory_path)
    scheduler = Scheduler(memory_manager, config.total_tasks, config.train_ratio, config.new_task_ratio)
    
    code_generator = CodeGenerator(
        model_path=config.model_path,
        lora_path=config.lora_adapter_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        max_model_len=config.max_model_len
    )
    
    trainer = Trainer(config)
    validation = Validation(kodcode_dataset, scheduler.select_val_tasks(), code_generator, config, trainer)
    dpo_collector = DPOData_Collector()
    train_sft(config, kodcode_dataset, memory_manager, scheduler, code_generator, trainer, validation, dpo_collector)
    train_dpo(config, dpo_collector, validation, code_generator, trainer)

if __name__ == "__main__":
    train()
    
    '''
    CUDA_VISIBLE_DEVICES=5,4 \
    python train.py
    
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5,4 python train.py
    '''
    
