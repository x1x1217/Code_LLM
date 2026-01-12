import os
import torch, datetime, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import torch_npu

import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

class Config:
    model_path = "/home/ma-user/work/models/Deepseek/deepseek-coder-6.7b-instruct"
    kodcode_path = "/home/ma-user/work/datasets/KodCode-V1/data"
    memory_path = "../memory/memory_infos/memory_kodcode_train.json"  
    checkpoint_dir = "./checkpoints/Deepseek-coder-6.7b-instruct_10000/duration_adapter"
    log_dir = "../../logs"
    lora_adapter_path = "./checkpoints/Deepseek-coder-6.7b-instruct_10000/current_lora_adapter"
    ref_adapter_path = "./checkpoints/Deepseek-coder-6.7b-instruct_10000/ref_lora_adapter"
    
    max_rounds = 1000
    sft_ratio = 1
    
    batch_size = 4
    gradient_accumulation_steps = 4
    seed = 42
    
    num_candidates = 5
    new_task_ratio = 0.7
    total_tasks = 10000
    train_ratio = 0.8
    
    tensor_parallel_size = 2
    gpu_memory_utilization = 0.7
    temperature = 0.7
    top_p = 0.95
    top_k = 50
    # max_tokens = 1024
    max_tokens = 2048
    max_model_len = 2048
    
    max_prompt_len = 1536
    max_solution_len = 512
    
    lora_r = 8
    lora_alpha = 16
    # lora_r = 16
    # lora_alpha = 32
    lora_dropout = 0.05
    
    # num_eval_workers = 24
    num_eval_workers = 32
    
    sft_lr = 5e-5
    task_per_round = 40
    
    dpo_lr = 1e-5
    dpo_beta = 0.1
    dpo_epochs = 2
    
    patience = 5
    val_interval = 5
    
    model_type = "Deepseek"

class Trainer:

    def __init__(self, config):
        self.config = config
        
        '''
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        original_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        ASCEND_RT_VISIBLE_DEVICES
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        set_seed(self.config.seed)
        
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1,2,3'
        original_visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '2,3'

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            padding_side='right'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            # max_memory={0: "0GB", 1: "64GB"}
            # device_map={"": 1}
            # device_map="auto"
            # device_map="cuda:1"
            # device_map={"": "cuda:1"}
        )
        
        self.peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
            
        # adapter_model_path = os.path.join(config.lora_adapter_path, "adapter_model.safetensors")
        # if os.path.exists(adapter_model_path):
        #     self.model = PeftModel.from_pretrained(self.model, config.lora_adapter_path)
        #     print(f"Loaded existing LoRA adapter from {config.lora_adapter_path}.")
        # else:
            # self.model = get_peft_model(self.model, self.peft_config)
            # print("Initialized new LoRA adapter.")
        self.model = get_peft_model(self.model, self.peft_config)
        print("Initialized new LoRA adapter.")
        
        # ======== new added ========
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        # ======== new added ========
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.sft_lr)
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.max_rounds, 
            eta_min=1e-6
        )

        '''
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        os.environ['CUDA_VISIBLE_DEVICES'] = original_visible
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        '''        
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = original_visible

        os.makedirs(config.log_dir, exist_ok=True)
        self.log_file = os.path.join(config.log_dir, "training_log.txt")
        
        print("Trainer initialized.")
        
    def format_samples(self, samples):
        processed_batch = []
        skipped_by_len = 0
        
        for sample in samples:
            prompt = sample['prompt']
            solution = sample['solution']
            
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            solution_ids = self.tokenizer.encode(solution + self.tokenizer.eos_token, add_special_tokens=False)
    
            # ======== new added ========
            if len(prompt_ids) > self.config.max_prompt_len:
                # prompt_ids = prompt_ids[-self.config.max_prompt_len:]
                skipped_by_len += 1
                continue

            if len(solution_ids) > self.config.max_solution_len:
                skipped_by_len += 1
                continue
            # ======== new added ========
                
            full_ids = prompt_ids + solution_ids
            
            labels = [-100] * len(prompt_ids) + solution_ids
            if len(full_ids) > self.config.max_tokens:
                full_ids = full_ids[:self.config.max_tokens]
                labels = labels[:self.config.max_tokens]
                
            attention_mask = [1] * len(full_ids)
            
            processed_batch.append({
                'input_ids': full_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
            
        if skipped_by_len > 0:
            print(f"Skipped {skipped_by_len}/{len(samples)} samples with long solutions")
            
        return processed_batch
    
    def train_step_sft(self, samples):
        if not samples:
            return 0.0
        
        self.model.train()
        total_loss = 0
        
        processed_batch = self.format_samples(samples)
        
        # ========new added========
        processed_batch.sort(key=lambda x: len(x['input_ids']))
        # ========new added========
                
        batch_size = self.config.batch_size
        accumulation_steps = self.config.gradient_accumulation_steps
        steps = range(0, len(processed_batch), batch_size)
        
        # ========new added========
        # self.optimizer.zero_grad()
        # ========new added========
        
        for step_idx, i in enumerate(steps):
            batch_items = processed_batch[i : i + batch_size]
            max_len = max([len(item["input_ids"]) for item in batch_items])
            
            input_ids_batch = []
            attention_mask_batch = []
            labels_batch = []
            
            for item in batch_items:
                current_len = len(item["input_ids"])
                pad_len = max_len - current_len

                padded_ids = item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
                padded_mask = item["attention_mask"] + [0] * pad_len
                padded_labels = item["labels"] + [-100] * pad_len
                
                input_ids_batch.append(torch.tensor(padded_ids))
                attention_mask_batch.append(torch.tensor(padded_mask))
                labels_batch.append(torch.tensor(padded_labels))
                
            input_ids_tensor = torch.stack(input_ids_batch).to(self.model.device)
            attention_mask_tensor = torch.stack(attention_mask_batch).to(self.model.device)
            labels_tensor = torch.stack(labels_batch).to(self.model.device)
            
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                labels=labels_tensor
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps
            
            if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(steps):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            
        current_lr = self.scheduler_lr.get_last_lr()[0]
        self.scheduler_lr.step()
        
        avg_loss = total_loss / len(steps)
            
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.datetime.now()} | Samples: {len(samples)} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}\n")
            
        print(f"Train Step Finished. Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        # ==========del cache==========
        del input_ids_tensor, attention_mask_tensor, labels_tensor, outputs, loss
        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
        # ==========del cache==========
        
        return avg_loss
    
    def save_adapter(self, adapter_path=None):
        if adapter_path is None:
            adapter_path = self.config.lora_adapter_path
            
        print(f"Saving adapter to {adapter_path}")
        self.model.save_pretrained(adapter_path)
        
if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
