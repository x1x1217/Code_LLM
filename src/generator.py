'''
generator k candidate codes with results
multi-process to evaluate them
'''

import os, json
from typing import List
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from multiprocessing import Pool
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_prompt_instruct(question, test_info):
    prompt = "<|im_start|>system\n"
    prompt += "You are a Python expert. Write only the function implementation without explanations.<|im_end|>\n"
    
    prompt += "<|im_start|>user\n"
    prompt += f"{question}\n\n"
    prompt += "Complete this function:\n"
    prompt += f"```python\n{test_info['function_declaration']}\n    pass\n```<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    prompt += "```python\n"
    
    return prompt

def build_prompt_deepseek(question, test_info):
    function_declaration = test_info['function_declaration']
    
    prompt = "### Instruction:\n"
    prompt += "Write a complete Python function for the problem with the provided function declaration. "
    prompt += "Output only clean Python code without any comments, docstrings, or explanations.\n\n"
    prompt += "Question: " + question.strip() + "\n"
    prompt += "Function declaration: " + function_declaration + "\n\n"
    prompt += "### Response:\n"
    prompt += "```python\n"
    
    return prompt

def build_prompt_codellama(question, test_info):
    function_declaration = test_info['function_declaration']
    
    prompt = "[INST] <<SYS>>\n"
    prompt += "You are an expert Python programmer. "
    prompt += "You always write clean, efficient, and correct code. "
    prompt += "You output only code without any explanations or comments.\n"
    prompt += "<</SYS>>\n\n"
    
    prompt += "Write a complete Python function for the problem with the provided function declaration.\n\n"
    prompt += f"Question: {question.strip()}\n"
    prompt += f"Function declaration: {function_declaration}\n"
    prompt += "[/INST] ```python\n"
    
    return prompt
    

def build_prompt(question, test_info, model_type=None):
    if model_type == "Deepseek":
        return build_prompt_deepseek(question, test_info)
    elif model_type == "Qwen":
        pass
    elif model_type == "CodeLlama":
        return build_prompt_codellama(question, test_info)
    else:
        function_declaration = test_info['function_declaration']
        prompt = "Write a complete Python function for the problem with the provided function declaration. Output only clean Python code without any comments, docstrings, or explanations.\n"
        prompt += "Question: " + question.strip() + "\n"
        prompt += "Function declaration: " + function_declaration + "\n\n"
        prompt += "```python\n"
    
    return prompt

def post_process_instruct(code):
    
    for tok in ["<|text|>", "<|code|>", "<|execution|>", "<|assistant|>", "<|user|>", "<|endofblock|>", "<|endofmessage|>"]:
        code = code.replace(tok, '')
        
    try:
        if code.count('```') % 2 == 1:
            code = code[:code.rfind('```')]
        else:
            code = code[code.find('```') + 3:]
            code = code[code.find('\n') + 1:]
    except:
        pass
    
    return code.strip()

def post_process(code, test_info):
    try:
        if code.count('```') % 2 == 1:
            code = code[:code.rfind('```')]
        elif code.count('```') >= 2:
            code = code[code.find('```') + 3:]
            code = code[code.find('\n') + 1:]
            code = code[:code.rfind('```')]
    except:
        pass
    
    function_declaration = test_info['function_declaration']
    if function_declaration not in code:
        code = function_declaration + '\n' + code
    
    return code.strip()

def remove_prompt_function_declaration(prompt, function_declaration):
    if not function_declaration:
        return prompt
    
    suffix = f"{function_declaration}\n"
    
    if prompt.endswith(suffix):
        return prompt[:-len(suffix)]
    
    if prompt.endswith(function_declaration):
        return prompt[:-len(function_declaration)]
    
    return prompt

class CodeGenerator:
    
    def __init__(self, model_path, lora_path=None, tensor_parallel_size=1, max_tokens=512, temperature=0.2, top_p=0.95, gpu_memory_utilization=0.6, max_model_len=4096):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True, 
            max_loras=8,
            max_lora_rank=64,
            # max_model_len=max_model_len,
            # enforce_eager=True
        )
        
        self.lora_request = None
        self.lora_id_counter = 1
        
        if lora_path and os.path.exists(lora_path):
            self.update_lora(lora_path)
        else:
            print("No LoRA adapter loaded. Running base model.")
        
    def generate(self, prompts, num_candidates=1):
        
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=['<|endofblock|>', '<|endofmessage|>']
        )
        
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_candidates)
        if self.lora_request:
            outputs = self.llm.generate(expanded_prompts, sampling_params, lora_request=self.lora_request)
        else:
            outputs = self.llm.generate(expanded_prompts, sampling_params)
            
        completions: List[List[str]] = [[] for _ in range(len(prompts))]
        
        for i, out in enumerate(outputs):
            prompt_index = i // num_candidates
            for o in out.outputs:
                completions[prompt_index].append(o.text)
            
        return completions
    
