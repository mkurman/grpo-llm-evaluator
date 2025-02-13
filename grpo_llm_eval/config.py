from dataclasses import dataclass
from typing import List
import os

@dataclass
class TrainingConfig:
    openai_base_url: str = "http://localhost/v1/"
    student_model_name: str = "mkurman/Llama-3.2-MedIT-SUN-2.5B-BT-GRPO"
    teacher_model_name: str = "mkurman/Qwen2.5-14B-DeepSeek-R1-1M"
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    output_dir: str = "~/.models/"
    save_steps: int = 50
    learning_rate: float = 5e-7
    max_new_tokens: int = 4096
    num_return_sequences: int = 1
    accumulation_steps: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_seq_length: int = 4096
    cache_dir: str = "~/.cache/"
    warmup_steps: int = 100
    total_steps: int = 1000
    seed: int = 3409
    max_grad_norm: float = 0.1
    grpo_beta: float = 0.05
    sft_beta: float = 0.05

    def __post_init__(self):
        # Validate openai_base_url
        if not isinstance(self.openai_base_url, str):
            raise TypeError("openai_base_url must be a string")
        
        # Validate student_model_name
        if not isinstance(self.student_model_name, str):
            raise TypeError("student_model_name must be a string")
        
        # Validate teacher_model_name
        if not isinstance(self.teacher_model_name, str):
            raise TypeError("teacher_model_name must be a string")
        
        # Validate dataset_name
        if not isinstance(self.dataset_name, str):
            raise TypeError("dataset_name must be a string")
        
        # Validate output_dir
        if not isinstance(self.output_dir, str):
            raise TypeError("output_dir must be a string")
        self.output_dir = os.path.expanduser(self.output_dir)
        
        # Validate save_steps
        if not isinstance(self.save_steps, int) or self.save_steps <= 0:
            raise ValueError("save_steps must be a positive integer")
        
        # Validate learning_rate
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float")
        
        # Validate max_new_tokens
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        
        # Validate num_return_sequences
        if not isinstance(self.num_return_sequences, int) or self.num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be a positive integer")
        
        # Validate accumulation_steps
        if not isinstance(self.accumulation_steps, int) or self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        
        # Validate temperature
        if not isinstance(self.temperature, float) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")
        
        # Validate top_p
        if not isinstance(self.top_p, float) or not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be a float between 0 and 1")
        
        # Validate top_k
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        
        # Validate max_seq_length
        if not isinstance(self.max_seq_length, int) or self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer")
        
        # Validate cache_dir
        if not isinstance(self.cache_dir, str):
            raise TypeError("cache_dir must be a string")
        self.cache_dir = os.path.expanduser(self.cache_dir)
        
        # Validate warmup_steps
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        
        # Validate total_steps
        if not isinstance(self.total_steps, int) or self.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        
        # Validate seed
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        
        # Validate max_grad_norm
        if not isinstance(self.max_grad_norm, float) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be a positive float")
        
        # Validate grpo_beta
        if not isinstance(self.grpo_beta, float) or self.grpo_beta < 0:
            raise ValueError("grpo_beta must be a non-negative float")
        
        # Validate sft_beta
        if not isinstance(self.sft_beta, float) or self.sft_beta < 0:
            raise ValueError("sft_beta must be a non-negative float")