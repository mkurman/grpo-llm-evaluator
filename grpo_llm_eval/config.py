from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TrainingConfig:
    """
    Configuration class for training.
    """

    use_unsloth: bool = True
    """Whether to use unsloth for fast inference."""
    openai_base_url: str = "http://localhost/v1/"
    """Base URL for the OpenAI API."""
    student_model_name: str = "mkurman/Llama-3.2-MedIT-SUN-2.5B-BT-GRPO"
    """Name or path of the student model."""
    teacher_model_name: str = "mkurman/Qwen2.5-14B-DeepSeek-R1-1M"
    """Name or path of the teacher model."""
    load_in_4bit: bool = True
    """Whether to load the model in 4-bit precision."""
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    """Name of the dataset to use."""
    dataset_question_column: str = "question"
    """Column name for the questions in the dataset."""
    dataset_answer_column: str = "answer"
    """Column name for the answers in the dataset."""
    output_dir: str = "~/.models/"
    """Directory to save the training outputs."""
    save_steps: int = 50
    """Number of steps between each save."""
    learning_rate: float = 5e-7
    """Learning rate for the optimizer."""
    max_new_tokens: int = 4096
    """Maximum number of new tokens to generate."""
    max_feedback_new_tokens: int = 4096
    """Maximum number of new tokens to generate for feedback."""
    num_return_sequences: int = 2
    """Number of sequences to return from the generator."""
    accumulation_steps: int = 1
    """Number of accumulation steps."""
    temperature: float = 0.7
    """Temperature for sampling."""
    top_p: float = 0.9
    """Top p value for sampling."""
    top_k: int = 50
    """Top k value for sampling."""
    repetition_penalty: float = 1.1
    """Repetition penalty for sampling."""
    max_seq_length: int = 4096
    """Maximum sequence length."""
    cache_dir: str = "~/.cache/"
    """Directory to cache the datasets and models."""
    warmup_steps: int = 100
    """Number of warmup steps."""
    total_steps: int = 1000
    """Total number of training steps."""
    seed: int = 3409
    """Random seed."""
    max_grad_norm: float = 0.1
    """Maximum gradient norm."""
    grpo_beta: float = 0.05
    """Beta value for GRPO loss."""
    sft_beta: float = 0.05
    """Beta value for SFT loss."""
    thought_process_weight: float = 0.07
    """Weight for the thought process score in the reward."""
    answer_weight: float = 0.1
    """Weight for the answer score in the reward."""
    format_weight: float = 0.03
    """Weight for the format score in the reward."""

    system_prompt: Optional[str] = (
        "Respond in the following format:\n<think>\n...\n</think>\n...\n$\\boxed{answer}$"
    )
    """System prompt to use for generation."""

    evaluation_prompt: Optional[str] = (
        "You are a teacher evaluating a student's answer. Evaluate the following student's response in two parts:\n1) Correctness of the thought process (rated from 1 to 10), and\n2) Correctness of the final answer (rated from 1 to 10).\n3) Format and clarity of the response (rated from 1 to 10). Student must have <think> and </think> tags included! If not, give 0 points for this assesment.\n\nThink about the student's thought process and the final answer using <think> ... </think> tags.\nProvide detailed feedback after your thinking process using the following XML format:\n<evaluation>\n  <thought_process><score>{score}</score><explanation>{explanation}</thought_process>\n  <answer><score>{score}</score><explanation>{explanation}</answer>\n  <format><score>{score}</score><explanation>{explanation}</format>\n</evaluation>\n\nStudent Response:\n{student_response}\nGround Truth:\n{ground_truth}\n"
    )

    """Evaluation prompt to use for the teacher model."""

    think_open_string: str = "<think>"
    """String to indicate the start of the thought process."""
    think_close_string: str = "</think>"
    """String to indicate the end of the thought process."""

    custom_model: Optional[str] = None
    """Custom model to use for training."""

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

        # Validate dataset_question_column
        if not isinstance(self.dataset_question_column, str):
            raise TypeError("dataset_question_column must be a string")

        # Validate dataset_answer_column
        if not isinstance(self.dataset_answer_column, str):
            raise TypeError("dataset_answer_column must be a string")

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

        # Validate max_feedback_new_tokens
        if (
            not isinstance(self.max_feedback_new_tokens, int)
            or self.max_feedback_new_tokens <= 0
        ):
            raise ValueError("max_feedback_new_tokens must be a positive integer")

        # Validate num_return_sequences
        if (
            not isinstance(self.num_return_sequences, int)
            or self.num_return_sequences <= 1
        ):
            raise ValueError("num_return_sequences must be an integer greater than 1")

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

        # Validate repetition_penalty
        if (
            not isinstance(self.repetition_penalty, float)
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

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

        # Validate thought_process_weight
        if (
            not isinstance(self.thought_process_weight, float)
            or self.thought_process_weight < 0
        ):
            raise ValueError("thought_process_weight must be a non-negative float")

        # Validate answer_weight
        if not isinstance(self.answer_weight, float) or self.answer_weight < 0:
            raise ValueError("answer_weight must be a non-negative float")

        # Validate format_weight
        if not isinstance(self.format_weight, float) or self.format_weight < 0:
            raise ValueError("format_weight must be a non-negative float")

        # Validate openai_base_url
        if not isinstance(self.openai_base_url, str):
            raise TypeError("openai_base_url must be a string")

        # Validate thought_process_weight
        if (
            not isinstance(self.thought_process_weight, float)
            or self.thought_process_weight < 0
        ):
            raise ValueError("thought_process_weight must be a non-negative float")

        # Validate answer_weight
        if not isinstance(self.answer_weight, float) or self.answer_weight < 0:
            raise ValueError("answer_weight must be a non-negative float")

        # Validate format_weight
        if not isinstance(self.format_weight, float) or self.format_weight < 0:
            raise ValueError("format_weight must be a non-negative float")

        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string")

        if self.evaluation_prompt is not None and not isinstance(
            self.evaluation_prompt, str
        ):
            raise TypeError("evaluation_prompt must be a string")

        if "{student_response}" not in self.evaluation_prompt:
            raise ValueError("evaluation_prompt must contain {student_response}")

        if "{ground_truth}" not in self.evaluation_prompt:
            raise ValueError("evaluation_prompt must contain {ground_truth}")

        if not isinstance(self.think_open_string, str):
            raise TypeError("think_open_string must be a string")

        if not isinstance(self.think_close_string, str):
            raise TypeError("think_close_string must be a string")

        # Ensure output_dir exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Ensure cache_dir exists
        os.makedirs(self.cache_dir, exist_ok=True)
