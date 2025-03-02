from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_student_model(config, load_in_4bit: bool = False):
    if config.use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            config.student_model_name,
            load_in_4bit=load_in_4bit,
            fast_inference=False,
            max_seq_length=config.max_seq_length,
            cache_dir=config.cache_dir,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name, load_in_4bit=load_in_4bit
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def load_teacher_model(config):
    return config.teacher_model_name
