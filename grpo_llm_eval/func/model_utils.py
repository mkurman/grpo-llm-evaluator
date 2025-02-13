from unsloth import FastLanguageModel

def load_student_model(config, load_in_4bit: bool=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.student_model_name,
        load_in_4bit=load_in_4bit,
        fast_inference=False,
        max_seq_length=config.max_seq_length,
        cache_dir=config.cache_dir,
    )
    return tokenizer, model


def load_teacher_model(config):
    return config.teacher_model_name
