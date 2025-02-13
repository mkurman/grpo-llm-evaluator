from datasets import load_dataset

def load_dataset_function(config):
    dataset = load_dataset(
        config.dataset_name, split="train", cache_dir=config.cache_dir
    )
    return dataset


def apply_chat_template(example):
    return f"<think>{example['input']}<think>"
