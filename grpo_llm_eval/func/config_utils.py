import yaml
from grpo_llm_eval.config import TrainingConfig

def load_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config_data = yaml.safe_load(f)
    return TrainingConfig(**config_data)
