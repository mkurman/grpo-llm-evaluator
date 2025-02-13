import yaml
from grpo_llm_eval.config import TrainingConfig


def load_config(yaml_file):
    """
    Loads a configuration from a YAML file.

    Args:
        yaml_file (str): The path to the YAML configuration file.

    Returns:
        TrainingConfig: The loaded training configuration.
    """
    with open(yaml_file, "r") as f:
        config_data = yaml.safe_load(f)
    return TrainingConfig(**config_data)
