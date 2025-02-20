from datasets import load_dataset


def load_dataset_function(config):
    """
    Loads a dataset from Hugging Face Datasets.

    Args:
        config: The configuration object containing dataset information.

    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(
        config.dataset_name, split="train", cache_dir=config.cache_dir
    )
    return dataset
