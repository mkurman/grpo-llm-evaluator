from datasets import load_dataset, load_from_disk


def load_dataset_function(config):
    """
    Loads a dataset from Hugging Face Datasets.

    Args:
        config: The configuration object containing dataset information.

    Returns:
        Dataset: The loaded dataset.
    """
    try:
        dataset = load_dataset(
            config.dataset_name, split="train", cache_dir=config.cache_dir
        )
    except:
        dataset = load_from_disk(config.dataset_name)
    return dataset
