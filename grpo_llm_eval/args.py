import argparse


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="GRPO LLM Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()
