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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()
