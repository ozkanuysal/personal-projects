import os


def get_data_path():
    """
    Get the path to the data directory.

    Returns:
    - str: Path to the data directory.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FILE_PATH = os.path.join(BASE_DIR.split("src")[0], "customer_purchases.csv")

    return FILE_PATH


def get_model_file_path():
    """
    Get the path to the data directory.

    Returns:
    - str: Path to the data directory.
    """
    BASE_DIR = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split("src")[0]),
                            "notebooks")

    return BASE_DIR