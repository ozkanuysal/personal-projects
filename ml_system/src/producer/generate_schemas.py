import argparse
import json
import os
import random
import shutil
from typing import Dict

import numpy as np


def generate_schema(num_features: int, sampled_features: list) -> Dict:
    """Generate a single Avro schema dictionary."""
    schema = {
        "doc": "Sample schema to help you get started.",
        "fields": [
            {"name": "device_id", "type": "int"},
            {"name": "created", "type": "string"},
        ],
        "name": "Device",
        "namespace": "example.avro",
        "type": "record",
    }
    for feature_idx in sampled_features:
        schema["fields"].append({"name": f"feature_{feature_idx}", "type": "float"})
    return schema


def prepare_schema_folder(folder_path: str):
    """Clean and create the schema folder."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def main(args: Dict):
    prepare_schema_folder(args["schema_folder"])
    for schema_idx in range(args["num_schemas"]):
        num_features = np.random.randint(args["min_features"], args["max_features"])
        sampled_features = random.sample(range(args["max_features"]), num_features)
        schema = generate_schema(num_features, sampled_features)
        schema_path = os.path.join(args["schema_folder"], f"schema_{schema_idx}.avsc")
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num_schemas", default=1, type=int,
        help="Number of avro schemas to generate."
    )
    parser.add_argument(
        "-m", "--min_features", default=2, type=int,
        help="Minimum number of features for each device"
    )
    parser.add_argument(
        "-a", "--max_features", default=10, type=int,
        help="Maximum number of features for each device"
    )
    parser.add_argument(
        "-o", "--schema_folder", default="./avro_schemas",
        help="Folder containing all generated avro schemas"
    )