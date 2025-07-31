import argparse
import json
import os
from time import sleep
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from schema_registry.client import SchemaRegistryClient, schema

load_dotenv()

DEFAULT_OUTPUT_TOPICS = os.getenv("KAFKA_OUTPUT_TOPICS", "tracking.raw_user_behavior")
DEFAULT_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:9092")
DEFAULT_SCHEMA_REGISTRY_SERVER = os.getenv(
    "KAFKA_SCHEMA_REGISTRY_URL", "http://schema-registry:8081"
)
NUM_DEVICES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", default="setup", choices=["setup", "teardown"],
        help="Whether to setup or teardown a Kafka topic with driver stats events."
    )
    parser.add_argument(
        "-b", "--bootstrap_servers", default=DEFAULT_BOOTSTRAP_SERVERS,
        help="Where the bootstrap server is"
    )
    parser.add_argument(
        "-s", "--schema_registry_server", default=DEFAULT_SCHEMA_REGISTRY_SERVER,
        help="Where to host schema"
    )
    parser.add_argument(
        "-c", "--avro_schemas_path",
        default=os.path.join(os.path.dirname(__file__), "avro_schemas"),
        help="Folder containing all generated avro schemas"
    )
    return parser.parse_args()


class KafkaManager:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.producer: Optional[KafkaProducer] = None
        self.admin: Optional[KafkaAdminClient] = None

    def connect(self, retries: int = 10, delay: int = 10):
        for attempt in range(retries):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.servers,
                    value_serializer=str.encode,
                    batch_size=16384,
                    buffer_memory=33554432,
                    compression_type="gzip",
                    linger_ms=50,
                    acks=1,
                )
                self.admin = KafkaAdminClient(bootstrap_servers=self.servers)
                print("SUCCESS: Connected to Kafka admin and producer")
                return
            except Exception as e:
                print(f"Attempt {attempt+1}: Kafka connection failed: {e}")
                sleep(delay)
        raise RuntimeError("Failed to connect to Kafka after retries")

    def create_topic(self, topic_name: str):
        try:
            topic = NewTopic(name=topic_name, num_partitions=12, replication_factor=1)
            self.admin.create_topics([topic])
            print(f"Created topic: {topic_name}")
        except Exception:
            print(f"Topic {topic_name} already exists. Skipping creation.")

    def delete_topic(self, topic_name: str):
        try:
            self.admin.delete_topics([topic_name])
            print(f"Deleted topic: {topic_name}")
        except Exception as e:
            print(f"Failed to delete topic {topic_name}: {e}")

    def send_records(self, topic_name: str, records: List[str]):
        print("Starting to send records...")
        for i, record in enumerate(records):
            self.producer.send(topic_name, value=record)
            if i % 1000 == 0:
                print(f"Sent {i} records")
            sleep(0.05)
        self.producer.flush()
        print(f"Finished sending {len(records)} records")


class SchemaRegistryManager:
    def __init__(self, url: str):
        self.client = SchemaRegistryClient(url=url)

    def connect(self, retries: int = 10, delay: int = 10):
        for attempt in range(retries):
            try:
                self.client.get_subjects()
                print("SUCCESS: Connected to schema registry")
                return
            except Exception as e:
                print(f"Attempt {attempt+1}: Schema registry connection failed: {e}")
                sleep(delay)
        raise RuntimeError("Failed to connect to schema registry after retries")

    def register_schema(self, subject: str, avro_schema: dict) -> int:
        avro_schema_obj = schema.AvroSchema(avro_schema)
        version_info = self.client.check_version(subject, avro_schema_obj)
        if version_info is not None:
            print(f"Found existing schema ID: {version_info.schema_id}. Skipping creation!")
            return version_info.schema_id
        schema_id = self.client.register(subject, avro_schema_obj)
        print(f"Registered new schema with ID: {schema_id}")
        return schema_id


def load_avro_schema(avro_schemas_path: str, schema_file: str = "ecommerce_events.avsc") -> dict:
    avro_schema_path = os.path.join(avro_schemas_path, schema_file)
    with open(avro_schema_path, "r") as f:
        return json.loads(f.read())


def load_and_format_records(avro_schema: dict, parquet_path: str = "data/sample.parquet") -> List[str]:
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} records from parquet file")

        def format_record(row):
            index = row.name
            record = {
                "event_time": str(row["event_time"]),
                "event_type": str(row["event_type"]),
                "product_id": int(row["product_id"]),
                "category_id": int(row["category_id"]),
                "category_code": str(row["category_code"]) if pd.notnull(row["category_code"]) else None,
                "brand": str(row["brand"]) if pd.notnull(row["brand"]) else None,
                "price": float(row["price"]) if index % 10 != 0 else -100,
                "user_id": int(row["user_id"]),
                "user_session": str(row["user_session"]),
            }
            formatted_record = {
                "schema": {"type": "struct", "fields": avro_schema["fields"]},
                "payload": record,
            }
            return json.dumps(formatted_record)

        print("Formatting records in parallel...")
        records = df.apply(format_record, axis=1).tolist()
        print(f"Formatted {len(records)} records")
        return records
    except Exception as e:
        print(f"Error loading/processing parquet file: {e}")
        return []


def main():
    args = parse_args()
    mode = args.mode
    servers = [args.bootstrap_servers]
    schema_registry_server = args.schema_registry_server
    avro_schemas_path = args.avro_schemas_path
    topic_name = DEFAULT_OUTPUT_TOPICS

    kafka_manager = KafkaManager(servers)
    kafka_manager.connect()

    schema_manager = SchemaRegistryManager(schema_registry_server)
    schema_manager.connect()

    # Always teardown before setup for a clean state
    print("Tearing down all existing topics!")
    for device_id in range(NUM_DEVICES):
        kafka_manager.delete_topic(topic_name)

    if mode == "setup":
        avro_schema = load_avro_schema(avro_schemas_path)
        records = load_and_format_records(avro_schema)
        if not records:
            print("No records to send. Exiting.")
            return

        kafka_manager.create_topic(topic_name)
        schema_manager.register_schema(f"{topic_name}-value", avro_schema)
        kafka_manager.send_records(topic_name, records)
    elif mode == "teardown":
        kafka_manager.delete_topic(topic_name)
        print("Teardown complete. All topics deleted.") 