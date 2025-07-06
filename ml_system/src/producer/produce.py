import argparse
import json
import os
from time import sleep

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from schema_registry.client import SchemaRegistryClient, schema

load_dotenv()

OUTPUT_TOPICS = os.getenv("KAFKA_OUTPUT_TOPICS", "tracking.raw_user_behavior")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:9092")
SCHEMA_REGISTRY_SERVER = os.getenv(
    "KAFKA_SCHEMA_REGISTRY_URL", "http://schema-registry:8081"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--mode",
    default="setup",
    choices=["setup", "teardown"],
    help="Whether to setup or teardown a Kafka topic with driver stats events. Setup will teardown before beginning emitting events.",
)
parser.add_argument(
    "-b",
    "--bootstrap_servers",
    default=BOOTSTRAP_SERVERS,
    help="Where the bootstrap server is",
)
parser.add_argument(
    "-s",
    "--schema_registry_server",
    default=SCHEMA_REGISTRY_SERVER,
    help="Where to host schema",
)
parser.add_argument(
    "-c",
    "--avro_schemas_path",
    default=os.path.join(os.path.dirname(__file__), "avro_schemas"),
    help="Folder containing all generated avro schemas",
)

args = parser.parse_args()

# Define some constants
NUM_DEVICES = 1


def create_topic(admin, topic_name):
    # Create topic if not exists
    try:
        # Create Kafka topic
        topic = NewTopic(name=topic_name, num_partitions=12, replication_factor=1)
        admin.create_topics([topic])
        print(f"A new topic {topic_name} has been created!")
    except Exception:
        print(f"Topic {topic_name} already exists. Skipping creation!")
        pass


def create_streams(servers, avro_schemas_path, schema_registry_client):
    producer = None
    admin = None

    # Add retry logic for Kafka connection
    for _ in range(10):
        try:
            producer = KafkaProducer(
                bootstrap_servers=servers,
                value_serializer=str.encode,  # Simple string encoding
                batch_size=16384,  # Increase batch size (default 16384)
                buffer_memory=33554432,  # 32MB buffer memory
                compression_type="gzip",  # Enable compression
                linger_ms=50,  # Wait up to 50ms to batch messages
                acks=1,  # Only wait for leader acknowledgment
            )
            admin = KafkaAdminClient(bootstrap_servers=servers)
            print("SUCCESS: instantiated Kafka admin and producer")
            break
        except Exception as e:
            print(
                f"Trying to instantiate admin and producer with bootstrap servers {servers} with error {e}"
            )
            sleep(10)
            pass

    # Add retry logic for schema registry
    for _ in range(10):
        try:
            schema_registry_client.get_subjects()
            print("SUCCESS: connected to schema registry")
            break
        except Exception as e:
            print(
                f"Failed to connect to schema registry: {e}. Retrying in 10 seconds..."
            )
            sleep(10)
    else:
        raise Exception("Failed to connect to schema registry after 10 attempts")

    # Load the Avro schema
    avro_schema_path = f"{avro_schemas_path}/ecommerce_events.avsc"
    with open(avro_schema_path, "r") as f:
        parsed_avro_schema = json.loads(f.read())

    # Load data and prepare for batch processing
    try:
        df = pd.read_parquet("data/sample.parquet")
        print(f"Loaded {len(df)} records from parquet file")

        # Pre-format all records in parallel
        def format_record(row):
            index = row.name  # Get the index from the row
            record = {
                "event_time": str(row["event_time"]),
                "event_type": str(row["event_type"]),
                "product_id": int(row["product_id"]),
                "category_id": int(row["category_id"]),
                "category_code": str(row["category_code"])
                if pd.notnull(row["category_code"])
                else None,
                "brand": str(row["brand"]) if pd.notnull(row["brand"]) else None,
                "price": float(row["price"]) if index % 10 != 0 else -100,
                "user_id": int(row["user_id"]),
                "user_session": str(row["user_session"]),
            }

            formatted_record = {
                "schema": {"type": "struct", "fields": parsed_avro_schema["fields"]},
                "payload": record,
            }
            return json.dumps(formatted_record)

        # Process records in parallel using all available CPU cores
        print("Formatting records in parallel...")
        records = df.apply(format_record, axis=1).tolist()
        print(f"Formatted {len(records)} records")

    except Exception as e:
        print(f"Error loading/processing parquet file: {e}")
        return

    # Get topic name and create it if needed
    topic_name = OUTPUT_TOPICS
    create_topic(admin, topic_name=topic_name)

    # Register schema if needed
    schema_version_info = schema_registry_client.check_version(
        f"{topic_name}-schema", schema.AvroSchema(parsed_avro_schema)
    )
    if schema_version_info is not None:
        schema_id = schema_version_info.schema_id
        print(f"Found existing schema ID: {schema_id}. Skipping creation!")
    else:
        schema_id = schema_registry_client.register(
            f"{topic_name}-schema", schema.AvroSchema(parsed_avro_schema)
        )
        print(f"Registered new schema with ID: {schema_id}")

    # Batch send records
    print("Starting to send records...")
    for i, record in enumerate(records):
        producer.send(topic_name, value=record)

        # Only print progress every 1000 records
        if i % 1000 == 0:
            print(f"Sent {i} records")

        sleep(0.05)

    # Make sure all messages are sent
    producer.flush()
    print(f"Finished sending {len(records)} records")


def teardown_stream(topic_name, servers=["localhost:9092"]):
    try:
        admin = KafkaAdminClient(bootstrap_servers=servers)
        print(admin.delete_topics([topic_name]))
        print(f"Topic {topic_name} deleted")
    except Exception as e:
        print(str(e))
        pass


if __name__ == "__main__":
    parsed_args = vars(args)
    mode = parsed_args["mode"]
    servers = parsed_args["bootstrap_servers"]
    schema_registry_server = parsed_args["schema_registry_server"]

    # Tear down all previous streams
    print("Tearing down all existing topics!")
    for device_id in range(NUM_DEVICES):
        try:
            teardown_stream(OUTPUT_TOPICS, [servers])
        except Exception as e:
            print(f"Topic device_{device_id} does not exist. Skipping...! {e}")

    if mode == "setup":
        avro_schemas_path = parsed_args["avro_schemas_path"]
        schema_registry_client = SchemaRegistryClient(url=schema_registry_server)
        create_streams([servers], avro_schemas_path, schema_registry_client)