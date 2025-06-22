from pymongo import MongoClient
import bson
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection details
uri = os.getenv('CONNECTION_URI')

#?directConnection=true
client = MongoClient(uri)

# Database name
database_name = "deneme"

# Path to the directory containing the BSON files for each collection
backup_directory = "/media/py"

# Check if database exists
if database_name not in client.list_database_names():
    # Create the database
    client.admin.command({"create": database_name})

try:
    # Connect to the database
    db = client[database_name]
    # Delete all documents from all collections in the database
    for collection_name in db.list_collection_names():
        db[collection_name].delete_many({})

    # Iterate over each BSON file in the backup directory
    for file_name in os.listdir(backup_directory):
        if file_name.endswith(".bson"):
            collection_name = os.path.splitext(file_name)[0]
            backup_file_path = os.path.join(backup_directory, file_name)

            # Read BSON file and insert data into the collection
            with open(backup_file_path, "rb") as bson_file:
                backup_data = bson_file.read()
                collection = db[collection_name]
                collection.insert_many(bson.decode_all(backup_data))

            print(f"Collection '{collection_name}' restored successfully")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Close the MongoDB connection
    client.close()