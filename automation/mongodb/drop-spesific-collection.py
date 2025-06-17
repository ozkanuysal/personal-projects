from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

connection_variable = os.getenv('CONNECTION_URI1')

# Step 1: Connect to MongoDB

uri = connection_variable

client = MongoClient(uri) 

db = client['deneme']  

# Step 3: Access the specific collection
collection = db['processlogs']  

# Step 4: Delete all documents in the collection
result = collection.delete_many({})

# Optional: Print the result
print(f"Deleted {result.deleted_count} documents from the collection.")

# Verify the collection is empty but still exists
if collection.count_documents({}) == 0:
    print("All documents deleted, collection still exists.")
else:
    print("Failed to delete all documents.")