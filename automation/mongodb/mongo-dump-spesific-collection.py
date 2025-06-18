import os
from dotenv import load_dotenv

import subprocess

load_dotenv()
connection_variable = os.getenv('CONNECTION_URI')


# MongoDB URI with authentication details
mongo_uri = connection_variable
mongo_db = 'mrkay'
mongo_collection = 'test'
output_directory = '/media/py'

# Construct the mongodump command
command = [
    'mongodump',
    '--uri', mongo_uri,
    '--db', mongo_db,
    '--collection', mongo_collection,
    '--out', output_directory
]

try:
    # Execute the command
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Collection {mongo_collection} from database {mongo_db} has been dumped to {output_directory}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e.stderr.decode('utf-8')}")