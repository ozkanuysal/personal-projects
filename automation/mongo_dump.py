import subprocess
import os
from dotenv import load_dotenv

load_dotenv()


# Create a client connection to your MongoDB instance
uri = os.getenv('CONNECTION_URI')

# Define your database name, and output directory
database_name = 'deneme'
output_dir = os.getenv('OUTPUT_DIR')

# Construct the mongodump command
command = f'mongodump --uri {uri} --db {database_name} --out {output_dir}'

# Run the command
subprocess.run(command, shell=True, check=True)