from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "zkn439",
    "start_date": datetime(2025, 1, 12, 14, 00)
}

def get_data():
    import requests


    res = requests.get("https://randomuser.me/api/") # it has 2 keys: info and results we need to take only results
    res = res.json()["results"][0]
    return res

def format_data(res):
    data = {}
    location = res["location"]
    data["first_name"] = res["name"]["first"]
    data["last_name"] = res["name"]["last"]
    data["gender"] = res["gender"]
    #location = res["location"]
    data["address"] = f"{str(location['street']['number'])} {location['street']['name']} {location['city']} {location['state']} {location['country']}"
    data["postcode"] = location["postcode"]
    data["email"] = res["email"]
    data["username"] = res["login"]["username"]
    data["dob"] = res["dob"]["date"]
    data["registered_date"] = res["registered"]["date"]
    data["phone"] = res["phone"]
    data["picture"] = res["picture"]["medium"]

    return data

def stream_data():
    import json
    from kafka import KafkaProducer
    import time

    res = get_data()
    res = format_data(res)
    #print(json.dumps(res, indent=3))

    producer = KafkaProducer(bootstrap_servers=["localhost:9092"], max_block_ms=5000)
    producer.send("users_created", json.dumps(res).encode("utf-8"))


# with DAG("user_automation",
#             default_args=default_args,
#             schedule_interval="@daily",
#             catchup=False) as dag:
    
#     streaming_task = PythonOperator(
#         "task_id": "stream_data_from_api",
#         python_callable=stream_data
#     )

stream_data()