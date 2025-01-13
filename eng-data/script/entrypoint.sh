#!/bin/bash

set -e

if [ -e "/opt/airflow/requirements.txt" ]; then
    $(command -v pip) installl --user -r requirements.txt
fi

if [ ! -f "/opt/airflow/airflow.db" ]; then
    airflow db init &&
    airflow users create \
        --username admin \
        --firstname admin \
        --lastname admin \
        --role Admin \
        --email
fi

