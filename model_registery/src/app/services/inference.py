import json
import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from ..services.report import report_service
from ..crud.settings import settings


class InferenceService:

    def __init__(self):
        """
        Initialize InferenceService with config and model as None.
        """
        self.config = None
        self.model = None

    def load_config(self, config_file_path):
        """
        Load configuration from a JSON file and remove the file after loading.

        Parameters:
        config_file_path (str): The path to the JSON configuration file.

        Returns:
        None
        """
        with open(config_file_path, "r") as file:
            self.config = json.load(file)

        os.remove(config_file_path)

    def load_model(self, model_file_path):
        """
        Load a trained model from a pickle file and remove the file after loading.

        Parameters:
        model_file_path (str): The path to the pickle file containing the trained model.

        Returns:
        None
        """
        self.model = pd.read_pickle(model_file_path)
        os.remove(model_file_path)

    def train_inference(self, data):
        """
        Placeholder for training the inference model.

        Parameters:
        data (DataFrame): The training data.

        Returns:
        None
        """
        pass

    def report_inference(self, model_path, artifact_path):
        """
        Generate a report using the trained model and configuration.

        Parameters:
        model_path (str): The path to the pickle file containing the trained model.
        artifact_path (str): The path to the JSON configuration file.

        Returns:
        None
        """
        self.load_config(artifact_path)
        self.load_model(model_path)
        engine = create_engine(settings.DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        result = pd.read_sql(self.get_inference_query(params={}), session.bind, params={})
        # Retrieves the latest record of each customer.
        result = result.sort_values(by="purchase_date_id", ascending=True).drop_duplicates(subset=["customer_id"],
                                                                                           keep="last")

        result["next_month_purchase_amount"] = self.model.predict(
            result[self.config["feat_cols"]])
        report_service.execute_report_pipeline(result)

    def test_inference(self, params, model_path, artifact_path):
        """
        Test the inference model with given parameters.

        Parameters:
        params (dict): The parameters for inference.
        model_path (str): The path to the pickle file containing the trained model.
        artifact_path (str): The path to the JSON configuration file.

        Returns:
        float: The predicted value for the given parameters.
        """
        self.load_config(artifact_path)
        self.load_model(model_path)

        engine = create_engine(settings.DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        result = pd.read_sql(self.get_inference_query(params), session.bind, params=params)
        data = result[self.config["feat_cols"]]

        pred = self.model.predict(data)
        return float(pred)

    def get_inference_query(self, params):
        """
        Generate a SQL query for inference based on the given parameters.

        Parameters:
        params (dict): The parameters for inference.

        Returns:
        text: The SQL query for inference.
        """
        base_query = """
            WITH all_combinations AS (
                SELECT c.customer_id, 
                       d.purchase_date_id
                FROM (SELECT DISTINCT customer_id FROM customer) AS c
                CROSS JOIN (SELECT DISTINCT date_trunc('month', purchase_date) AS purchase_date_id FROM transaction) AS d
            ),
            agg_data AS (
                SELECT 
                    customer_id,
                    date_trunc('month', purchase_date) AS purchase_date_id,
                    SUM(purchase_amount) AS total_purchase_amount,
                    AVG(purchase_amount) AS mean_purchase_amount,
                    MAX(purchase_amount) AS max_purchase_amount,
                    MIN(purchase_amount) AS min_purchase_amount,
                    COUNT(purchase_amount) AS count_purchase_amount
                FROM transaction
                GROUP BY customer_id, purchase_date_id
            ),
            combined_data AS (
                SELECT 
                    ac.customer_id,
                    ac.purchase_date_id,
                    COALESCE(ad.total_purchase_amount, 0) AS purchase_amount,
                    COALESCE(ad.mean_purchase_amount, 0) AS purchase_amount_mean,
                    COALESCE(ad.max_purchase_amount, 0) AS purchase_amount_max,
                    COALESCE(ad.min_purchase_amount, 0) AS purchase_amount_min,
                    COALESCE(ad.count_purchase_amount, 0) AS purchase_amount_count
                FROM all_combinations AS ac
                LEFT JOIN agg_data AS ad ON ac.customer_id = ad.customer_id 
                                          AND ac.purchase_date_id = ad.purchase_date_id
            ),
            cumulative_data AS (
                SELECT *,
                       SUM(COALESCE(purchase_amount_count, 0)) OVER (PARTITION BY customer_id ORDER BY purchase_date_id) AS total_transaction,
                       LEAD(purchase_amount) OVER (PARTITION BY customer_id ORDER BY purchase_date_id) AS next_month_purchase_amount
                FROM combined_data
            )
            SELECT c.*, 
                   cd.*
            FROM customer AS c
            LEFT JOIN cumulative_data AS cd ON c.customer_id = cd.customer_id
        """

        where_clauses = []
        if "customer_id" in params:
            where_clauses.append("cd.customer_id = :customer_id")
        if "purchase_date_id" in params:
            where_clauses.append("cd.purchase_date_id = :purchase_date_id")

        for key in params:
            if key not in ["customer_id", "purchase_date_id"]:
                where_clauses.append(f"cd.{key} = :{key}")

        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)

        base_query += " ORDER BY c.customer_id, cd.purchase_date_id;"

        return text(base_query)

inference_service = InferenceService()