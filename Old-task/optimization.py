import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import joblib
import json
import logging

# Constants
BASE_PATH = Path('/home/ec2-user/SageMaker/ATOLYE_PLANLAMA/')
MODEL_FILE = 'fonk_atolye.pkl'
ENCODE_FILE = 'encode_transform.json'
INVERSE_ENCODE_FILE = 'inverse_encode_transform.json'
WORK_ORDER_FILE = 'her_isemri_ile_gelecek_veri.csv'
OPT_DATA_FILE = 'opt_icin_hazir_df.csv'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_encodings(base_path: Path) -> tuple:
    """Load model and encoding files."""
    try:
        model = joblib.load(base_path / MODEL_FILE)
        with open(base_path / ENCODE_FILE) as f:
            encode_transform = json.load(f)
        with open(base_path / INVERSE_ENCODE_FILE) as f:
            inverse_encode_transform = json.load(f)
        return model, encode_transform, inverse_encode_transform
    except Exception as e:
        logger.error(f"Error loading model/encodings: {e}")
        raise

def load_and_prepare_data(base_path: Path) -> tuple:
    """Load and prepare input data."""
    df = pd.read_csv(base_path / WORK_ORDER_FILE, sep="|")
    model_data = pd.read_csv(base_path / OPT_DATA_FILE, sep="|").sort_values(by="START_DATETIME_R")
    
    df["arac_sinif"] = df["BRAND_DEFINITION_R"].astype("str") + "-" + \
                       df["TOPMODEL_DEFINITION"].astype("str") + "-" + \
                       df["MOTOR_GAS_TYPE"].astype("str")
    
    columns_to_process = ['LABOR_CODE_R', 'BRAND_DEFINITION_R',
                         'COMING_TYPE_EXPLANATION', 'FIRM_CODE',
                         'SERVICE_MILEAGE', 'MUSTERI_BEKLIYOR_MU', 
                         'MOTOR_GAS_TYPE', "arac_sinif"]
    
    for col in columns_to_process:
        df[col] = df[col].fillna("missing").astype("str")
        model_data[col] = model_data[col].astype("str")
    
    return df, model_data

def predict_single_case(labor: str, 
                       personel: str, 
                       df: pd.DataFrame, 
                       model_data: pd.DataFrame,
                       clf,
                       encode_transform: Dict,
                       X_train_dtypes: Dict) -> pd.DataFrame:
    """Make prediction for a single labor-personnel combination."""
    final_tablo = pd.DataFrame()
    final_tablo.loc[0, "LABOR_CODE_R"] = str(labor)
    final_tablo.loc[0, "personel"] = str(personel)
    
    personel_require = df[df["LABOR_CODE_R"] == labor].drop_duplicates(keep="first").reset_index(drop=True)
    
    try:
        X_future = get_future_data(personel, labor, model_data, personel_require)
        if not X_future.empty:
            prediction = make_prediction(X_future, personel_require, clf, encode_transform, X_train_dtypes)
            final_tablo.loc[0, "tahmin"] = int(prediction)
        else:
            final_tablo.loc[0, "tahmin"] = get_fallback_duration(personel_require)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        final_tablo.loc[0, "tahmin"] = 999
    
    return final_tablo

def prediction(labor_code: List[str], personals: List[str]) -> pd.DataFrame:
    """Main prediction function."""
    try:
        clf, encode_transform, _ = load_model_and_encodings(BASE_PATH)
        df, model_data = load_and_prepare_data(BASE_PATH)
        
        atama_tablo = pd.DataFrame()
        for labor in labor_code:
            for personel in personals:
                result = predict_single_case(labor, personel, df, model_data, 
                                          clf, encode_transform, X_train_dtypes)
                atama_tablo = pd.concat([atama_tablo, result])
        
        return atama_tablo
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise