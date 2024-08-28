from flask import render_template, request, jsonify
from flask import Flask
import flask
import numpy as np
import traceback
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
from datetime import date
# from config.db import DBClass
import db
import csv
import yaml, os
import json

# App definition
app = Flask(__name__,template_folder='templates')

def readConfig():
    _file = None
    with open ('/app/config/appsettings.json') as json_file:
        _file = json.load(json_file)
    return _file

dbInfo = readConfig()

lgbRegressor_Random = joblib.load('LGBMRegressor_Random_opt.pkl')


@app.route('/')
def welcome():
    return jsonify({"abount": "Hello Venus"})

@app.route('/HealthCheck')
def HealthCheck():
    env = dbInfo["env"]
    return jsonify({"Environment": env})

@app.route('/TodayPrediction', methods=['POST'])
def TodayPrediction():
    try:
        json_ = request.json
        daily_money_transfer = GetDailyMoneyTransferForToday(json_["T"])
        filled_train_data = FillTrainData(daily_money_transfer)
        trainset = pd.DataFrame([filled_train_data])
        prediction = lgbRegressor_Random.predict(trainset)
        vdf_prediction = ReadFromVdfPrediction(json_["T"])
        if not vdf_prediction:
            vdf_prediction = '0'
        return jsonify({  
            "Beklenen Tutar : ": str(round(float(prediction[0])))
        })

    except:
        return jsonify({
            "trace": traceback.format_exc()
            }) 

@app.route('/TodayPredictionData', methods=['POST'])
def TodayPredictionData():
    try:
        json_ = request.json
        daily_money_transfer = GetDailyMoneyTransferForToday(json_["T"])
        filled_train_data = FillTrainData(daily_money_transfer)
        return filled_train_data
    except:
        return jsonify({
            "trace": traceback.format_exc()
            }) 

@app.route('/ExpireDayPrediction', methods=['POST','GET'])
def ExpireDayPrediction():

   if flask.request.method == 'GET':
       return "Prediction page"
 
   if flask.request.method == 'POST':
       try:
           json_ = request.json
           daily_money_transfer = GetDailyMoneyTransfer(json_["T"])
           filled_train_data = FillTrainData(daily_money_transfer)
           trainset = pd.DataFrame([filled_train_data])
           prediction = lgbRegressor_Random.predict(trainset)
           vdf_prediction = ReadFromVdfPrediction(json_["T"])
           if not vdf_prediction:
               vdf_prediction = '0'
           return jsonify({
               "Amount": str(round(float(daily_money_transfer["Amount"]))),
               "VDF Prediction": str(round(float(vdf_prediction))),
               "LGBM Loss": str(CalculateLoss(float(daily_money_transfer["Amount"]) - float(prediction[0]), 2)),
               "LGBM Deviation %": str(CalculateDiffPersentaga(daily_money_transfer["Amount"], prediction[0])),
               "LGBM Deviation": str(round(float(daily_money_transfer["Amount"]) - float(prediction[0]))),
               "LGBM Prediction": str(round(float(prediction[0])))
           })

       except:
           return jsonify({
               "trace": traceback.format_exc()
               }) 

@app.route('/ExpireDayPredictionData', methods=['POST'])
def ExpireDayPredictionData():
    try:
        json_ = request.json
        daily_money_transfer = GetDailyMoneyTransfer(json_["T"])
        filled_train_data = FillTrainData(daily_money_transfer)
        return filled_train_data
    except:
        return jsonify({
            "trace": traceback.format_exc()
            })


@app.route('/uploadCSV', methods=['POST'])
def uploadCSV():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            return df.to_json(orient='records')
        else:
            return jsonify({"error": "File is not a CSV"}), 400
    except Exception as e:
        return jsonify({"trace": traceback.format_exc()}), 500


def CalculateDiffPersentaga(actual, predicted):
    return ((float(predicted) - float(actual)) / float(actual) * 100)

def CalculateLoss(deviation, lossPersent):
    return (abs(float(deviation)) * abs(float(lossPersent)) / 100)

def GetdateFromRow(row):
    try:
        data = {
            "T": row[0],
            "BeforDayCount": row[1],
            "T1_Amount": float(row[2]),
            "T2_Amount": float(row[3]),
            "T3_Amount": float(row[4]),
            "T4_Amount": float(row[5]),
            "DayOfWeek": row[6],
            "DayOfMonth": row[7],
            "IsEndOfMonth": row[8],
            "IsEndOfWeek": row[9],
            "WeekAVG": float(row[10]),
            "WMTFY_Count": row[11],
            "WMTFY_Amount": float(row[12]),
            "WMTBY_Count": row[13],
            "WMTBY_Amount": float(row[14]),
            "Month": row[15],
            "ActiveCampaingCount": row[18],
            "CampaignCreditCount": row[19],
            "OpenCampaingCount": row[20],
            "CloseCampaingCount": row[21],
            "T_trend": float(row[22]),
            "T1_trend": float(row[23]),
            "T2_trend": float(row[24]),
            "TrendAvg": float(row[25]),
            "ExchangeSellRate": float(row[26])
        }
    except:
        return jsonify({
            "trace": traceback.format_exc()
            }) 
    return data

def GetDailyMoneyTransfer(startdate):
    sql = 'exec vsp_sel_MoneyTransfer \''+ startdate + '\', \'' + startdate + '\''
    dbClass = db.DBClass()
    cursor = dbClass.GetOpenCurser(dbInfo)
    cursor.execute(sql)
    row = cursor.fetchone()
    data = ''
    while row:
        data = GetdateFromRow(row)
        row = cursor.fetchone()
    dbClass.CloseConnection()
    return data

def GetDailyMoneyTransferForToday(tomorrow):
    try:
        sql = 'exec vsp_sel_MoneyTransferForToday \''+ tomorrow +'\''
        dbClass = db.DBClass()
        cursor = dbClass.GetOpenCurser(dbInfo)
        cursor.execute(sql)
        row = cursor.fetchone()
        data = ''
        while row:
            data = GetdateFromRow(row)
            row = cursor.fetchone()
        dbClass.CloseConnection()
        return data
    except:
        return jsonify({
            "trace": traceback.format_exc()
            }) 

def FillTrainData(money_transfer_data):
    result = 'veri seti bos'
    if len(money_transfer_data) > 0 :
        result = {
                "BeforDayCount": money_transfer_data["BeforDayCount"],
                "T1_Amount": money_transfer_data["T1_Amount"],
                "T2_Amount": money_transfer_data["T2_Amount"],
                "T3_Amount": money_transfer_data["T3_Amount"],
                "T4_Amount": money_transfer_data["T4_Amount"],
                "DayOfWeek": money_transfer_data["DayOfWeek"],
                "DayOfMonth": money_transfer_data["DayOfMonth"],
                "IsEndOfMonth": money_transfer_data["IsEndOfMonth"],
                "IsEndOfWeek": money_transfer_data["IsEndOfWeek"],
                "WeekAVG": money_transfer_data["WeekAVG"],
                "WMTFY_Count": money_transfer_data["WMTFY_Count"],
                "WMTFY_Amount": money_transfer_data["WMTFY_Amount"],
                "WMTBY_Count": money_transfer_data["WMTBY_Count"],
                "WMTBY_Amount": money_transfer_data["WMTBY_Amount"],
                "Month": money_transfer_data["Month"],
                "ActiveCampaingCount": money_transfer_data["ActiveCampaingCount"],
                "CampaignCreditCount": money_transfer_data["CampaignCreditCount"],
                "OpenCampaingCount": money_transfer_data["OpenCampaingCount"],
                "CloseCampaingCount": money_transfer_data["CloseCampaingCount"],
                "T_trend": money_transfer_data["T_trend"],
                "T1_trend": money_transfer_data["T1_trend"],
                "T2_trend": money_transfer_data["T2_trend"],
                "TrendAvg": money_transfer_data["TrendAvg"],
                "ExchangeSellRate": money_transfer_data["ExchangeSellRate"],
            }
    return result

def WriteFile(data):
    with open('dataset.csv', mode='a', newline="\n") as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerow(data)

def ReadFromVdfPrediction(date):
    result = ''
    with open('ds_vdf_prediction.csv', mode='r') as dataset_file:
        reader = csv.reader(dataset_file, delimiter=';')
        for row in reader:
            if date in row:
                result = row[1]
    return result



if __name__ == "__main__":
    app.run(host="0.0.0.0",port = 8080)