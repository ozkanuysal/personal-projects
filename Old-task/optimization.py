import pandas as pd
import numpy as np
import joblib
import json

def prediction(labor_code: None, personals: None):
    '''
    It must be only one work_order_id
    labor_code: It must be list -str
    personals : It must be list -str
    '''
    
    X_train_dtypes = {'LABOR_CODE_R': np.dtype('int64'), 
     'BRAND_DEFINITION_R': np.dtype('int64'),
     'standart_duration': np.dtype('float64'),
     'EDUCATION_STATUS_EXPLANATION': np.dtype('int64'),
     'job_name': np.dtype('int64'),
     'FIRM_CODE': np.dtype('int64'),
     'COMING_TYPE_EXPLANATION': np.dtype('int64'),
     'SERVICE_MILEAGE': np.dtype('float64'),
     'MUSTERI_BEKLIYOR_MU': np.dtype('int64'),
     'MOTOR_GAS_TYPE': np.dtype('int64'),
     'ortalama_net_sure_30': np.dtype('float64'),
     'ortalama_net_sure_90': np.dtype('float64'),
     'ortalama_net_sure_150': np.dtype('float64'),
     'ortalama_net_sure_240': np.dtype('float64'),               
     'max_30': np.dtype('float64'),
     'max_90': np.dtype('float64'),
     'max_150': np.dtype('float64'),
     'max_240': np.dtype('float64'),
     'adet_30': np.dtype('float64'),
     'adet_90': np.dtype('float64'),
     'adet_150': np.dtype('float64'),
     'adet_240': np.dtype('float64'),
     'personel_tecrube': np.dtype('int64'),
     'arac_sinif': np.dtype('int64')}
    
    
    drop_cols = ['BASEMODEL_DEFINITION', 'CLASS_CODE', 'DEFINITION_R', 'DELIVERY_DATE', 'FK_PERSONNEL_ID',
                 'START_DATETIME_R', 'START_OF_WORKDATE', 'TOPMODEL_DEFINITION', 'WORK_ORDER_PLANNING_DETAIL_ID_R',
                 'net_calisma_suresi']
    
    atama_tablo = pd.DataFrame()
    path = '/home/ec2-user/SageMaker/ATOLYE_PLANLAMA/' # your path
    clf = joblib.load(path + "fonk_atolye.pkl") # model
    
    with open('encode_transform.json') as f: #mapping
        encode_transform = json.load(f)
    
    with open('inverse_encode_transform.json') as f: # inverse mapping
        inverse_encode_transform = json.load(f)    
    
    df = pd.read_csv(path + "her_isemri_ile_gelecek_veri.csv", sep = "|") # read work_order and require columns (From DataBase/Turkuaz) (df_merve den 1 tane iş emri filtreyebilirisinz)
    model_data = pd.read_csv(path + "opt_icin_hazir_df.csv", sep = "|").sort_values(by="START_DATETIME_R") # full df (feat eng)
    
    df["arac_sinif"] = df["BRAND_DEFINITION_R"].astype("str") + "-" + df["TOPMODEL_DEFINITION"].astype("str") + "-" + df["MOTOR_GAS_TYPE"].astype("str")

    for col in ['LABOR_CODE_R', 'BRAND_DEFINITION_R',
           'COMING_TYPE_EXPLANATION', 'FIRM_CODE',
           'SERVICE_MILEAGE', 'MUSTERI_BEKLIYOR_MU', 
           'MOTOR_GAS_TYPE',"arac_sinif"]:
        
        df[col] = df[col].replace(np.NAN, np.NaN)
        df[col] = df[col].replace(pd.np.nan, np.NaN)
        df[col] = df[col].replace(np.nan, np.NaN)
        df[col].fillna(value= "missing", inplace=True)
        df[col] = df[col].astype("str")  
        model_data[col] = model_data[col].astype("str")
        
        
    for labor in labor_code:
        for personel in personals:
            final_tablo = pd.DataFrame()
            personel_require_for_modelling = None
            X_future = None
            tahmin_df = None
            preds_tahmin_df = None
            
            final_tablo.loc[final_tablo.shape[0] + 1,"LABOR_CODE_R"] = str(labor)
            final_tablo.loc[final_tablo.shape[0], "personel"] = str(personel)
            personel_require_for_modelling = df[(df["LABOR_CODE_R"] == labor)].drop_duplicates(keep = "first").reset_index(drop=True)
            
            
            try:
                # feat eng yapılmış veriyle birleştirme
                X_future = model_data[(model_data["FK_PERSONNEL_ID"] == int(personel)) &
                                    (model_data["LABOR_CODE_R"] == labor) &
                                    (model_data["arac_sinif"] == personel_require_for_modelling["arac_sinif"].iloc[0]) &
                                    (model_data["standart_duration"] == int(personel_require_for_modelling["standart_duration"]))]


                X_future = X_future.sort_values(by="START_DATETIME_R").tail(1)
            except:
                X_future = pd.DataFrame()
                
            if (X_future.shape[0] > 0):                
                tahmin_df = pd.merge(X_future, personel_require_for_modelling[["LABOR_CODE_R","standart_duration","FIRM_CODE","arac_sinif"]], 
                                 how = "left", 
                                 on = ["LABOR_CODE_R","arac_sinif","standart_duration","FIRM_CODE"]).drop(drop_cols, axis = 1).reset_index(drop=True)
                
                
                tahmin_df = tahmin_df.replace(encode_transform)
                tahmin_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in tahmin_df.columns]
                tahmin_df = tahmin_df.astype(X_train_dtypes)
                preds_tahmin_df = clf.predict(tahmin_df)
                final_tablo.loc[final_tablo.shape[0], "tahmin"] = int(preds_tahmin_df)
                
                
            else:
                try:
                    #eğer tahmin yapamıyorsak standart süre atanır (daha önce bu personel bu işçiliği bu araba üzerinde yapmadıysa)
                    final_tablo.loc[final_tablo.shape[0], "tahmin"] = int(personel_require_for_modelling["standart_duration"]) 
                except:
                    final_tablo.loc[final_tablo.shape[0], "tahmin"] = int(999) # kapsam dışı 

                
            atama_tablo = pd.concat([atama_tablo,final_tablo])
            
    
    return atama_tablo
