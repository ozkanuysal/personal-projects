import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



tah = pd.read_csv("CustomerCollectionInfo_v2.csv",sep = ';')

# Date Manipulation
tah.Payment = pd.to_datetime(tah.Payment)
tah.sort_values("Payment", inplace = True)
tah.rename({"Payment":"Date"}, axis = 1, inplace = True)
tah.set_index("Date", inplace = True)

##############################
# OUTLIER: çalışması yapılmalı
tah["PaidAmount"].loc["2020-05-30"] = tah["PaidAmount"].loc["2020-05-30"]-80915015
tah["PaidAmount"].loc["2020-07-22"] = tah["PaidAmount"].loc["2020-07-22"]-15189667
tah["PaidAmount"].loc["2020-08-24"] = tah["PaidAmount"].loc["2020-08-24"]-10252200

tah.reset_index(inplace = True)



########## CALENDAR

#Adding the Calendar Public Holidays, Ramadan etc. from the Turkuaz dataset
calendar = pd.read_csv('DT_CALENDAR_CORRECTED_12082020.csv', sep = ';', parse_dates= ['CALENDAR_DATE'], dayfirst=True)
# Bayi verisine göre filtrelenmesi
# İlgili değişkenlerin seçimi
calendar = calendar[['CALENDAR_DATE', 'RAMADAN_FLAG', 'RELIGIOUS_DAY_FLAG_SK', 'NATIONAL_DAY_FLAG_SK', 'PUBLIC_HOLIDAY_FLAG', 
                     "WEEKEND_FLAG",'SEASON_SK', "DAY_OF_WEEK_SK", "QUARTER_OF_YEAR", "WEEK_OF_YEAR"]]


# LAbel Encoder
from sklearn.preprocessing import LabelEncoder
# Apply label encoder 
label_encoder = LabelEncoder()
cols = ["RAMADAN_FLAG", "PUBLIC_HOLIDAY_FLAG", "WEEKEND_FLAG"]
for col in cols:
    calendar[col] = label_encoder.fit_transform(calendar[col])
    
    
# BAYRAM ve RESMI TAILLERIN DUZENLENMESİ
calendar["DINI_BAYRAM"] = np.where(calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,151,152, 160,161,162,163]), 1, 0)
calendar["RESMI_BAYRAM"] = np.where(calendar.NATIONAL_DAY_FLAG_SK.isin([201, 203, 207, 208, 202, 204]), 1, 0)


calendar["RESMI_TATIL"] = np.where(
    (calendar.NATIONAL_DAY_FLAG_SK.isin([201, 203, 207, 208, 202, 204])) | (calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,151,152, 160,161,162,163])), 1, 0
)
calendar["RESMI_TATIL"] = np.where((calendar.CALENDAR_DATE.dt.month == 1) & (calendar.CALENDAR_DATE.dt.day == 1), 1, calendar["RESMI_TATIL"])


# AREFE: Ramazan, Kurban, Cumhuriyet Bayramı Arefeleri (Public Holiday Flag'de Kurban Arefesi olmadığından bunu ayrı oluşturdum)
calendar["AREFE"] = np.where(calendar.CALENDAR_DATE.isin(
    (calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,160])].CALENDAR_DATE - pd.DateOffset(1)).tolist()
    ), 1, 0
)

calendar["AREFE"] = np.where(calendar.CALENDAR_DATE.isin(
    (calendar[calendar.NATIONAL_DAY_FLAG_SK.isin([204])].CALENDAR_DATE - pd.DateOffset(1)).tolist()
    ), 1, calendar["AREFE"]
)


# BAYRAM TATIL ÖNCESİ GÜNLER
calendar["TATIL_ONCESI"+str(1)] = np.where(calendar.CALENDAR_DATE.isin(
        calendar[(calendar.RESMI_TATIL == 1) & (calendar.WEEKEND_FLAG == 0)].CALENDAR_DATE - pd.DateOffset(1)
        ),1,0                                
    )

calendar["TATIL_ONCESI"+str(1)] = np.where(calendar.CALENDAR_DATE.isin(
    calendar[(calendar.AREFE == 1) & (calendar["TATIL_ONCESI"+str(1)] == 1)].CALENDAR_DATE - pd.DateOffset(1)
),1,calendar["TATIL_ONCESI"+str(1)]                                
                                          )
    
calendar["TATIL_ONCESI"+str(1)] = np.where(calendar.CALENDAR_DATE.isin(
        calendar[(calendar.AREFE == 1) & (calendar["TATIL_ONCESI"+str(1)] == 1)].CALENDAR_DATE
        ), 0, calendar["TATIL_ONCESI"+str(1)]
    )

calendar["TATIL_ONCESI1"] = np.where(calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,151,152, 160,161,162,163]), 0, calendar.TATIL_ONCESI1)
 


calendar["TATIL_ONCESI2"] = np.where(
    calendar.CALENDAR_DATE.isin(
        calendar[(calendar["TATIL_ONCESI1"] == 1)].CALENDAR_DATE - pd.DateOffset(1)
    ), 1, 0)

calendar["TATIL_ONCESI2"] = np.where(calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,151,152, 160,161,162,163]), 0, calendar.TATIL_ONCESI2)

calendar["TATIL_ONCESI3"] = np.where(
    calendar.CALENDAR_DATE.isin(
        calendar[(calendar["TATIL_ONCESI2"] == 1)].CALENDAR_DATE - pd.DateOffset(1)
    ), 1, 0)
calendar["TATIL_ONCESI3"] = np.where(calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,151,152, 160,161,162,163]), 0, calendar.TATIL_ONCESI3)
    
    
calendar["TATIL_SONRASI1"] = np.where(calendar.CALENDAR_DATE.isin(
    calendar[(calendar.RESMI_TATIL == 1)& (calendar.WEEKEND_FLAG == 0)].CALENDAR_DATE + pd.DateOffset(1)
), 1,0)
calendar["TATIL_SONRASI1"] = np.where(calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,151,152, 160,161,162,163]), 0, calendar.TATIL_SONRASI1)
  
calendar["TATIL_SONRASI2"] = np.where(calendar.CALENDAR_DATE.isin(
    calendar[(calendar.TATIL_SONRASI1 == 1) & (calendar.WEEKEND_FLAG == 0)].CALENDAR_DATE + pd.DateOffset(1)
), 1,0)

calendar["TATIL_SONRASI3"] = np.where(calendar.CALENDAR_DATE.isin(
    calendar[(calendar.TATIL_SONRASI2 == 1) & (calendar.WEEKEND_FLAG == 0)].CALENDAR_DATE + pd.DateOffset(1)
), 1,0)


calendar['DAY'] = calendar['CALENDAR_DATE'].dt.day
calendar['MONTH'] = calendar['CALENDAR_DATE'].dt.month
calendar['YEAR'] = calendar['CALENDAR_DATE'].dt.year

calendar["AYIN_15"] = np.where(calendar.CALENDAR_DATE.dt.day.isin([15]), 1, 0)
calendar["AYIN_BASI"] = np.where(calendar.CALENDAR_DATE.dt.day.isin([1]), 1, 0)
calendar["AYIN_30"] = np.where(calendar.DAY == 30, 1, 0)

calendar.rename({"CALENDAR_DATE":"Date"}, axis = 1, inplace = True)

calendar.sort_values("Date", inplace = True)


cal_cols = ['Date', 'RAMADAN_FLAG',"DAY_OF_WEEK_SK","WEEKEND_FLAG",
    'DINI_BAYRAM', 'RESMI_BAYRAM', 'RESMI_TATIL', 'AREFE', 'TATIL_ONCESI1',
    'TATIL_ONCESI2', 'TATIL_ONCESI3', 'TATIL_SONRASI1', 'TATIL_SONRASI2',
    'TATIL_SONRASI3', 'DAY', 'MONTH', 'YEAR', 'AYIN_15', 'AYIN_BASI', "AYIN_30"]

tah = pd.merge(tah, calendar[cal_cols], how = "left", on = "Date")

########## FEATURE ENGINEERING

d = tah.copy()

###### HAFTASONLARININ YANSITILMASI

# V7 notebookunda paartesi de var
for i in ["Installment", "Capital"]:
    # 6. ve 7. günler toplanıp pazartesi ve cumaya yazdırılır
    weekend = d[d.DAY_OF_WEEK_SK == 6][["Date",i]].rename({"Date":"d6", i:i+"6"}, axis = 1)
    weekend = pd.concat([weekend, d[d.DAY_OF_WEEK_SK == 7][["Date",i]]], axis = 1)
    weekend[["Date",i]] = weekend[["Date",i]].shift(-1)
    weekend = weekend[weekend.d6.isnull() == False]
    weekend[i] = weekend[i].fillna(0)

    # Cumartesi Pazar toplanır
    weekend["WEEKEND_"+i.upper()] = weekend[i+"6"] + weekend[i]


    # Haftasonu cumaya yansıtılır
    weekend["Date2"] = weekend["d6"] - pd.DateOffset(1)
    
    # Pazartesi ve Cumaya yansıtılır ana veriyle
    d = pd.merge(d, weekend[["Date2", "WEEKEND_"+i.upper()]].rename({"WEEKEND_"+i.upper():"WEEKEND_"+i.upper()+"_FRI", "Date2":"Date"}, axis = 1), how = "left", on = "Date")
    d["WEEKEND_"+i.upper()+"_FRI"] = d["WEEKEND_"+i.upper()+"_FRI"] + d[i]
    d["WEEKEND_"+i.upper()+"_FRI"] = np.where(d["WEEKEND_"+i.upper()+"_FRI"].isnull() == True, d[i], d["WEEKEND_"+i.upper()+"_FRI"])

    
    
# PAZAR ve Cumartesi ayrı ayrı yansıtılır
day7 = d[d.DAY_OF_WEEK_SK == 7][["Date","Installment"]].rename({"Installment":"DAY7_INSTALLMENT"}, axis = 1)
day7["Date"] = day7["Date"] + pd.DateOffset(1)
d = pd.merge(d, day7, how = "left", on = "Date")
d["DAY7_INSTALLMENT"] = d.DAY7_INSTALLMENT + d.Installment
d["DAY7_INSTALLMENT"] = np.where(d.DAY7_INSTALLMENT.isnull() == True, d.Installment, d["DAY7_INSTALLMENT"])

day6 = d[d.DAY_OF_WEEK_SK == 6][["Date","Installment"]].rename({"Installment":"DAY6_INSTALLMENT"}, axis = 1)
day6["Date"] = day6["Date"] - pd.DateOffset(1)
d = pd.merge(d, day6, how = "left", on = "Date")
d["DAY6_INSTALLMENT"] = d.DAY6_INSTALLMENT + d.Installment
d["DAY6_INSTALLMENT"] = np.where(d.DAY6_INSTALLMENT.isnull() == True, d.Installment, d["DAY6_INSTALLMENT"])

######## TATIL SONRASI
for var in ["Installment", "Capital"]:
    
    d["TATIL_SONRASI_"+var] = np.nan
    
    # Günlere Göre
    for i in d[d.RESMI_BAYRAM == 1].Date:
        # Cumaya denk gelirse
        if ((pd.Series(pd.to_datetime(i)).dt.dayofweek == 4).loc[0] == True):
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(i) + pd.DateOffset(3))].index] = d[d.Date == i][var].values + d[d.Date == (pd.to_datetime(i) + pd.DateOffset(1))][var].values + d[d.Date == (pd.to_datetime(i) + pd.DateOffset(2))][var].values+ d[d.Date == (pd.to_datetime(i) + pd.DateOffset(3))][var].values
        # Cumartesi olursa
        elif ((pd.Series(pd.to_datetime(i)).dt.dayofweek == 5).loc[0] == True):
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(i) + pd.DateOffset(2))].index] = d[d.Date == i][var].values + d[d.Date == (pd.to_datetime(i) + pd.DateOffset(1))][var].values + d[d.Date == (pd.to_datetime(i) + pd.DateOffset(2))][var].values
        # Pazar olursa
        elif ((pd.Series(pd.to_datetime(i)).dt.dayofweek == 6).loc[0] == True):
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(i) + pd.DateOffset(1))].index] = d[d.Date == i][var].values + d[d.Date == (pd.to_datetime(i) + pd.DateOffset(1))][var].values + d[d.Date == (pd.to_datetime(i) - pd.DateOffset(1))][var].values
        # Diğer Günler
        else:
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(i) + pd.DateOffset(1))].index] = d[d.Date == i][var].values + d[d.Date == (pd.to_datetime(i) + pd.DateOffset(1))][var].values
    
    # Yılbaşı
    for j in d[(d.Date.dt.day == 1) & (d.Date.dt.month == 1)].Date:
        # Cumaya denk gelirse
        if ((pd.Series(pd.to_datetime(j)).dt.dayofweek == 4).loc[0] == True):
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(j) + pd.DateOffset(3))].index] = d[d.Date == j][var].values + d[d.Date == (pd.to_datetime(j) + pd.DateOffset(1))][var].values + d[d.Date == (pd.to_datetime(j) + pd.DateOffset(2))][var].values+ d[d.Date == (pd.to_datetime(j) + pd.DateOffset(3))][var].values
        # Cumartesi olursa
        elif ((pd.Series(pd.to_datetime(j)).dt.dayofweek == 5).loc[0] == True):
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(j) + pd.DateOffset(2))].index] = d[d.Date == j][var].values + d[d.Date == (pd.to_datetime(j) + pd.DateOffset(1))][var].values + d[d.Date == (pd.to_datetime(j) + pd.DateOffset(2))][var].values
        # Pazar olursa
        elif ((pd.Series(pd.to_datetime(j)).dt.dayofweek == 6).loc[0] == True):
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(j) + pd.DateOffset(1))].index] = d[d.Date == j][var].values + d[d.Date == (pd.to_datetime(j) + pd.DateOffset(1))][var].values + d[d.Date == (pd.to_datetime(j) - pd.DateOffset(1))][var].values
        # Diğer Günler
        else:
            d["TATIL_SONRASI_"+var].loc[d[d.Date == (pd.to_datetime(j) + pd.DateOffset(1))].index] = d[d.Date == j][var].values + d[d.Date == (pd.to_datetime(j) + pd.DateOffset(1))][var].values

    # Dini Bayramlar
        # Kurban Tarihleri ve Arefe Günü Seçilir
    kurban = (calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([160, 161, 162, 163])].Date.tolist())#+(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([160])].PaidDate - pd.DateOffset(1)).tolist()
    # Yıl değişkenine göre kurban bayramı günlerinin toplamı alınır
    kurban = d[d.Date.isin(kurban)][["Date", var]]
    kurban["CAT"] = pd.Series(np.arange(1,(kurban.shape[0] / 4)+1).tolist()*4).sort_values().values
    kurban = kurban.groupby("CAT")[var].sum().reset_index()


    # Kurbandan sonraki güne yazdırılır
    # Cuma
    kurban["Date"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date)].DAY_OF_WEEK_SK == 5,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(3))].Date, 
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(1))].Date                      
    )

    # Cumartesi
    kurban["Date"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date)].DAY_OF_WEEK_SK == 6,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(2))].Date, 
                                kurban["Date"])

    # Pazar
    kurban["Date"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date)].DAY_OF_WEEK_SK == 7,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(1))].Date, 
                                kurban["Date"])


    # Haftasonuna denk gelen günlerde haftasonu günleri ve ilk iş gününün op installment değerleri alınır ve toplanır
    kurban["T"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date)].DAY_OF_WEEK_SK == 5,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(3))][var].values + d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(2))][var].values + d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(1))][var].values, 
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(1))][var].values)

    kurban["T"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date)].DAY_OF_WEEK_SK == 6,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(2))][var].values + d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([163])].Date + pd.DateOffset(1))][var].values, 
            kurban["T"])

    # Kurban + Ertesi Günlerin Toplamı
    kurban[var] = kurban[var] + kurban["T"]

    kurban = kurban.drop(["CAT", "T"], axis = 1).rename({var:"TATIL_SONRASI_"+var+"2"}, axis = 1)

    
    # Ramazan Tarihleri ve Arefe Günü Seçilir
    ramazan = (calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([150,151,152])].Date.tolist())#+(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([160])].PaidDate - pd.DateOffset(1)).tolist()
    # Yıl değişkenine göre kurban bayramı günlerinin toplamı alınır
    ramazan = d[d.Date.isin(ramazan)][["Date", var]]
    ramazan["CAT"] = pd.Series(np.arange(1,(ramazan.shape[0] / 3)+1).tolist()*3).sort_values().values
    ramazan = ramazan.groupby("CAT")[var].sum().reset_index()


    
    # Kurbandan sonraki güne yazdırılır
    # Cuma
    ramazan["Date"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date)].DAY_OF_WEEK_SK == 5,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(3))].Date, 
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(1))].Date                      
)

    # Cumartesi
    ramazan["Date"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date)].DAY_OF_WEEK_SK == 6,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(2))].Date, 
                            ramazan["Date"])
    
    # Pazar
    ramazan["Date"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date)].DAY_OF_WEEK_SK == 7,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(1))].Date, 
                            ramazan["Date"])
    

    # Haftasonuna denk gelen günlerde haftasonu günleri ve ilk iş gününün op installment değerleri alınır ve toplanır
    ramazan["T"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date)].DAY_OF_WEEK_SK == 5,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(3))][var].values + d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(2))][var].values + d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(1))][var].values, 
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(1))][var].values)

    ramazan["T"] = np.where(d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date)].DAY_OF_WEEK_SK == 6,
            d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(2))][var].values + d[d.Date.isin(calendar[calendar.RELIGIOUS_DAY_FLAG_SK.isin([152])].Date + pd.DateOffset(1))][var].values, 
            ramazan["T"])

    # Kurban + Ertesi Günlerin Toplamı
    ramazan[var] = ramazan[var] + ramazan["T"]

    ramazan = ramazan.drop(["CAT", "T"], axis = 1).rename({var:"TATIL_SONRASI_"+var+"2"}, axis = 1)
    
    d = pd.merge(d, kurban.append(ramazan).reset_index(drop = True), how = "left", on = "Date")
    # Resmi Tatiller ile Dini Bayramlar Birleştirilir
    d["TATIL_SONRASI_"+var] = d["TATIL_SONRASI_"+var].fillna(0) 
    d["TATIL_SONRASI_"+var+"2"] =d["TATIL_SONRASI_"+var+"2"].fillna(0)

    # Dini Bayram ve Resmi Tatil çakışırsa
    d["TATIL_SONRASI_"+var] = np.where((d["TATIL_SONRASI_"+var] > 0) & (d["TATIL_SONRASI_"+var+"2"] > 0), d["TATIL_SONRASI_"+var+"2"], d["TATIL_SONRASI_"+var+"2"] + d["TATIL_SONRASI_"+var])

    d["TATIL_SONRASI_"+var] = np.where(d["TATIL_SONRASI_"+var] == 0, np.nan, d["TATIL_SONRASI_"+var])
    d["TATIL_SONRASI_"+var] = np.where(d["TATIL_SONRASI_"+var].isnull(), d[var], d["TATIL_SONRASI_"+var])
    d.drop("TATIL_SONRASI_"+var+"2", axis = 1, inplace = True)
    
    
###### GEÇMİŞ İSTATİSTİKLER    
# Ödeme Durumu Max: 28 Günlük
d["ODURUM_MAX"+str(28)] = (d.PaidAmount / d.TATIL_SONRASI_Installment).shift(1).rolling(28).max()
d["ODURUM_MEAN"+str(7)] = (d.PaidAmount / d.TATIL_SONRASI_Installment).shift(1).rolling(7).mean()
# Hareketli Ödeme Durumu: 7 Günlük
d["ROLL_INSTALMENT_RATIO"] = (d.PaidAmount.rolling(7).sum() / d.Installment.rolling(7).sum()).shift(1)    
d["ROLL_CAPITAL_RATIO"] = (d.PaidAmount.rolling(7).sum() / d.Capital.rolling(7).sum()).shift(1)    
# Installment & Capital Hareketli Özet İstatistikler
d["ROLL_OP_INSTALLMENT_MIN"+str(28)] = d.Installment.rolling(28).min()
d["ROLL_OP_INSTALLMENT_MEAN"+str(7)] = d.Installment.rolling(7).mean()
d["ROLL_OP_CAPITAL_MIN"+str(28)] = d.Capital.rolling(28).min()
d["ROLL_CAPITAL4"] = d.Capital.rolling(4).sum()
# Geçen Hafta
d.sort_values(["DAY_OF_WEEK_SK", "Date"], inplace = True)
    "DAY_OF_WEEK_SK", "RESMI_TATIL", "AYIN_15", "AYIN_30", "DAY",
d["GECEN_HAFTA_MAX4"] = d.groupby("DAY_OF_WEEK_SK").PaidAmount.rolling(4).max().shift(1).values
d.sort_values(["Date"], inplace = True)

# Ödeme Durumu Ortalamsı 7 günlük * Installment
d["MUL_ODURUM_MEAN7_OP_Ins"] = d.ODURUM_MEAN7 * d.Installment


#### GEREKSİZLERİ SİL
d.drop(["RAMADAN_FLAG",	"WEEKEND_FLAG", "DINI_BAYRAM", "RESMI_BAYRAM",	"AREFE",
"TATIL_ONCESI1", "TATIL_ONCESI2", "TATIL_ONCESI3", "TATIL_SONRASI1", "TATIL_SONRASI2",
"TATIL_SONRASI3", "MONTH", "YEAR", "AYIN_BASI", "ODURUM_MEAN7"], axis = 1, inplace = True)

### TRAIN - TEST
train = d[(d.Date < "2020-12-01")]
valid = d[(d.Date >="2020-12-01") & (d.Date < "2021-03-01")]

train.drop(["Date"], axis = 1, inplace = True)
valid.drop(["Date"], axis = 1, inplace = True)

y_train = train.PaidAmount 
y_valid = valid.PaidAmount


##### VADESİ GELENİ MODELDEN ÇIKART
x_train = train.drop(["PaidAmount"], axis = 1)
x_valid = valid.drop(["PaidAmount"], axis = 1)

cols = [
    # Ham Veri
    "Installment", "Capital",
    # Tatil Sonrası
    "TATIL_SONRASI_Installment",
    "TATIL_SONRASI_Capital",
    # Takvim
    # Haftasonu yansıtmaları
    "WEEKEND_INSTALLMENT_FRI",  
    "WEEKEND_CAPITAL_FRI", 
    "DAY7_INSTALLMENT", "DAY6_INSTALLMENT", 
    # Ödeme Durumu
    "ODURUM_MAX28", 
    # Ödeme Oranı
    "ROLL_INSTALMENT_RATIO", "ROLL_CAPITAL_RATIO", 
    # ROLL ÖP
    "ROLL_OP_INSTALLMENT_MIN28", 
    "ROLL_OP_INSTALLMENT_MEAN7",
    "ROLL_OP_CAPITAL_MIN28",
    "ROLL_CAPITAL4", 
    "MUL_ODURUM_MEAN7_OP_Ins",
    "GECEN_HAFTA_MAX4",  
]


x_train = x_train[cols]
x_valid = x_valid[cols]


######### MODEL: LGBM
#import lightgbm as lgb

# Tüm Veri
trainLGB = lgb.Dataset(x_train, label = y_train)
validLGB = lgb.Dataset(x_valid, label = y_valid)

lgb_param = {
    'nthread': -1,
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
    "max_depth":7,
    "num_leaves":16,
    "learning_rate":0.05, #98
    "max_bin":150,
    #"feature_fraction":1,
    "min_data_in_leaf":20,#25
    "lambda_l1":1,
    "lambda_l2":0.3,
 
}


evals_result = {}

clf = lgb.train(params = lgb_param, train_set=trainLGB, num_boost_round=50000, 
          valid_sets=[trainLGB, validLGB], valid_names=["Train", "Valid"], verbose_eval=1000, 
          early_stopping_rounds = 1000, evals_result=evals_result)

from joblib import dump
# save your model or results
dump(clf, 'mt_lgbm_20210303.pkl')

valid_res = d[(d.Date >="2020-12-01") & (d.Date < "2021-03-01")]
valid_res["PRED"] = clf.predict(x_valid)
