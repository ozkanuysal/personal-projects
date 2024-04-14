import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib
import datetime as dt


credits = pd.read_csv('ds_30_06_2020_V2.csv', sep=';')
features = credits.copy()
features = features.drop(['DayOfWeek', 'HeavyVehicleCount', 'HeavyVehicleExit', 'DayOfMonth', 'LuxBrandCount', 'LuxBrandExit'], axis=1)
#print(features.head(5))
features['T'] =  pd.to_datetime(features['T'],infer_datetime_format=True)
split_date = dt.datetime(2018,3,16)
test_split_date = dt.datetime(2020,9,1)

features = features.loc[features['T'] >= split_date]

print(features.describe())
#precentile
#print(features['Amount'].quantile(1))

#Region Z-Score Outliers Calculation

#features['zscore'] = np.abs((features.Amount - features.Amount.mean())/features.Amount.std(ddof=0))
features['zscore'] = (features.Amount - features.Amount.mean())/features.Amount.std(ddof=0)
#print(features.describe())

mean = features.Amount.mean() 
std = features.Amount.std(ddof=0)
#print('mean of the dataset is', mean) 
#print('std. deviation is', std) 
outliers = features.loc[features['zscore'] < -1.22]
outlierRestirected = features.loc[features['zscore'] <2]
print(outliers.head(50))
#outlierRestirected.sort_values(by=['Amount'], inplace=True)
#print(outlierRestirected.describe())
#plt.hist(outlierRestirected.Amount, bins=50, rwidth=0.6, density= True)
#plt.show()



#Region IQR Outliers Calculation
'''Q0 = features.Amount.quantile(0)
Q1 = features.Amount.quantile(0.25)
Q2 = features.Amount.quantile(0.50)
Q3 = features.Amount.quantile(0.75)
Q4 = features.Amount.quantile(1)
print(Q0)
print(Q1)
print(Q2)
print(Q3)
print(Q4)
'''