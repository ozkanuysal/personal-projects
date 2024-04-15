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
import lightgbm

random_state=42
n_iter=50

credits = pd.read_csv('2020_Eylul.csv', sep=',')
features = credits.copy()
features = features.drop([
                          #'DayOfWeek', 
                          #'HeavyVehicleCount',
                          #'HeavyVehicleExit',
                         #'DayOfMonth',
                          #'Month',
                          'QuarterOfYear',
                          #'IsEndOfWeek',
                          #'IsEndOfMonth',
                          #'ActiveCampaingCount', 
                          #'CampaignCreditCount', 
                          #'OpenCampaingCount',
                          #'CloseCampaingCount',
                         # 'ExchangeSellRate',  
                         # 'LuxBrandCount', 
                        #'LuxBrandExit',
                        #'LuxBrandCount',
                        'WeekOfYear'
                        #'CurrencyDate'
                        ], axis=1)
#print(features.head(5))
features['T'] =  pd.to_datetime(features['T'],infer_datetime_format=True)
split_date = pd.datetime(2017,7,3)
test_split_date = pd.datetime(2020,9,5)

features = features.loc[features['T'] >= split_date]

#print(features.describe())
x_training = features.loc[features['T'] <= test_split_date]
x_test = features.loc[features['T'] > test_split_date]

train_data = x_training.iloc[:,1:-1].values 
test_data =  x_test.iloc[:,1:-1].values 

train_targets = x_training.iloc[:,-1].values
test_targets = x_test.iloc[:,-1].values 

num_folds=3
kf = KFold(shuffle=True, n_splits=num_folds, random_state=random_state)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(4, 32, num = 2)]
#max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'num_leaves': [20],
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = LGBMRegressor()
randomModel = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=1, random_state=42, n_jobs = -1)

# Fit the random search model
randomModel.fit(train_data,train_targets)
print("BEST PARAMETERS: " + str(randomModel.best_params_))
print("BEST CV SCORE: " + str(randomModel.best_score_))

y_pred = randomModel.predict(test_data)

# summarize the fit of the model
print(); print(metrics.r2_score(test_targets, y_pred))
print(); print(metrics.mean_squared_log_error(test_targets, y_pred))

#plt.figure(figsize=(10,10))
#sns.regplot(test_targets, y_pred, fit_reg=True, scatter_kws={"s": 100})

print(); print(test_targets)
print(); print(y_pred.astype(int))


#Feature importance for top 50 predictors
predictors = [x for x in features.columns]
predictors.remove('T')
predictors.remove('Amount')
feat_imp = pd.Series(randomModel.best_estimator_.feature_importances_, predictors).sort_values(ascending=False)
feat_imp = feat_imp[0:50]
plt.rcParams['figure.figsize'] = 18, 5
feat_imp.plot(kind='bar', title='Feature Importance')
plt.ylabel('Feature Importance Score')
#plt.show()

#plotImp(clf, X_test)
#plotImp(randomModel, x_test)


#joblib.dump(randomModel, 'LGBMRegressor_Random_opt.pkl')
#gbm_pickle = joblib.load('LGBMRegressor_Random_opt.pkl')