'''
@Date : 12.01.2023  @Author : Ozkan
'''
# !pip install missingno optuna xgboost lightgbm
import pandas as pd
import numpy as np
import warnings
import pickle
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.display.max_rows = 50
pd.options.display.max_columns = 50

def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Analyzing dataframe")
    print(df.isnull().sum() * 100 / len(df))
    print('---------------------------')
    print(df.duplicated().value_counts())
    print('-------------------------------')
    msno.bar(df)
    col = []
    d_type = []
    uniques = []
    n_uniques = []
    for i in df.columns:
        col.append(i)
        d_type.append(df[i].dtypes)
        uniques.append(df[i].unique()[:10])
        n_uniques.append(df[i].nunique())

    insight_df = pd.DataFrame({'Column': col, 'd_type': d_type, 'unique_values': uniques, 'n_uniques': n_uniques})

    return insight_df

def plot_data(df: pd.DataFrame) -> None:
    logger.info("Plotting data")
    plt.figure(figsize=(25,10))
    plt.xlabel('Campaign_Type hue with Audience_Age_Group')
    sns.histplot(x=df["Campaign_Type"],hue='Audience_Age_Group',data=df)

    plt.figure(figsize=(25,10))
    plt.xlabel('Audience_Size hue with Campaign_Type')
    sns.histplot(x=df["Audience_Size"],hue='Campaign_Type',data=df)

    plt.figure(figsize=(25,10))
    plt.xlabel('Duration hue with Campaign_Type')
    sns.histplot(x=df["Duration"],hue='Campaign_Type',data=df)

def plot_correlation(df: pd.DataFrame) -> None:
    logger.info("Plotting correlation matrix")
    corr_df = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
    plt.figure(figsize=(14, 14))
    ax = sns.heatmap(
        corr_df,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot = True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        horizontalalignment='right'
    )
    ax.set_ylim(len(corr_df)+0.5, -0.5)

def print_describe_and_boxplot(df: pd.DataFrame, column: str) -> None:
    logger.info(f"Describing and plotting boxplot for {column}")
    print(df[column].describe())
    print('-----------------\n')
    plt.boxplot(df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

def print_group_stats(df: pd.DataFrame, group_col: str) -> None:
    logger.info(f"Printing group stats for {group_col}")
    print(df.groupby(group_col).Budget.agg(["count", "mean", "median", "std", "min", "max"]))
    print('-----------------\n')

def scale_column(df: pd.DataFrame, column_name: str, scaler: MinMaxScaler) -> None:
    logger.info(f"Scaling column {column_name}")
    df[f'{column_name}_Scaled'] = scaler.fit_transform(df[[column_name]])
    print(df[f'{column_name}_Scaled'].describe())

def categorize_column(df: pd.DataFrame, column_name: str, categorize_function) -> None:
    logger.info(f"Categorizing column {column_name}")
    df[f'Category_of_{column_name}'] = df[column_name].apply(categorize_function)

def categorize_duration(duration: int) -> int:
    if duration <= 30:
        return 0 
    elif 30 < duration <= 60:
        return 1
    else: 
        return 2

def generate_category_of_Campaign(row: pd.Series) -> int:
    if row['Campaign_Type'] == 'Snapchat':
        return 1
    elif row['Campaign_Type'] == 'Instagram':
        return 2
    elif row['Campaign_Type'] == 'YouTube':
        return 3
    elif row['Campaign_Type'] == 'Facebook':
        return 4
    elif row['Campaign_Type'] == 'TikTok':
        return 5
    else:
        return 'Unknown'

def generate_category_of_Age(row: pd.Series) -> int:
    if row['Audience_Age_Group'] == '45-54':
        return 1
    elif row['Audience_Age_Group'] == '18-24':
        return 2
    elif row['Audience_Age_Group'] == '25-34':
        return 3
    elif row['Audience_Age_Group'] == '35-44':
        return 4
    elif row['Audience_Age_Group'] == '55-64':
        return 5
    else:
        return 'Unknown'

def one_hot_encode(df: pd.DataFrame, cols_to_encode: List[str]) -> pd.DataFrame:
    logger.info(f"One-hot encoding columns: {cols_to_encode}")
    df[cols_to_encode] = df[cols_to_encode].astype(str)
    return pd.get_dummies(df, columns=cols_to_encode)

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    logger.info("Training model")
    return LinearRegression().fit(X_train, y_train)

def evaluate_model(model: LinearRegression, X_val: pd.DataFrame, y_val: pd.Series) -> None:
    logger.info("Evaluating model")
    y_pred = model.predict(X_val)
    print(f'RMSE: {np.sqrt(mean_squared_error(y_val, y_pred))}')
    print(f'MAE: {mean_absolute_error(y_val, y_pred)}')

def cross_validate_model(model: LinearRegression, X: pd.DataFrame, y: pd.Series, cv: int, scoring) -> None:
    logger.info("Cross-validating model")
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f'Cross-validation mean score: {cv_scores.mean()}')

def plot_residuals(y_val: pd.Series, y_pred: np.ndarray) -> None:
    logger.info("Plotting residuals")
    residuals = y_val - y_pred
    plt.figure(figsize=(10,6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

def plot_feature_importances(model: LinearRegression, X: pd.DataFrame) -> None:
    logger.info("Plotting feature importances")
    importances = model.coef_
    importances_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    importances_df['abs_importance'] = importances_df['importance'].abs()
    importances_df = importances_df.sort_values('abs_importance', ascending=False).drop('abs_importance', axis=1)
    top_20_features = importances_df.head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x='importance', y='feature', data=top_20_features)
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def create_model(params: Dict) -> LGBMRegressor:
    logger.info("Creating model")
    return LGBMRegressor(**params)

def fit_model(model: LGBMRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    logger.info("Fitting model")
    model.fit(X_train, y_train)
    return model

def predict_model(model: LGBMRegressor, X_val: pd.DataFrame) -> np.ndarray:
    logger.info("Predicting with model")
    return model.predict(X_val)

def calculate_rmse(y_val: pd.Series, y_pred: np.ndarray) -> float:
    logger.info("Calculating RMSE")
    return np.sqrt(mean_squared_error(y_val, y_pred))

def main() -> None:
    logger.info("Starting main function")
    df = pd.read_csv('/app/marketing_campaign_data_socail_media.csv',encoding='utf-8')
    insight_df = analyze_dataframe(df)
    print(insight_df)
    plot_data(df)
    plot_correlation(df)
    plt.show()

    print_describe_and_boxplot(df, 'Engagement_Rate')
    print_describe_and_boxplot(df, 'Audience_Size')

    group_columns = ["Campaign_Type", "Audience_Age_Group"]
    for col in group_columns:
        print_group_stats(df, col)

    duratıon_rate = df.groupby('Campaign_Type')['Duration'].value_counts().to_frame().rename(columns={'Duration': 'Freq'}).reset_index().sort_values('Campaign_Type')
    print(duratıon_rate.head(10))
    print('---------\n')
    Audience_Age_Group_rate = df.groupby('Campaign_Type')['Audience_Age_Group'].value_counts().to_frame().rename(columns={'Audience_Age_Group': 'Freq'}).reset_index().sort_values('Campaign_Type')
    print(Audience_Age_Group_rate.head(10))
    
    scaler = MinMaxScaler(feature_range=(1, 10))
    scale_column(df, 'Audience_Size', scaler)
    df['Month_Of_Duration'] = df['Duration'].apply(categorize_duration)
    df['Category_of_Campaign_Type'] = df.apply(generate_category_of_Campaign, axis=1)
    df['Category_of_Audience_Age'] = df.apply(generate_category_of_Age, axis=1)

    numerical_values = ['Budget', 'Audience_Size', 'Engagement_Rate']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_data = poly.fit_transform(df[numerical_values])

    interaction_features = poly.get_feature_names_out(input_features=numerical_values)
    interaction_df = pd.DataFrame(interaction_data, columns=interaction_features)
    df = pd.merge(df, interaction_df, on='Budget', how='left')

    df['Duration_in_weeks'] = df['Duration'] / 7
    df['Duration_in_months'] = df['Duration'] / 30
    df['Duration_fraction_of_year'] = df['Duration'] / 365

    df['Budget_binned'] = pd.cut(df['Budget'], bins=10)
    df['Audience_Size_binned'] = pd.cut(df['Audience_Size_x'], bins=10)
    df['Engagement_Rate_binned'] = pd.cut(df['Engagement_Rate_x'], bins=10)

    df['Budget_log'] = np.log(df['Budget'])
    df['Audience_Size_log'] = np.log(df['Audience_Size_x'])
    df['Engagement_Rate_log'] = np.log(df['Engagement_Rate_x'])

    df['Campaign_Type_Audience_Age_Group'] = df['Campaign_Type'] + "_" + df['Audience_Age_Group']

    df['Budget_squared'] = df['Budget'] ** 2
    df['Audience_Size_squared'] = df['Audience_Size_x'] ** 2
    df['Engagement_Rate_squared'] = df['Engagement_Rate_x'] ** 2

    df['Budget_cubed'] = df['Budget'] ** 3
    df['Audience_Size_cubed'] = df['Audience_Size_x'] ** 3
    df['Engagement_Rate_cubed'] = df['Engagement_Rate_x'] ** 3

    df['Budget_inverse'] = 1 / df['Budget']
    df['Audience_Size_inverse'] = 1 / df['Audience_Size_x']
    df['Engagement_Rate_inverse'] = 1 / df['Engagement_Rate_x']
    df.drop(['Campaign_Type','Audience_Age_Group'],axis=1,inplace=True)

    cols_to_encode = ['Budget_binned', 'Audience_Size_binned', 'Engagement_Rate_binned','Campaign_Type_Audience_Age_Group']
    df[cols_to_encode] = df[cols_to_encode].astype(str)
    df = pd.get_dummies(df, columns=cols_to_encode)

    X = df.drop('Conversion_Rate', axis=1)
    y = df['Conversion_Rate']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)
    cross_validate_model(model, X, y, 5, make_scorer(mean_squared_error))
    plot_residuals(y_val, model.predict(X_val))
    plot_feature_importances(model, X)
    study = optuna.create_study(direction='minimize')

if __name__ == "__main__":
    main()