from celery import Celery
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

app = Celery('tasks', broker='amqp://guest@localhost//')

@app.task
def train_model(data_path, target_column, model_params):
    """Train a random forest classifier model"""
    df = pd.read_csv(data_path)

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    model.save('model.pkl')
    with open('metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.3f}')

    return accuracy