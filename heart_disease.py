import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings('ignore') 

data = pd.read_csv("heart-disease.csv")

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.18, random_state=0
)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

y_preds = log_reg.predict(X_test)
y_preds_proba = log_reg.predict_proba(X_test)

print(y_preds)

pickle.dump(log_reg, open('model-heart-disease.pkl', 'wb'))

model = pickle.load(open('model-heart-disease.pkl', 'rb'))