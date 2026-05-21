#LOGISTIC REGRESSION USING SKLEARN

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pprint
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('C06-IMAGINE/Task 2/b/breast-cancer.csv')
df['diagnosis'] = np.where(df['diagnosis'] == 'M', 1, 0)

corr = df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, cmap='mako_r',annot=True)                      #gives correlation plot
plt.show()

cor_target = abs(corr["diagnosis"])
relevant_features = cor_target[cor_target > 0.2]                 #makes a list of features which are highly correlated
names =  relevant_features.index.tolist()
names.remove('diagnosis')
print(names)

X = df[names].values
y = df['diagnosis'].values                                       #converts datafram into numpy array

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
