#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
%matplotlib inline
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

pd.options.display.float_format = '{:.0f}'.format

df = pd.read_csv('train.csv')
df.head()

df.describe()

print len(df['timestamp'].unique())
print len(df['KPI ID'].unique())

df['label'].value_counts()

sns.countplot(x="KPI ID", data=df)

keys = df['KPI ID'].unique()
values = range(1, 27)
mydic = dict(zip(keys, values))

df['KPI ID'] = df['KPI ID'].apply(lambda x: mydic[x])
print df.tail()

total = len(df)
size = int(total * 0.3)
rows = random.sample(df.index, size)
df_test = df.ix[rows]
df_train = df.drop(rows)
print df_test.shape
print df_train.shape

y = df_train['label']
X = df_train.drop('label', axis=1)

yy = df_test['label']
XX = df_test.drop('label', axis=1)

# Logistic Regression
clf_0 = LogisticRegression().fit(X, y)

pred_y_0 = clf_0.predict(XX)
print(accuracy_score(yy, pred_y_0))

prob_y_0 = clf_0.predict_proba(XX)
prob_y_0 = [p[1] for p in prob_y_0]
print(roc_auc_score(yy, prob_y_0))

# Random Rorest
clf_2 = RandomForestClassifier()
clf_2.fit(X, y)

pred_y_2 = clf_2.predict(XX)
print(accuracy_score(yy, pred_y_2))

prob_y_2 = clf_2.predict_proba(XX)
prob_y_2 = [p[1] for p in prob_y_2]
print(roc_auc_score(yy, prob_y_2))

# XGBoost
clf_3 = XGBClassifier()
clf_3.fit(X, y)

pred_y_3 = clf_3.predict(XX)
print(accuracy_score(yy, pred_y_3))

prob_y_3 = clf_3.predict_proba(XX)
prob_y_3 = [p[1] for p in prob_y_3]
print(roc_auc_score(yy, prob_y_3))