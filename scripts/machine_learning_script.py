import os
import pandas as pd
import numpy as np
import sklearn.model_selection import train_test_split


df = pd.read_csv(os.path.join('..','data','healthcare-dataset-stroke-data.csv'))
df = df.dropna

df_dum = pd.getdummies(df)

X = df_dum.drop('smoking_status', axis=1)
y = df.smoking_status.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state=42)
y_train, y_test = y_train.ravel(), y_test.ravel()


