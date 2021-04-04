import os
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron


xgboost_json = os.path.join('..','models','xgboost_stroke_pred_model.json')
svm_sav = os.path.join('..','models','SVC().sav')
random_forest_sav = os.path.join('..','models','RandomForestClassifier().sav')
bernoulli_sav = os.path.join('..','models','BernoulliNB().sav')
perceptron_sav = os.path.join('..','models','Perceptron().sav')

def load_XGBoost_model(xgboost_json):
    xgb = XGBClassifier()
    xgb_model = xgb.load_model(xgboost_json)
    return xgb_model

def load_sklearn_model(model_path):
    return pickle.load(open(model_path, 'rb'))

xgb_model = load_XGBoost_model(xgboost_json)
svm_model = load_sklearn_model(svm_sav)
random_forest_model = load_sklearn_model(random_forest_sav)
bernoulli_model = load_sklearn_model(bernoulli_sav)
perceptron_model = load_sklearn_model(perceptron_sav)