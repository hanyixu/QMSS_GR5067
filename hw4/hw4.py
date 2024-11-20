#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:39:45 2024

@author: xuhanyi
"""

from utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer



data = read_pickle("", "hw4")
t_form_data = xform_fun(data["body"], 1, 1, "tf", "")

chi_data, chi_m = chi_fun(t_form_data, data["label"],len(t_form_data.columns), "", "tf", 0.01) 


#parameters = {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}

parameters = {
    'max_depth': [3,5,7,10],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': [10, 20, 30 , 40],
    'min_samples_leaf': [1, 2, 4]
}

m = model_fun(chi_data, data.label, parameters, 0.80, "rf", "")


















"""
parameter = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}



model_grid = GridSearchCV(
    estimator=model,
    param_grid=parameter,
    scoring='accuracy',
    cv=2
)

model_grid.fit(X_train, y_train)


best_perf = model_grid.best_score_
print (best_perf)
best_params = model_grid.best_params_
print (best_params)
"""









