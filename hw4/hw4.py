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
import time


#=========timer=========
start_time_1 = time.time()
#=========timer=========


data = read_pickle("", "hw4")
t_form_data = xform_fun(data["body"], 1, 3, "tf", "")

chi_data, chi_m = chi_fun(t_form_data, data["label"],len(t_form_data.columns), "", "tf", 0.01) 

parameters = {
    'max_depth': [3,5,7,10],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': [10, 20, 30 , 40],
    'min_samples_leaf': [1, 2, 4]
}


X_train, X_test, y_train, y_test = train_test_split(
    chi_data, data.label, test_size=0.75, random_state=110)

model = RandomForestClassifier(random_state=110)
    
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)
    
best_perf = clf.best_score_
print (best_perf)
best_params = clf.best_params_
print (best_params)
    
model = RandomForestClassifier(random_state=110, **best_params)

#=========timer=========
end_time_1 = time.time()
print("First timer: ")
print(end_time_1 - start_time_1)
start_time_2 = time.time()
#=========timer=========

#========done gridsearch========

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_likelihood = pd.DataFrame(model.predict_proba(X_test))
y_pred_likelihood.columns = model.classes_


metrics = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
metrics.index = ["precision", "recall", "fscore", None]


#=========timer=========
end_time_2 = time.time()
print("Second timer: ")
print(end_time_2 - start_time_2)
#=========timer=========











