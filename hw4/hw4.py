#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:39:45 2024

@author: xuhanyi
"""


# =============================================================================
"""
file path:
    -> data (folder)
        -> hw4.pk
    -> output (folder)
        -> rf.pk
        -> gnb.pk
    -> hw4.py (this file)
    -> utils.py
"""
# =============================================================================



from utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

the_path = "data/"
o_path = "output/"

def model_fun(df_in, lab_in, g_in, t_s, sw_in, p_o):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import GridSearchCV
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, lab_in, test_size=t_s, random_state=42)
    
    if sw_in == "rf":
        model = RandomForestClassifier(random_state=123)
        text = str("Random Forest Classifier")
    elif sw_in == "gnb":
        model = GaussianNB()
        text = str("Gaussian Naive Bayes")
    
    clf = GridSearchCV(model, g_in)
    clf.fit(X_train, y_train)
    
    best_perf = clf.best_score_
    print ("The best score for " + text + " model: " + str(best_perf))
    best_params = clf.best_params_
    print ("The best parameter for " + text + " model: " + str(best_params))
    
    if sw_in == "rf":
        model = RandomForestClassifier(random_state=110, **best_params)
    elif sw_in == "gnb":
        model = GaussianNB(**best_params)
    
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
        X_test, y_test, test_size=0.10, random_state=110)
    
    model.fit(X_train_val, y_train_val)
    write_pickle(model, p_o, sw_in)
    y_pred = model.predict(X_test_val)
    y_pred_likelihood = pd.DataFrame(
        model.predict_proba(X_test_val))
    y_pred_likelihood.columns = model.classes_
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test_val, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "fscore", None]
    
    #feature importance
    try:
        feat_imp = pd.DataFrame(model.feature_importances_)
        feat_imp.index = X_train_val.columns
        feat_imp.columns = ["score"]
        feat_imp.to_csv(p_o + sw_in + "_m.csv")
        perc_prop = len(feat_imp[feat_imp["score"] > 0]) / len(feat_imp) * 100
        print (perc_prop)
    except:
        print ("Not transparent")
        pass
    return model, metrics

data = read_pickle(the_path, "hw4")
t_form_data = xform_fun(data["body"], 1, 6, "tf", "")

chi_data, chi_m = chi_fun(t_form_data, data["label"],len(t_form_data.columns), "", "tf", 0.01) 


#Run models
sw = "rf"
parameters_rf = {"n_estimators": [50, 100], "max_depth": [None, 10]}
model_rf, metrics_rf = model_fun(chi_data, data["label"], parameters_rf, 0.80, sw, o_path)
print("\n")
sw = "gnb"
parameters_gnb = {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}
model_gnb, metrics_gnb = model_fun(chi_data, data["label"], parameters_gnb, 0.80, sw, o_path)
print("\n")
print("Random Forest Classifier metrics: ", metrics_rf)
print("\n")
print("Gaussian Naive Bayes metrics: ", metrics_gnb)

#It turns out that Gaussian Naive Bayes model has higher score, when I run the code.