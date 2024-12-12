# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:22:40 2024

@author: pathouli
"""

from utils import *
import pandas as pd
o_path = "C:/Users/pathouli/Box Sync/myStuff/academia/columbia/socialSciences/GR5067/2024_fall/output/"

"""
steps:
1. clean_text
2. rem_sw
3. call up the vectorizer and transform 
4. call up chi function and transform
5. call up the fitted model, and input the vector and predict
"""

sample_text = "Mathematics is the science and study of quality, structure, space, and change. Mathematicians seek out patterns, formulate new conjectures, and establish truth by rigorous deduction from appropriately chosen axioms and definitions."

#step 1
txt_tmp = clean_txt(sample_text)
#step 2
txt_tmp = rem_sw(txt_tmp)
#step 3
txt_tmp = stem_fun(txt_tmp, "stem")
#step 3
vec_tmp = read_pickle(o_path, "tf")
tmp_res = pd.DataFrame(
    vec_tmp.transform([txt_tmp]).toarray())
tmp_res.columns = vec_tmp.get_feature_names_out()
#step for chi-squared
stat_sig = 0.05
chi_tmp = read_pickle(o_path, "chi")
tmp_chi = pd.DataFrame(chi_tmp.transform(tmp_res))
p_val = pd.DataFrame(list(chi_tmp.pvalues_))
p_val.columns = ["pval"]
feat_index = list(p_val[p_val.pval <= stat_sig].index)
tmp_chi = tmp_chi[feat_index]
feature_names = tmp_res.columns[feat_index]
#tmp_chi.columns = feature_names

#step 4
mod_f = read_pickle(o_path, "rf")
pred = mod_f.predict(tmp_chi)
pred_proba = pd.DataFrame(mod_f.predict_proba(tmp_chi))
pred_proba.columns = mod_f.classes_
print (pred_proba)

