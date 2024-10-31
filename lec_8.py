# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:47:31 2024

@author: pathouli
"""
from utils import *

the_path = "data/"
o_path = "output/"

the_data = file_crawler(the_path)

the_data["body_sw"] = the_data["body"].apply(rem_sw)

# test = word_fun(the_data, "body_sw")

# #apply word_fun to the "body_sw_stem" columns
ex_text = "i was fishing for fishes while i was running"

the_data["body_sw_stem"] = the_data["body_sw"].apply(stem_fun)
the_data["body_sw_stem"] = the_data["body_sw"].apply(
     lambda x: stem_fun(x, "stem"))
the_data["body_sw_lemma"] = the_data["body_sw"].apply(
     lambda x: stem_fun(x, "lemma"))

# def write_pickle(obj_in, path_in, name_in):
#     import pickle
#     pickle.dump(obj_in, open(path_in + name_in + ".pk", "wb"))

write_pickle(the_data, o_path, "the_data")

# def read_pickle(path_in, name_in):
#     import pickle
#     the_data_t = pickle.load(open(path_in + name_in + ".pk", "rb"))
#     return the_data_t

the_data = read_pickle(o_path, "the_data")

t_form_data = xform_fun(the_data["body"], 1, 3, "tf", o_path)


dim_data = pca_fun(t_form_data, 0.95, o_path, "pca")

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
cos_sim = pd.DataFrame(cosine_similarity(
    t_form_data, t_form_data))
cos_sim.index = the_data.label
cos_sim.columns = the_data.label






chi_data, chi_m = chi_fun(t_form_data, the_datadata.label, 
                          len(t_form_data.columns), o_path, "tf", 0.05)

"""
refactor xform_fun to write cv
"""

#pickle


# body = word_fun(the_data, "body")
# body_sw = word_fun(the_data, "body_sw")
# body_sw_stem = word_fun(the_data, "body_sw_stem")
# body_sw_lemma = word_fun(the_data, "body_sw_lemma")