# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:47:31 2024

@author: pathouli
"""

from utils import *

the_path = "C:/Users/pathouli/Box Sync/myStuff/academia/columbia/socialSciences/GR5067/2024_fall/data/"
o_path = "C:/Users/pathouli/Box Sync/myStuff/academia/columbia/socialSciences/GR5067/2024_fall/output/"

# the_data = file_crawler(the_path)

# the_data["body_sw"] = the_data["body"].apply(rem_sw)

# test = word_fun(the_data, "body_sw")

# #apply word_fun to the "body_sw_stem" columns
# ex_text = "i was fishing for fishes while i was running"

# #the_data["body_sw_stem"] = the_data["body_sw"].apply(stem_fun)
# the_data["body_sw_stem"] = the_data["body_sw"].apply(
#     lambda x: stem_fun(x, "stem"))
# the_data["body_sw_lemma"] = the_data["body_sw"].apply(
#     lambda x: stem_fun(x, "lemma"))

# def write_pickle(obj_in, path_in, name_in):
#     import pickle
#     pickle.dump(obj_in, open(path_in + name_in + ".pk", "wb"))

# write_pickle(the_data, o_path, "the_data")

# def read_pickle(path_in, name_in):
#     import pickle
#     the_data_t = pickle.load(open(path_in + name_in + ".pk", "rb"))
#     return the_data_t

the_data = read_pickle(o_path, "the_data")

t_form_data = xform_fun(the_data["body_sw"], 1, 3, "tf", o_path)

chi_data, chi_m = chi_fun(t_form_data, the_data.label,
                      len(t_form_data.columns), o_path, "tf", 0.01)

"""
make a function called pca_fun enable user to dial in desired
explained variance and save off the 
"""

# dim_data = pca_fun(t_form_data, 0.95, o_path, "pca")

"""
turn into a function
"""

#test = cos_fun(t_form_data, t_form_data, the_data.label)

#cos
#chi
#embed




