# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:43:50 2024

@author: pathouli
"""

"""
download data_fun.pk from data
looad that into a variable called the_data_ex
using technique we have learned, determine what the topics
might be
"""

from utils import *
o_path = "C:/Users/pathouli/Box Sync/myStuff/academia/columbia/socialSciences/GR5067/2024_fall/output/"

the_data_ex = read_pickle(o_path, "data_fun")

the_data_ex["body"] = the_data_ex["body"].apply(clean_txt)

the_data_ex["body_sw"] = the_data_ex["body"].apply(rem_sw)

the_data_ex["body_sw_stem"] = the_data_ex["body_sw"].apply(
            lambda x: stem_fun(x, "lemma"))

t_form_data = xform_fun(the_data_ex["body_sw_stem"], 1, 3, "tfidf", o_path)
#wrd_dictionary = word_fun(the_data_ex, "body_sw_stem")
t_form_data_stats = t_form_data.sum(axis=0)
t_form_data_stats.index = t_form_data.columns

