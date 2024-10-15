# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:47:31 2024

@author: pathouli
"""

from utils import *

the_path = "C:/Users/pathouli/Box Sync/myStuff/academia/columbia/socialSciences/GR5067/2024_fall/data/"

the_data = file_crawler(the_path)

the_data["body_sw"] = the_data["body"].apply(rem_sw)

test = word_fun(the_data, "body_sw")

#apply word_fun to the "body_sw_stem" columns
ex_text = "i was fishing for fishes while i was running"

#the_data["body_sw_stem"] = the_data["body_sw"].apply(stem_fun)
the_data["body_sw_stem"] = the_data["body_sw"].apply(
    lambda x: stem_fun(x, "stem"))
the_data["body_sw_lemma"] = the_data["body_sw"].apply(
    lambda x: stem_fun(x, "lemma"))

body = word_fun(the_data, "body")
body_sw = word_fun(the_data, "body_sw")
body_sw_stem = word_fun(the_data, "body_sw_stem")
body_sw_lemma = word_fun(the_data, "body_sw_lemma")

test = xform_fun(the_data["body"], 1, 3, "tfidf")

#pickle