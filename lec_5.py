# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:47:31 2024

@author: pathouli
"""

from utils import *

the_path = "C:/Users/pathouli/Box Sync/myStuff/academia/columbia/socialSciences/GR5067/2024_fall/data/"

the_data = file_crawler(the_path)

the_data["body_sw"] = the_data["body"].apply(rem_sw)

#test_sw = rem_sw("the cat chased the dog up the hill")

#test_corpus = "the cat chased the dog up the hill"

test = word_fun(the_data, "body_sw")


#turn the below into a function that outputs the
#joined text after stemming

#create a new column on the_data called "body_sw_stem"
#which is the output of the above

#apply word_fun to the "body_sw_stem" columns
ex_text = "i was fishing for fishes while i was running"

def stem_fun(var_in):
    #stemming pick up next
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    split_ex = var_in.split()
    t_l = list()
    for word in split_ex:
        tmp = ps.stem(word)
        t_l.append(tmp)
    tmp = ' '.join(t_l)
    return tmp

the_data["body_sw_stem"] = the_data["body_sw"].apply(stem_fun)

body = word_fun(the_data, "body")
body_sw = word_fun(the_data, "body_sw")
body_sw_stem = word_fun(the_data, "body_sw_stem")






