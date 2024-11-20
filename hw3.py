#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:09:07 2024

@author: xuhanyi
"""

from utils import *


"""
#load data

the_path = "data/"
o_path = "output/"

the_data = file_crawler(the_path)

the_data["body_sw"] = the_data["body"].apply(rem_sw)

test = word_fun(the_data, "body_sw")

# #apply word_fun to the "body_sw_stem" columns
ex_text = "i was fishing for fishes while i was running"

#the_data["body_sw_stem"] = the_data["body_sw"].apply(stem_fun)
the_data["body_sw_stem"] = the_data["body_sw"].apply(
     lambda x: stem_fun(x, "stem"))
the_data["body_sw_lemma"] = the_data["body_sw"].apply(
     lambda x: stem_fun(x, "lemma"))

#def write_pickle(obj_in, path_in, name_in):
#    import pickle
#    pickle.dump(obj_in, open(path_in + name_in + ".pk", "wb"))

write_pickle(the_data, o_path, "the_data")

# def read_pickle(path_in, name_in):
#     import pickle
#     the_data_t = pickle.load(open(path_in + name_in + ".pk", "rb"))
#     return the_data_t

the_data = read_pickle(o_path, "the_data")

t_form_data = xform_fun(the_data["body"], 1, 3, "tf", o_path)

"""


def word_prob(the_data, token, column='body', label_column='label'):
    token = clean_txt(token)
    
    freq = word_fun(the_data, column)
    
    target = ['all', 'fishing', 'hiking', 'machinelearning', 'mathematics']
    output = {'all': None, 'fishing': None, 'hiking':None, 'machinelearning':None, 'mathematics':None}
    
    for label in target:
        if label == 'all':
            total = sum(sum(words.values()) for words in freq.values())
            count = sum(words.get(token, 0) for words in freq.values())
        else:
            words = freq.get(label, {})
            total = sum(words.values())
            count = words.get(token, 0)
        
        if count > 0:
            output[label] = count / total
    
    return output


#Test
print(word_prob(the_data, "mathematics", column="body"))
