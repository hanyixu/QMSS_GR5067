#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:16:32 2024

@author: xuhanyi
"""


from utils import * 

the_path = "data/"

        
the_data = file_crawler(the_path)



fishing_t = the_data[the_data["label"] == "fishing"]
tmp_t = fishing_t["body"].str.cat(sep = " ")


"""
create a dictionaru that shows the word frequencu count for the entire column of tmp_t

"""

import collections
freq = dict(collections.Counter(tmp_t.split()))

"""
create a function that outputs a dictionary whose keys point to a dictionary that represents the word frequency count of each respective label in the_data["label"]

"""

##word_fun(the_path)


"""
NLTL
"""

from nltk.corpus import stopwords

sw = stopwords.words('english')

test_corpus = "the cat chased the dog at the hill"

tmp = [word for word in test_corpus.split() if word not in sw]
tmp = ' '.join(tmp)







def stem_fun(var_in): 
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    split_ex = var_in.split()
    t_l = list()
    for word in split_ex:
        tmp = ps.stem(word)
        t_l.append(tmp)
    tmp = ' '.join(t_l)
    return tmp


the_data["body_sw_stem"] = the_data["body_sw_stem"].apply(stem_fun)











