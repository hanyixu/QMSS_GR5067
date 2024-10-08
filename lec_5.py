#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:13:41 2024

@author: xuhanyi
"""

import nltk
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd




the_data = pd.DataFrame(['body_sw_stem'])

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