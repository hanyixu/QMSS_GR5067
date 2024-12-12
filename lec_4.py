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

#stemming pick up next
