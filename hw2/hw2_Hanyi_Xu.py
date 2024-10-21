#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:23:32 2024

@author: xuhanyi

Note: negative-words.txt, utils.py, hw2_Hanyi_Xu.py, and positive-words.txt are all under same folder

"""
from utils import *

#Question 1
def gen_senti(var_in, pw_path, nw_path):
    var_process = clean_txt(var_in).split()
    pw = open(pw_path, "r", encoding="utf-8", errors="replace").read().split()
    nw = open(nw_path, "r", encoding="utf-8", errors="replace").read().split()
    nc = 0
    pc = 0
    S = 0
    for word in var_process:
        if word in pw:
             pc += 1
        if word in nw:
             nc += 1
    S = (pc-nc)/(pc+nc) if (pc + nc) != 0 else 0
    return S
    
    
var_in = 'In sheer amazement, it amazes me how the amazing ability of my cat to nap for hours amazingly drains all my energy, as if a drastic spell has been cast!'

print(gen_senti(var_in, "positive-words.txt", "negative-words.txt"))


#Question 2
the_path = "data/"
the_data = file_crawler(the_path)
the_data["simple_senti"] = the_data["body"].apply(lambda body: gen_senti(body, "positive-words.txt", "negative-words.txt"))

#Question 3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
the_data["vader"] = the_data["body"].apply(lambda body: analyzer.polarity_scores(body)["compound"])

#Question 4
print("vader mean: ", the_data["vader"].mean())
print("simple_senti mean: ", the_data["simple_senti"].mean())

print("vader median: ", the_data["vader"].median())
print("simple_senti median: ", the_data["simple_senti"].median())

print("vader stdev: ", the_data["vader"].std())
print("simple_senti stdev: ", the_data["simple_senti"].std())