# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:47:31 2024

@author: pathouli
"""

from utils import * #jd_fun, loop_fun

the_path = "data/fishing/"

file_name = "fish_121827061000.txt"

text = read_file(the_path + file_name)

#text = " lets say we had [] { in our text!"
#import re
#text_pre = re.sub("[^A-Za-z']+", " ", text).strip()#strip remove trailing space


# f = open(the_path + file_name, "r", encoding="UTF-8")
# text = f.readlines() #reads entire file
# f.close()

#good for large files when you dont want everything read at once
# with open(the_path + file_name, "r", encoding="UTF-8") as fp:
#     for line in fp:
#         print(line)


"/n"

"""
print ("hello world")

the_ans = jd_fun("abc def def ghi", "abc fgh ghi fty def")

names = ["patrick", "bob", "sue", "jenny"]

a = loop_fun(names)

#exercise
refactor the example above but if the number of characters
in a name exceeds 3, then do NOT populate that value in the
pandas dataframe called my_pd_t
"""