# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:20:59 2024

@author: pathouli
"""
def clean_txt(var_in):
    import re
    tmp_t = re.sub("[^A-Za-z']+", " ", var_in
                   ).strip().lower()
    return tmp_t

def read_file(full_path_in):
    f_t = open(full_path_in, "r", encoding="UTF-8")
    text_t = f_t.read() #reads entire file
    text_t = clean_txt(text_t)
    f_t.close()
    return text_t

def jd_fun(str_a_in, str_b_in):
    j_d_i = None #initialization
    try:
        str_a_i_s = str_a_in.split()
        str_b_i_s = str_b_in.split()
        str_a_s_set_i = set(str_a_i_s)
        str_b_s_set_i = set(str_b_i_s)
        the_u_i = str_a_s_set_i.union(str_a_s_set_i)
        the_i_i = str_a_s_set_i.intersection(str_b_s_set_i)
        j_d_i = len(the_i_i) / len(the_u_i) 
    except:
        print ("houston we have an issue")
        pass
    return j_d_i

def loop_fun(list_in):
    import pandas as pd
    my_pd_t = pd.DataFrame()
    #list_in_t = [word for word in list_in if len(word) <= 3]
    for name in list_in:
        if len(name) <= 3:
            tmp = pd.DataFrame(
                {"name": name, "len": len(name)}, index=[0])
            my_pd_t = pd.concat([my_pd_t, tmp], ignore_index=True)
    return my_pd_t

def file_crawler(path_in):
    import os
    import pandas as pd
    my_pd_t = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
       for name in files:
           try:
               txt_t = read_file(root + "/" + name)
               if len(txt_t) > 0:
                   the_lab = root.split("/")[-1]
                   tmp_pd = pd.DataFrame(
                       {"body": txt_t, "label": the_lab}, index=[0])
                   my_pd_t = pd.concat(
                       [my_pd_t, tmp_pd], ignore_index=True)
           except: 
               print (root + "/" + name)
               pass
    return my_pd_t

def word_fun(df_in, col_in):
    import collections
    all_dictionary = dict()
    the_topics = set(df_in["label"]) #the_data["label"].unique()
    for x in the_topics:
        tmp = df_in[df_in["label"] == x]
        tmp = tmp[col_in].str.cat(sep=" ")
        all_dictionary[x] = dict(collections.Counter(tmp.split()))
    return all_dictionary

def rem_sw(str_in):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    tmp = [word for word in str_in.split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

def stem_fun(var_in, sw_in):
    #stemming pick up next
    if sw_in == "stem":
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
    else:
        from nltk.stem import WordNetLemmatizer
        ps = WordNetLemmatizer()
    split_ex = var_in.split()
    t_l = list()
    for word in split_ex:
        if sw_in == "stem":
            tmp = ps.stem(word)
        else:
            tmp = ps.lemmatize(word)
        t_l.append(tmp)
    tmp = ' '.join(t_l)
    return tmp

def xform_fun(df_in, m_in, n_in, sw_in):
    import pandas as pd
    if sw_in == "tf":
        from sklearn.feature_extraction.text import CountVectorizer 
        cv = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        cv = TfidfVectorizer(ngram_range=(m_in, n_in))
    x_f_data_t = pd.DataFrame(
        cv.fit_transform(df_in).toarray()) #be careful
    x_f_data_t.columns = cv.get_feature_names_out()
    return x_f_data_t