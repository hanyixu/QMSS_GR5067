# -*- coding: utf-8 -*-





def read_file(full_path_in):
    import re
    f_t = open(full_path_in, "r", encoding="UTF-8")
    text_t = f_t.read().lower() #reads entire file
    text_t = re.sub("[^A-Za-z']+", " ", text_t).strip()
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






def clean_txt(var_in):
    import re
    tmp_t = re.sub("[^A-Za-z']+", " ", var_in
                   ).strip().lower()
    return tmp_t






def file_crawler(path_in):
    import os
    import pandas as pd
    my_pd_t = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
        for name in files:
            try:
                txt_t = read_file(root + "/" + name)
                if len(txt_t)>0:
                    the_lab = root.split("/")[-1]
                    tmp_pd = pd.DataFrame(
                        {"body": txt_t, "label": the_lab}, index=[0])
                    my_pd_t = pd.concat(
                        [my_pd_t, tmp_pd], ignore_index=True)
            except:
                print(root + '/' + name)
                pass
    return my_pd_t







def word_fun(the_data):
    import collections
    out = {}
    #need a set or the_data["label"].unique()
    for i in the_data["label"].unique():
        data_t = the_data[the_data["label"] == i]
        tmp_t = data_t["body"].str.cat(sep = " ")
        freq_t = dict(collections.Counter(tmp_t.split()))
        out[i] = freq_t
    return out