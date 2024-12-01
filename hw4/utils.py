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

def xform_fun(df_in, m_in, n_in, sw_in, path_in):
    import pandas as pd
    if sw_in == "tf":
        from sklearn.feature_extraction.text import CountVectorizer 
        cv = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        cv = TfidfVectorizer(ngram_range=(m_in, n_in), use_idf=False)
    x_f_data_t = pd.DataFrame(
        cv.fit_transform(df_in).toarray()) #be careful
    write_pickle(cv, path_in, sw_in)
    x_f_data_t.columns = cv.get_feature_names_out()
    return x_f_data_t

def read_pickle(path_in, name_in):
    import pickle
    the_data_t = pickle.load(open(path_in + name_in + ".pk", "rb"))
    return the_data_t

def write_pickle(obj_in, path_in, name_in):
    import pickle
    pickle.dump(obj_in, open(path_in + name_in + ".pk", "wb"))
    
def cosine_fun(df_in_a, df_in_b, lab_in, path_in, name_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_sim = pd.DataFrame(cosine_similarity(df_in_a, df_in_b))
    cos_sim.index = lab_in
    cos_sim.columns = lab_in
    write_pickle(cos_sim, path_in, name_in)
    return cos_sim

def extract_embeddings_pre(df_in, out_path_i, name_in):
    #https://code.google.com/archive/p/word2vec/
    #https://pypi.org/project/gensim/
    #pip install gensim
    #name_in = 'models/word2vec_sample/pruned.word2vec.txt'
    import pandas as pd
    from nltk.data import find
    from gensim.models import KeyedVectors
    import pickle
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(my_model_t.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    word2vec_sample = str(find(name_in))
    my_model_t = KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False)
    # word_dict = my_model.key_to_index
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model_t, open(out_path_i + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(out_path_i + "embeddings_df.pkl", "wb" ))
    return tmp_data, my_model_t

def domain_train(df_in, path_in, name_in):
    #domain specific
    import pandas as pd
    import gensim
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(model.wv.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    model = gensim.models.Word2Vec(df_in.str.split())
    model.save(path_in + 'body.embedding')
    #call up the model
    #load_model = gensim.models.Word2Vec.load('body.embedding')
    model.wv.similarity('fish','river')
    tmp_data = pd.DataFrame(df_in.str.split().apply(get_score))
    return tmp_data, model

def chi_fun(df_in, lab_in, k_in, p_in, n_in, stat_sig):
    from sklearn.feature_selection import chi2, SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(feat_sel.fit_transform(df_in, lab_in))
    p_val = pd.DataFrame(list(feat_sel.pvalues_))
    p_val.columns = ["pval"]
    feat_index = list(p_val[p_val.pval <= stat_sig].index)
    dim_data = dim_data[feat_index]
    feature_names = df_in.columns[feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, p_in, n_in)
    write_pickle(dim_data, p_in, "chi_data_" + n_in)
    return dim_data, feat_sel

def pca_fun(df_in, ratio_in, path_o, n_in):
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA(n_components=ratio_in)
    dim_red = pd.DataFrame(pca.fit_transform(df_in))
    exp_var = sum(pca.explained_variance_ratio_)
    print ("Explained variance", exp_var)
    write_pickle(pca, path_o, n_in)
    return dim_red

def cos_fun(df_a, df_b, label_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_sim = pd.DataFrame(cosine_similarity(
        df_a, df_b))
    try:
        cos_sim.index = label_in
        cos_sim.columns = label_in
    except:
        pass
    return cos_sim

def model_fun(df_in, lab_in, g_in, t_s, sw_in, p_o):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import GridSearchCV
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, lab_in, test_size=t_s, random_state=42)
    
    if sw_in == "rf":
        model = RandomForestClassifier(random_state=123)
    elif sw_in == "gnb":
        model = GaussianNB()
    
    clf = GridSearchCV(model, g_in)
    clf.fit(X_train, y_train)
    
    best_perf = clf.best_score_
    print (best_perf)
    best_params = clf.best_params_
    print (best_params)
    
    if sw_in == "rf":
        model = RandomForestClassifier(random_state=123, **best_params)
    elif sw_in == "gnb":
        model = GaussianNB(**best_params)
    
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
        X_test, y_test, test_size=0.10, random_state=42)
    
    model.fit(X_train_val, y_train_val)
    write_pickle(model, p_o, sw_in)
    y_pred = model.predict(X_test_val)
    y_pred_likelihood = pd.DataFrame(
        model.predict_proba(X_test_val))
    y_pred_likelihood.columns = model.classes_
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test_val, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "fscore", None]
    
    #feature importance
    try:
        feat_imp = pd.DataFrame(model.feature_importances_)
        feat_imp.index = X_train_val.columns
        feat_imp.columns = ["score"]
        feat_imp.to_csv(p_o + sw_in + "_m.csv")
        perc_prop = len(feat_imp[feat_imp["score"] > 0]) / len(feat_imp) * 100
        print (perc_prop)
    except:
        print ("Not transparent")
        pass
    return model