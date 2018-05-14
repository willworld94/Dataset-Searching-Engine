#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import re
from collections import defaultdict
import json
from pyspark import SparkContext, SparkConf
import sys
import math

def dotprod(a, b):
    """ Compute dot product
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        dotProd: result of the dot product with the two input dictionaries
    """
    return sum([a[term]*b[term] for term in a.keys() & b.keys()])

def norm(a):
    """ Compute square root of the dot product
    Args:
        a (dictionary): a dictionary of record to value
    Returns:
        norm: a dictionary of tokens to its TF values
    """
    return math.sqrt(dotprod(a,a))

def cossim(a, b):
    """ Compute cosine similarity
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        cossim: dot product of two dictionaries divided by the norm of the first dictionary and
                then by the norm of the second dictionary
    """
    if float(norm(a)) != 0.0 and float(norm(b)) != 0.0:
        return dotprod(a,b)/(float(norm(a))*float(norm(b)))
    else:
        return -1

def cosineSimilarity(t1, t2):
    """ Compute cosine similarity between two strings
    Args:
        t1 (tfidf of s1): first string
        t2 (tfidf of s2): second string
        idfsDictionary (dictionary): a dictionary of IDF values
    Returns:
        cossim: cosine similarity value
    """
#     print(type(string1), type(string2))
    t1 = tfidf(t1)
    t2 = t2.decode("utf-8")
    t2 = tfidf(t2)
    return cossim(t1, t2)

def tokenize(string, dsid):
    """ An implementation of input string tokenization that excludes stopwords
    Args:
        string (str): input string
    Returns:
        list: a list of tokens without stopwords
    """
    return [(x.strip(), dsid) for x in re.split(split_regex,string.lower()) if x != '']

def exact_col_ser(keyword):
    names = []  
    ids = col_ids.lookup(keyword)[0].split('&')
    if len(ids) == 0:
        print("No dataset containing this column!")
        return
    names = [id_name.lookup(k)[0] for k in ids]
    res = pd.DataFrame({'Name':names, 'Id':ids})
    print("You searched the column called: "+keyword)
    print("The datasets containing this column are :")
    print(res)
    download = input("If you want to download the dataset, press 0, or you want to do a new search, press 1 \n")
    whether_download(download)

def exact_ds_ser(keyword):
    res = name_id.lookup(keyword.encode("utf-8"))
    if len(res) == 0:
        print("No such dataset")
        cont_or_end()
    else:
        print("The ID of the dataset of this name: \n")
        print("\n".join([l for l in res]))
        download = input("If you want to download the dataset, press 0, or you want to do a new search, press 1 \n")
        whether_download(download)

def get_file_print(file):
    os.system("hdfs dfs -get /user/bigdata/nyc_open_data/"+file+'.json')
    with open(file+'.json', "r") as f:
        read_file = json.load(f)
    cols = read_file["meta"]["view"]["columns"]
    col_name = [i["name"] for i in cols]
    dataset_name = read_file['meta']['view']['name'].replace(' ', '_')
    data_frame = pd.DataFrame(data=read_file["data"], columns=col_name)
    data_frame.to_csv("{}.csv".format(dataset_name))
    os.system("rm "+file+'.json')
    os.system("hdfs dfs -rm "+file+'.json')
    print(data_frame.head(10))
   
def whether_download(download):
    if download =="0":
        file_name = input("Please enter file Id \n")
        get_file_print(file_name)
        cont_or_end()
    else:
        cont_or_end()

 
def tokenize_vag(string):
     return [x.strip() for x in string.lower().split(" ") if x != '']
def tf(tokens):
    return {token : tokens.count(token)/float(len(tokens)) for token in tokens}
def idfs(corpus):
    N = corpus.count()
    uniqueTokens = corpus.map(lambda item: list(set(tokenize_vag(item[1].decode("utf-8")))))
#     print(uniqueTokens.take(9))
    tokenCountPairTuple = uniqueTokens.flatMap(lambda thing: [(element,1) for element in thing])
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a, b: a + b)
    return (tokenSumPairTuple.map(lambda tuples: (tuples[0],float(N)/float(tuples[1]))))
def idfs_col(corpus):
    N = corpus.count()
    uniqueTokens = corpus.map(lambda item: list(set(tokenize_vag(item[1]))))
#     print(uniqueTokens.take(9))
    tokenCountPairTuple = uniqueTokens.flatMap(lambda thing: [(element,1) for element in thing])
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a, b: a + b)
    return (tokenSumPairTuple.map(lambda tuples: (tuples[0],float(N)/float(tuples[1]))))


def tfidf(string):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tokens = tokenize_vag(string)
    tfs = tf(tokens)
    tfIdfDict = {token: tfs[token]*idf_dict[token] for token in tokens}
    return tfIdfDict

def tfidf_col(string):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tokens = tokenize_vag(string)
    tfs = tf(tokens)
    tfIdfDict = {token: tfs[token]*col_idf_dict[token] for token in tokens}
    return tfIdfDict
def cosineSimilarity_col(t1, t2):
    """ Compute cosine similarity between two strings
    Args:
        t1 (tfidf of s1): first string
        t2 (tfidf of s2): second string
        idfsDictionary (dictionary): a dictionary of IDF values
    Returns:
        cossim: cosine similarity value
    """
#     print(type(string1), type(string2))
    t1 = tfidf_col(t1)
    t2 = tfidf_col(t2)
    return cossim(t1, t2)



def vag_ds_ser(kw):
    kw = kw.lower().strip()
    try:
        id_sim = id_name.map(lambda t: (t[0], cosineSimilarity(kw, t[1]))).sortBy(lambda l: l[1], False).take(5)
        ids = [t[0] for t in id_sim]
        names = [id_name.lookup(i) for i in ids]
        result = pd.DataFrame({"data set id": ids, "data set name": names})
        print(result)
        download = input("If you want to download the dataset, press 0, or you want to do a new search, press 1 \n")
        whether_download(download)
    except:
        sys.exit(0)
        print("no such a result")

def vag_col_ser(kw):
    kw = kw.lower().strip()
    try:
        id_sim = ids_col.map(lambda t: (t[0], cosineSimilarity_col(kw, t[1]))).sortBy(lambda l: l[1], False).take(5)
#         ids = [t[0] for t in id_sim]
        ids = [l[0].split("&") for l in id_sim]
        ids = sum(ids, [])[:5]
#         col = [ids_col.lookup(i) for i in ids]
        names = [id_name.lookup(i) for i in ids]
        result = pd.DataFrame({"data set id": ids, "data set name": names})
        print(result)
        download = input("If you want to download the dataset, press 0, or you want to do a new search, press 1 \n")
        whether_download(download)
    except:
        sys.exit(0)
        print("no such a result")

def exa_search_level():
    level = input("Please select searching level: 0 = Data Set Level, 1 = Column Level Search \n")
    if level == "0":
        sr_level = input("Please input data set name: \n")
        exact_ds_ser(sr_level)
    else:
        sr_level = input("Please input column name: \n")
        exact_col_ser(sr_level)

def vag_search_level():
    level = input("Please select searching level: 0 = Data Set Level, 1 = Column Level Search \n")
    if level == "0":
        sr_level = input("Please input data set name: \n")
        vag_ds_ser(sr_level)
    else:
        sr_level = input("Please input column name: \n")
        vag_col_ser(sr_level)



# continue search or end 
def exit_or_cont(contin):
    if contin == "0":
        interface()
    else:
        print("Bye Bye XO XO \n")
        sys.exit(0)
# continue search or exit
def cont_or_end():
    contin = input("Thanks for using GOSSIP SEARCHING ENGINE. If you wanna do more search, press 0; Exit, press 1 \n")
    exit_or_cont(contin)

# terminal interface 
def interface():
    welcome = input("Welcome to GOSSIP SEARCHING ENGINE, please select searching mode: 0 = Exact Search, 1 = Vague Search \n")
    if welcome == "0":
        exa_search_level()
        cont_or_end()
    else:
        vag_search_level()
        cont_or_end()


if __name__ == "__main__":
    sc = SparkContext()

    df = sc.textFile("metadf_new.txt").map(lambda line: line.split("\t")).filter(lambda line: len(line)>1).map(lambda line: (line[2],line[1],line[3].encode("utf-8")))
    split_regex = r'\W+'
    id_name = df.map(lambda l: (l[0], l[2]))
    d_name = df.map(lambda l: (l[0], l[2]))
    col_id = df.flatMap(lambda l: (tokenize(l[1],l[0])))
    col_ids = col_id.reduceByKey(lambda x, y: x +'&'+ y)
    name_id = df.map(lambda l: (l[2].strip(), l[0])).reduceByKey(lambda x, y: x + "&" + y)
    id_name_idf = idfs(id_name)
    idf_dict = id_name_idf.collectAsMap()
    ids_col = col_ids.map(lambda l: (l[1],l[0]))
    ids_col_idf = idfs_col(ids_col)
    col_idf_dict = ids_col_idf.collectAsMap()
#id_name_idf = idfs(id_name)
#idf_dict = id_name_idf.collectAsMap()

    #split_regex = r','

    interface()
