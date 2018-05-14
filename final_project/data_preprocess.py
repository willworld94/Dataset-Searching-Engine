import pandas as pd
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import nltk
import ast
from nltk.tokenize import word_tokenize

##specify stop words
stop_words = []
with open("stop_word.txt", "r") as f:
    for word in f:
        stop_words.append(word.strip())

##data preprocessing
meta_df = pd.read_csv("metadf.csv") 
lemma = nltk.wordnet.WordNetLemmatizer()
for index, i in enumerate(meta_df.loc[:, "Cols"]):
    list_obj = i.strip("[]")
    list_obj = ast.literal_eval(i)
    list_obj = [n.strip() for n in list_obj]
#     list_obj = [lemma.lemmatize(i) for n in list_obj for]
    for index, w in enumerate(list_obj):
        new_list = []
        for q in w:
            if q not in stop_words:
                new_list.append(q)
        list_obj[index] = " ".join(new_list)
    meta_df.loc[:, "Cols"][index] = list_obj[8: ]

print(meta_df.loc[:, "Cols"].head())

