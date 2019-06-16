
# coding: utf-8

# In[1]:


import os
os.chdir("/Users/laurensuarez/Desktop/deep_wisdom/deep_wisdom_django/dwsite/data")
os.getcwd()


# # Get version data conversion

# In[2]:


import requests
import re
import string
from bs4 import BeautifulSoup
import bs4
import pickle
import numpy as np
import keras
import nltk
import string
import pickle
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
#nltk.download('punkt')

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


### GET DATA
with open("bible_data_20181129_update.pkl", "rb") as handle:
    bible_data=pickle.load(handle)

### DISPLAY DIFFERENCES
for k,v in bible_data.items():
    print("{} {}".format(k, len(v.items())))


### CONSTRUCT EACH ENGINE
int2verse=0
verse2int=0
force=False
version_bible_data=0
for version,versionText in bible_data.items():
    version_formatted = version.replace(" ", "_")
    if not os.path.exists(version_formatted):
        os.mkdir("{}".format(version_formatted))
        os.chdir(version_formatted)
    else:
        if force:
            pass
        else:
            continue
    print(version)
    version_bible_data=versionText

    int2verse={}
    verse2int={}
    for i,v in enumerate(version_bible_data.keys()):
        int2verse[i]=v
        verse2int[v]=i
    #int2verse
    #verse2int

    bible_not_found_cf=set()
    version_bible_mapping={}
    num_verses=len(int2verse.keys())
    i=0
    for k,obj in version_bible_data.items():
        #print(k, obj[0], obj[1])
        mapping=np.zeros((num_verses))
        for cf in obj[1]:
            try:
                mapping[verse2int[cf]]=1
            except Exception as e:
                bible_not_found_cf.add(cf)
        version_bible_mapping[k]=[obj[0],mapping]
        i+=1
        if i % 10000 == 0:
            print(i)
    #kjv_bible_mapping["Genesis 1:1"][1]
    
    """ Uncomment for production saving ...
    """
    pickle_dump(version_bible_mapping, "{}_bible_mapping.pkl".format(version_formatted))
    pickle_dump(int2verse, "{}_int2verse_mapping.pkl".format(version_formatted))
    pickle_dump(verse2int, "{}_verse2int_mapping.pkl".format(version_formatted))
    print("Successful saving {}.".format(version_formatted))
    print("Lost {} verses for some reason.".format(bible_not_found_cf))
    
    
    print("Getting Corpus")
    corpus=list(map(lambda x:x[1][0], version_bible_mapping.items()))
    print(corpus[0])

    print("TFIDF Vectorizing")
    tfidf_vectorizer = TfidfVectorizer(#tokenizer=tokenize, 
                                       stop_words="english",
                                       lowercase=True, 
                                       min_df=0.0001,
                                       max_df=0.9999)
    print("Length of corpus: {}".format(len(corpus)))
    tfidf_fit=tfidf_vectorizer.fit(corpus)
    pickle_dump(tfidf_fit, "{}_tfidf_fit.pkl".format(version_formatted))
    tfidf_mat=tfidf_vectorizer.transform(corpus).todense()
    X=tfidf_mat
    y=np.array(list(map(lambda x:x[1][1], version_bible_mapping.items())))

    print(X.shape)
    print(y.shape)
    print(len(corpus))
    print(len(list(version_bible_mapping.keys())))

    print("Building Model")
    from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
    from keras import metrics
    import os
    import datetime
    def create_model(X,y):
        # Input layers
        print(X.shape)
        print(y.shape)
        model = Sequential()
        model.add(Dense(10000, input_shape=(X.shape[1],)))
        model.add(Dense(1000))
        model.add(Dense(100))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])
        return model
    
    #model_path="weights-improvement-01-54.2576.hdf5"
    def load_trained_model(weights_path, X, y):
        model = create_model(X,y)
        model.load_weights(weights_path)
        print("Loaded")
        return model

    #model=load_trained_model(model_path,X,y)

    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model = create_model(X,y)
    model.summary()

    model.fit(X, y,
                  batch_size=128,
                  epochs=1,
                  callbacks=callbacks_list)
    os.chdir('..')
    #break


# In[3]:


"""
vect=tfidf_fit.transform(["love"]).todense()

print(model.predict(vect).argsort()[0][0])
i=model.predict(vect).argsort()[0][-50]
versionText[int2verse[i]]
"""


# In[4]:


"""
print("TFIDF Vectorizing")
tfidf_vectorizer = TfidfVectorizer(#tokenizer=tokenize, 
                                   stop_words="english",
                                   lowercase=True, 
                                   min_df=0.0001,
                                   max_df=0.9999)
print("Length of corpus: {}".format(len(corpus)))
tfidf_fit=tfidf_vectorizer.fit(corpus)
"""


# In[5]:


"""
len(tfidf_fit.vocabulary_)
"""


# In[6]:


"""
for k,v in bible_data.items():
    print("{} {}".format(k, len(v.items())))
"""

