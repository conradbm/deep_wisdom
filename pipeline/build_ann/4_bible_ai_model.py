
# coding: utf-8

# ## Model Bible AI

# In[1]:


import pickle


# In[2]:


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
    


# In[3]:


print("Loading data.")
kjv_bible_data=pickle_load("data/bible_data_20181123_update.pkl")
kjv_bible_mapping=pickle_load("data/kjv_bible_mapping.pkl")
int2verse=pickle_load("data/int2verse_mapping.pkl")
verse2int=pickle_load("data/verse2int_mapping.pkl")
print("Load complete.")


# In[4]:


""" UNCOMMENT IF YOU WANT TO DUMP OUT NEW VECTORIZORS """
"""
from sklearn.feature_extraction.text import TfidfVectorizer
#nltk.download('punkt')


corpus=list(map(lambda x:x[1][0], kjv_bible_mapping.items()))
print("Corpus head:")
print(corpus[:5])


# In[5]:


tfidf_vectorizer = TfidfVectorizer(stop_words="english",
                                   lowercase=True, 
                                   norm='l2')
tfidf_fit=tfidf_vectorizer.fit(corpus)


# ## Parallelize the transformation
# <hr>
# Freaky fast..

# In[13]:


import multiprocessing
import pandas as pd
import numpy as np
from multiprocessing import Pool
import scipy.sparse as sp
#num_partitions = 5
num_cores = multiprocessing.cpu_count()
num_partitions = num_cores-1 # I like to leave some cores for other
#processes
print(num_partitions)

def parallelize_dataframe(df, func):
    a = np.array_split(df, num_partitions)
    del df
    pool = Pool(num_cores)
    #df = pd.concat(pool.map(func, [a,b,c,d,e]))
    df = sp.vstack(pool.map(func, a), format='csr')
    pool.close()
    pool.join()
    return df

def test_func(data):
    #print("Process working on: ",data)
    tfidf_matrix = tfidf_fit.transform(data)
    #return pd.DataFrame(tfidf_matrix.toarray())
    return tfidf_matrix

tfidf_parallel = parallelize_dataframe(corpus, test_func)
pickle_dump(tfidf_parallel, 'data/tfidf_bible_matrix.pkl')
pickle_dump(tfidf_fit, 'data/tfidf_bible_fit.pkl')
tf_idf_bible_matrix=pickle_load('data/tfidf_bible_matrix.pkl')
tf_idf_bible_fit=pickle_load('data/tfidf_bible_fit.pkl')
"""

#uncomment if recollecting
tfidf_parallel=pickle_load('data/tfidf_bible_matrix.pkl')
tfidf_fit=pickle_load('data/tfidf_bible_fit.pkl')

print("tfidf data head:")
print(tf_idf_bible_matrix[:4])

# ## Construct the model

# In[7]:


import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

X=tfidf_parallel.todense()
y=np.array(list(map(lambda x:x[1][1], kjv_bible_mapping.items())))

print("Data shape:")
print(X.shape)
print(y.shape)
print(len(corpus))
print(len(list(kjv_bible_mapping.keys())))

# In[8]:


from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from keras import metrics
import os
from keras import optimizers

def create_model_baseline(X,y):
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
def create_model_experiment(X,y):
    # Input layers
    print(X.shape)
    print(y.shape)
    #(31102, 12302)
    model = Sequential()
    model.add(Dense(10000, input_shape=(X.shape[1],)))
    model.add(Dense(10000))
    model.add(Dense(1000))
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(y.shape[1], activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy'])
    return model

def load_trained_model(X, y, which="baseline"):

    if which == "baseline":
        weights_path="data/weights-improvement-01-54.2576.hdf5"
        model = create_model_baseline(X,y)
    elif which == "experiment1":
        weights_path="data/weights-improvement-01-56.8923.hdf5"
        model = create_model_experiment(X,y)
    else:
        print("Error occurred loading model.")
    model.load_weights(weights_path)
    print("Loaded")
    return model

#model=load_trained_model(X,y,which="experiment1")

"""UNCOMMENT IF YOU NEED A NEW MODEL"""

filepath="data/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

new_model = create_model_experiment(X,y)

print("About to train model.")
print(new_model.summary())
input("...")
new_model.fit(X, y,
              batch_size=128,
              epochs=3,
              callbacks=callbacks_list)



while True:
    searchText=input("Enter search text...")
    if searchText == "exit":
        break
    tfidf_matrix = tfidf_fit.transform([searchText])
    x1=tfidf_matrix.todense()
    v=model.predict(x1)[0]
    indices=[]
    int2verse[np.argmax(v)]
    for i in range(6):
        idx = np.argmax(v)
        print(v[idx], idx)
        indices.append(idx); 
        v[idx]=0
    for index in indices:
        print
        print(int2verse[index], kjv_bible_mapping[int2verse[index]][0])






if __name__ == '__main__':
    main(loaded=True)

