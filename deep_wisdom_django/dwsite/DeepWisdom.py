import numpy as np
import pickle
import os
import subprocess
import sqlite3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from keras import metrics
from keras import optimizers
from keras import backend as K
import tensorflow as tf
K.clear_session()

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

def get_db_connection(root_dir="",loc=os.path.join("data","bible.db")):
    """ DATABASE CONNECTION """
    # connect
    print("Connecting to database.")
    conn=sqlite3.connect(os.path.join(root_dir,loc) )
    if conn is not None:
        pass
    else:
        print("Error! cannot create the database connection.")
    return conn

class DeepWisdom:

    def __init__(self, root_dir="",db_loc=os.path.join("data","bible.db")):
        print("Constructing DeepWisdom")
        mappings, tfidfs, data, model = self.load_data(root_dir=root_dir)
        #self.kjv_bible_mapping=mappings[0]
        self.int2verse=mappings[1]
        self.verse2int=mappings[2]
        self.tf_idf_bible_fit=tfidfs[0]
        #self.tf_idf_bible_matrix=tfidfs[1]
        #self.X=data[0]
        #self.y=data[1]
        self.model=model

    def create_model_baseline(self):
        #X Shape:  (31102, 12302)
        #y Shape:  (31102, 31102)
        print("X Shape: ", (31102, 12302))
        print("y Shape: ", (31102, 31102))
        model = Sequential()
        model.add(Dense(10000, input_shape=(12302,)))
        model.add(Dense(1000))
        model.add(Dense(100))
        model.add(Dense(31102, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])
        return model

    def load_trained_model(self, weights_path):
        print("Loading model weights.")
        model = self.create_model_baseline()
        model.load_weights(weights_path)
        print("Load complete.")
        return model
    
    def load_data(self,root_dir=""):

        """ DATA LOADING """
        print("Loading data.")
        #kjv_bible_data=pickle_load("../data/bible_data_20181123_update.pkl")
        #print(".")
        #kjv_bible_mapping=pickle_load(root_dir+"data/kjv_bible_mapping.pkl")
        #print(".",)
        int2verse=pickle_load(os.path.join(root_dir,"data/int2verse_mapping.pkl"))
        print(".",)
        verse2int=pickle_load(os.path.join(root_dir,"data/verse2int_mapping.pkl"))
        print(".",)
        tf_idf_bible_fit=pickle_load(os.path.join(root_dir,'data/tfidf_bible_fit.pkl'))
        #print(".",)
        #tf_idf_bible_matrix=pickle_load(root_dir+'data/tfidf_bible_matrix.pkl')
        #print(".",)
        #X=tf_idf_bible_matrix.todense()
        #print(".",)
        #y=np.array(list(map(lambda x:x[1][1], kjv_bible_mapping.items())))
        #print(".",)
        #Baseline -- Quality!
        weights_path=os.path.join(root_dir,"data/weights-improvement-01-54.2576.hdf5")
        global model
        model=self.load_trained_model(weights_path)
        global graph
        graph = tf.get_default_graph()
        print(".")
        print("Load complete.")

        return ((None,int2verse,verse2int),(tf_idf_bible_fit,None),(None,None),model)

    def query(self,searchText, debug=True, connect=True):

        if connect:
            self.conn=get_db_connection()
        if debug:
            print("Getting search r")
            print(searchText)
        tfidf_matrix = self.tf_idf_bible_fit.transform([searchText])
        x1=tfidf_matrix.todense()
        #print("x1 shape", x1.shape)
        
        with graph.as_default():
            v=self.model.predict(x1)[0]
            indices = np.argsort(v)[-50:][::-1]
            self.topK=[]
            for index in indices:
                firstPart=self.int2verse[index].split(" ")
                book=" ".join(i for i in firstPart[:-1])
                chapter_verse=firstPart[-1]
                results = self.conn.cursor().execute("SELECT * FROM T_Bible where T_Bible.book=? and T_Bible.chapter_verse=?", (book, chapter_verse)).fetchall()[0]
                self.topK.append((self.int2verse[index], results[3], v[index]))

            ### NOTE: We reverse the topK because as they are written in Tkinter the first show up last. Ironic right? ###    
            results_dict={}
            for thing in self.topK:
                print(thing[0],thing[2])
                results_dict[thing[0]]=thing[1]
            print()
            print()
            print(results_dict)
            return results_dict