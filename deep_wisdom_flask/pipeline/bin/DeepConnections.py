import numpy as np
import pickle
import os
from tkinter import *
import subprocess
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

import sqlite3
def select_book_chapter_verse(conn,params):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_Bible where T_Bible.book=? and T_Bible.chapter_verse=?", params)
 
    rows = cur.fetchall()
 
    #for row in rows:
    #    print(row)
    return rows

# connect
conn=sqlite3.connect("data/bible.db")
if conn is not None:
    pass
else:
    print("Error! cannot create the database connection.")

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
    
print("Loading data.")
#kjv_bible_data=pickle_load("../data/bible_data_20181123_update.pkl")
#print(".")
kjv_bible_mapping=pickle_load("data/kjv_bible_mapping.pkl")
print(".",)
int2verse=pickle_load("data/int2verse_mapping.pkl")
print(".",)
verse2int=pickle_load("data/verse2int_mapping.pkl")
print(".",)
tf_idf_bible_fit=pickle_load('data/tfidf_bible_fit.pkl')
print(".",)
tf_idf_bible_matrix=pickle_load('data/tfidf_bible_matrix.pkl')
print(".",)
X=tf_idf_bible_matrix.todense()
print(".",)
y=np.array(list(map(lambda x:x[1][1], kjv_bible_mapping.items())))
print(".",)
def create_model_baseline(X,y):
    # Input layers
    #(31102, 12302)
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

def load_trained_model(weights_path, X, y):
    #print("Loading model weights.")
    model = create_model_baseline(X,y)
    #model = create_model_experiment(X,y)
    model.load_weights(weights_path)
    #print("Load complete.")
    return model

#Baseline -- Quality!
model=load_trained_model("data/weights-improvement-01-54.2576.hdf5", X, y)
print(".")
print("Load complete.")

class Window(Frame):
    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("Deep Connections (Beta 1.0.0)")
        self.pack(fill=BOTH, expand=1)

        self.cmd1 = StringVar()
        self.mEntry = Entry(self,textvariable=self.cmd1, bd=5, width=60)
        self.mEntry.pack()
        
        self.searchButton = Button(self, text="Search", command=self.OR_system_search)
        self.searchButton.pack()
        
        self.textbox = Text(height=100, width=1000)
        self.textbox.insert(END, "Results")
        self.textbox.pack()
        
        #installButton = Button(self, text="Install", command=self.system_install)
        #installButton.place(x=50, y=150)

    def OR_system_search(self):
        
        searchText = self.cmd1.get()
        tfidf_matrix = tf_idf_bible_fit.transform([searchText])
        x1=tfidf_matrix.todense()
        v=model.predict(x1)[0]
        indices = np.argsort(v)[-50:][::-1]
        topK=[]
        for index in indices:
            print
            firstPart=int2verse[index].split(" ")
            book=" ".join(i for i in firstPart[:-1])
            chapter_verse=firstPart[-1]
            results=select_book_chapter_verse(conn, (book, chapter_verse))[0]
            topK.append((int2verse[index], results[3], v[index]))

        self.textbox.delete("1.0",END)

        print
        ### NOTE: We reverse the topK because as they are written in Tkinter the first show up last. Ironic right? ###
        print(searchText)
        for thing in reversed(topK):
            self.textbox.insert('1.0', thing[0] + "    " + thing[1] + "\n")
            print(thing[0], thing[2])
            print()

    def system_exit(self):
        exit()


root = Tk()
root.geometry("1240x680")
app = Window(root)

root.mainloop()