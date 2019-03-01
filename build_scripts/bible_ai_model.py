#!/usr/bin/env python
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
kjv_bible_data=pickle_load("../data/bible_data_20181123_update.pkl")
kjv_bible_mapping=pickle_load("../data/kjv_bible_mapping.pkl")
int2verse=pickle_load("../data/int2verse_mapping.pkl")
verse2int=pickle_load("../data/verse2int_mapping.pkl")
print("Load complete.")


# In[ ]:





# In[4]:


import nltk
import string
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#nltk.download('punkt')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text, stemmer = PorterStemmer()):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

corpus=list(map(lambda x:x[1][0], kjv_bible_mapping.items()))
corpus[:5]


# In[5]:


tfidf_vectorizer = TfidfVectorizer(#tokenizer=tokenize, 
                                   stop_words="english",
                                   lowercase=True, norm='l2')
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
pickle_dump(tfidf_parallel, '../data/tfidf_bible_matrix.pkl')
pickle_dump(tfidf_fit, '../data/tfidf_bible_fit.pkl')
tf_idf_bible_matrix=pickle_load('../data/tfidf_bible_matrix.pkl')
tf_idf_bible_fit=pickle_load('../data/tfidf_bible_fit.pkl')

tf_idf_bible_matrix[:4]


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

print(X.shape)
print(y.shape)
print(len(corpus))
print(len(list(kjv_bible_mapping.keys())))


# In[8]:


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
model_path="weights-improvement-01-54.2576.hdf5"
def load_trained_model(weights_path, X, y):
    model = create_model(X,y)
    model.load_weights(weights_path)
    print("Loaded")
    return model

model=load_trained_model(model_path,X,y)



# ## Train step
# Be careful before you do this, may take 20 mins.

# In[ ]:


#filepath="../data/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

rnn_model = create_model(X,y)
rnn_model.summary()

rnn_model.fit(X, y,
              batch_size=128,
              epochs=1,
              callbacks=callbacks_list)


# In[ ]:





# ## Test cases
# 
# Enter search text...joy hope peace
# 
# Psalm 71:23 My lips shall greatly rejoice when I sing unto thee; and my soul, which thou hast redeemed.
# 
# Psalm 64:10 The righteous shall be glad in the LORD, and shall trust in him; and all the upright in heart shall glory.
# Psalm 97:12 Rejoice in the LORD, ye righteous; and give thanks at the remembrance of his holiness.
# 
# Isaiah 66:10 Rejoice ye with Jerusalem, and be glad with her, all ye that love her: rejoice for joy with her, all ye that mourn for her:
# 
# - Very expected results.
# 
# Enter search text...Unto whom it was revealed, that not unto themselves, but unto us they did minister the things, which are now reported unto you by them that have preached the gospel unto you with the Holy Ghost sent down from heaven; which things the angels desire to look into.
# 
# John 3:33 He that hath received his testimony hath set to his seal that God is true.
# 
# Ezekiel 1:1 Now it came to pass in the thirtieth year, in the fourth month, in the fifth day of the month, as I was among the captives by the river of Chebar, that <strong>the heavens were opened, and I saw visions of God.</strong>
# 
# Romans 12:6 Having then <strong>gifts differing according to the grace that is given to us, whether prophecy, let us prophesy according to the proportion of faith</strong>;
# 
# 1 Peter 1:25 But the word of the Lord endureth for ever. And this is the word which by the gospel is preached unto you.
# 
# Acts 2:2 And suddenly there came a sound from heaven as of a rushing mighty wind, and it filled all the house where they were sitting.
# 
# <strong>Acts 2:4</strong> And they were all filled with the Holy Ghost, <strong>and began to speak with other tongues, as the Spirit gave them utterance.</strong>
# 
# - It drew the connection to the `why` of the verse. Very interesting.
# 
# Enter search text...despair
# 
# Galatians 5:19 Now the <strong>works of the flesh</strong> are manifest, which are these; Adultery, fornication, uncleanness, lasciviousness,
# 
# Psalm 140:5 The proud have hid a snare for me, and cords; they have spread a net by the wayside; they have set gins for me. Selah.
# 
# Psalm 57:4 My soul is among lions: and I lie even among them that are set on fire, even the sons of men, whose teeth are spears and arrows, and their tongue a sharp sword.
# 
# Psalm 1:1 Blessed is the man that walketh not in the <strong>counsel of the ungodly</strong>, nor standeth in the <strong>way of sinners</strong>, nor sitteth in the seat of the scornful.
# 
# Isaiah 11:4 But with righteousness shall he judge the poor, and reprove with equity for the meek of the earth: and he shall smite the earth with the rod of his mouth, and <strong>with the breath of his lips shall he slay the wicked</strong>.
# 
# Ecclesiastes 2:24 <strong>There is nothing better for a man, than that he should eat and drink, and that he should make his soul enjoy good in his labour.</strong> This also I saw, that it was from the hand of God.
# 
# - It drew an example of despair (Phsa 57:4) and related it to characteristics of people (the wicked). It also went so far as to suggest verses that could remedy it (Eccl 2:24).
# 
# **Amazing**
# 
# Enter search text...wisdom understanding
# 
# Job 28:28 And unto man he said, <strong>Behold, the fear of the Lord, that is wisdom; and to depart from evil is understanding.</strong>
# 
# Proverbs 1:7 The fear of the LORD is the beginning of knowledge: but fools despise wisdom and instruction.
# 
# 1 Chronicles 29:25 And the LORD magnified Solomon exceedingly in the sight of all Israel, and bestowed upon him such royal majesty as had not been on any king before him in Israel.
# 
# 2 Chronicles 9:22 And king Solomon passed all the kings of the earth in riches and wisdom.
# 
# Proverbs 8:14 Counsel is mine, and sound wisdom: I am understanding; I have strength.
# 
# 1 Kings 3:12 Behold, I have done according to thy words: lo, I have given thee a wise and an understanding heart; so that there was none like thee before thee, neither after thee shall any arise like unto thee.
# 
# - Absolutely, the most relevent 6 verses in the bible. King solomon, Job directly talking about it, and Christ himself.
# 
# Enter search text...plan path predestined
# 
# Psalm 7:9 Oh let the wickedness of the wicked come to an end; but establish the just: for the righteous God trieth the hearts and reins.
# 
# Daniel 5:7 The king cried aloud to bring in the astrologers, the Chaldeans, and the soothsayers. And the king spake, and said to the wise men of Babylon, Whosoever shall read this writing, and shew me the interpretation thereof, shall be clothed with scarlet, and have a chain of gold about his neck, and shall be the third ruler in the kingdom.
# 
# Psalm 19:10 More to be desired are they than gold, yea, than much fine gold: sweeter also than honey and the honeycomb.
# 
# Psalm 25:4 Shew me thy ways, O LORD; teach me thy paths.
# 
# Matthew 24:45 Who then is a <strong>faithful and wise servant</strong>, whom his lord hath made ruler over his household, to give them meat in due season?
# 
# Romans 9:20 Nay but, O man, who art thou that repliest against God? <strong>Shall the thing formed say to him that formed it, Why hast thou made me thus?</strong>
# 
# - I told it nothing about being `formed` before I was born, yet it found the verse. 
# - I also told it nothing about parables, yet it found the faithful and wise servant, going along perfectly with the plan path and destiny of a child of God.
# 
# ## Conclusion
# 
# The cross reference finder has mixed expert opinion with NLP turning this into a powerful tool drawing some very novel and relevent links between verses within the text.

# In[10]:


searchText=input("Enter search text...")
print
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


# In[29]:


print("Actual cross references")

s=set()
i=0
for k,v in kjv_bible_data['King James Bible'].items():
    for cf in v[1]:
        s.add(cf)
print(len(s))
print(len(set(kjv_bible_data['King James Bible'])))
for i in set(kjv_bible_data['King James Bible']):
    if i not in s:
        print(i)


# In[28]:


kjv_bible_data['King James Bible']


# In[ ]:




