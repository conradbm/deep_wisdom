
# coding: utf-8

# ## Construct the dataset

# In[1]:


import requests
import re
import string
from bs4 import BeautifulSoup
import bs4
import pickle
import numpy as np
import keras


# In[2]:


with open("data/bible_data_20181123_update.pkl", "rb") as handle:
    bible_data=pickle.load(handle)
kjv_bible_data=bible_data["King James Bible"]


# ## Heatmap 
# 
# Of verse dependecies based on cross reference

# ## Mappings

# In[3]:


int2verse={}
verse2int={}
for i,v in enumerate(kjv_bible_data.keys()):
    int2verse[i]=v
    verse2int[v]=i
int2verse
verse2int


# ## Storing mappings

# In[4]:


kjv_bible_mapping={}
num_verses=len(int2verse.keys())
i=0
for k,obj in kjv_bible_data.items():
    #print(k, obj[0], obj[1])
    mapping=np.zeros((num_verses))
    for cf in obj[1]:
        mapping[verse2int[cf]]=1
    kjv_bible_mapping[k]=[obj[0],mapping]
    i+=1
    
    if i % 1000 == 0:
        print(i)
kjv_bible_mapping["Genesis 1:1"][1]


# ## Save mappings and dataset

# In[5]:


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
    


# In[10]:


pickle_dump(kjv_bible_mapping, "data/kjv_bible_mapping.pkl")
pickle_dump(int2verse, "data/int2verse_mapping.pkl")
pickle_dump(verse2int, "data/verse2int_mapping.pkl")
print("Successful saving.")
