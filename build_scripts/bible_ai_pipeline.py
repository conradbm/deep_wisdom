#!/usr/bin/env python
# coding: utf-8

# ## bible.ai
# 
# The objective of this script is to crawl through the entire archive of https://www.biblehub.com. By doing such, as can gain content relating to
# every single bible version, its verse by verse text, and all of the related cross references.
# This is a three-fold project with the objective to discover new relationships between bible verses for pastors, priests, or any spiritual leader.
# 
# ### data.ai - Data Pipeline (<em> Phase 1 of 4 </em>)
# 
# Phase 1, collect the data with cross references. This is done by crawling the entire https://www.biblehub.com website. We leverage some structure in the site and predictability with related links, and by such, we construct a large data set with everything we need in our vision.
# 
# ### bible.ai - Data Construction (<em> Phase 2 of 4 </em>)
# 
# Phase 2, we seek to clean and shape the data. A necessary part to any <em>Machine Learning</em> application. 
# 
# ### bible.ai - Model Training (<em> Phase 3 of 4 </em>)
# 
# Phase 3, we seek to utilize the cross references found in the bible as training data. We will learn structure of verses by context and relate that to their cross references. After we do that, we will use a `Recurrent Neural Network (RNN)` to predict based on the sequence of verses without cross references, which ones they should be associated with, to hopefully discover new connections in the bible that were previously not possible to know.
# 
# ### bible.ai - Model Embedding/Deployment (<em> Phase 4 of 4 </em>)
# 
# Phase 4, the goal at this final stage is to have a clean, serialized model that can take any string, from the bible or not, and refer you to exact places in the bible that we believe are highly related to the text you are researching. This can become useful when studying external books, such as `Plato's Republic` or `The Apostolic Fathers` to discover similar verses, that are not explicitely linked to the bible. The goal of this is to augment the users current capability of research with a tool that blends state of the art predictive analysis with real biblical connectivity, previously unseen. 
# 
# 
# ## Further Research
# 
# We want to build the best product for our customers. In this spirit, why stop with the bible? Do you have literature with well known inter-literary references, or a network of references that is `closed-form`? If so, we can expand our work here from just within the bible to accross multiple domains of literature to give you high verse by verse probabilities that the words you're seeking are related. This type of extension makes literary analysis possible between domains such as psychology, social sciences, philosophy, and much more. Please P.M. to discuss details on your custom solution.
# 
# This could also be considered as a general application, `lit.ai` to mitigate gaps between social sciences and machine learning.

# ## Libraries

# In[1]:


import requests
import re
import string
from bs4 import BeautifulSoup
import bs4
import pickle
from collections import defaultdict


# ## Data Pipeline

# In[6]:



"""
Provided a base_prefix, and a next suffix, we can crawl the entire biblehub
website to collect their bible versions, data, and cross references. The
objective of this function is to be called in a loop-like fashion, updating the
`next_suffix` part of the url. This should always start with a / because we simply
want to do string addition and have a live url at any point. Makes things easier.
Expected output should look like below:

{"NIV":{"Genesis 1:1":("in the beginning god created ...", ["John 1:!", "Revelation 4:14, ..."]),
        "Genesis 1:2":("...",[...])
        ...}
 "KJV":{...},
 ...
 }
"""
def populate_data_from_link(base_prefix, next_suffix, 
                            debug=True, terminating_verse="Revelation 22:21"):
    
    if debug:
        print("Scraping text.")
        
    # Get content
    link=base_prefix+""+next_suffix
    page = requests.get(link)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    # Get verse
    verse=soup.find(id="topheading").text
    verse=" ".join(verse.split(" ")[1:-1])
    if debug:
        print("Collecting data on ", verse)
        print("Book: ", verse.split(" ")[0])
        print("Chapter: ", verse.split(" ")[1].split(":")[0])
        print("Verse: ", verse.split(" ")[1].split(":")[1])

    # Populate cross references
    crossrefs=[]
    for nested in soup.findAll(attrs={'crossverse'}): 
        crossrefs.append(nested.text)
    
    # Populate data 
    ## P.S. There does exist some wholes in Rev 22:7,8 and probably more.
    ## In order to mitigate this, we should simply keep getting next sibling's text
    ##   until it is another Tag object (or another THING we see a pattern form).
    ## There may be more work to do here to correctly get 100% of each verse.
    
    def get_all_inner_content(nested):
        content=""
        # Keep going until the next `Tag` you find has `href` as an attribute
        sib=nested.nextSibling
        while True:
           #print("stuck")
            #print(sib)
            #input("..")
            if isinstance(sib, bs4.element.Tag):
                try:
                    if sib["class"][0]=='versiontext':
                        break
                except:
                    pass
                content+=sib.text
            else:
                content+=sib
                
            sib=sib.nextSibling
            if sib is None:
                break
        # Keep getting next siblings Tag text or raw text until see we `class=textversion`
        
        return content
    
    for nested in soup.findAll(attrs={'versiontext'}):        
        content=get_all_inner_content(nested)

        # New, beta
        version_dict[nested.text][verse]=[content, crossrefs, next_suffix]
        # Old, chopy
        #version_dict[nested.text][verse]=[str(nested.nextSibling.nextSibling), crossrefs]
    
        if debug:
            print("Content: ", version_dict[nested.text][verse][0])
            print("References:" ,version_dict[nested.text][verse][1])
    #print(version_dict)
    
    print()
    # Set up next page to crawl
    nextLink=""
    for thing in soup.find(id={'topheading'}):
        if isinstance(thing, bs4.element.Tag):
            nextLink=thing.get_attribute_list('href')[0]
    next_suffix=nextLink
    
    if debug:
        print("Next link ", str(base_prefix+next_suffix))
    
    if verse == terminating_verse:
        return (False, False)
    else:
        return base_prefix,next_suffix

    
def crawl_link(base_prefix, next_suffix):
    base=base_prefix
    s=next_suffix
    while True:

        base,s_next=populate_data_from_link(base,
                                            s,
                                            debug=True)
        if base != False and s_next != False:
            s=s_next
            print("Completed: ", base+s)
            continue
        else:
            "Finished data collection."
            with open("bible_data_20181129_update.pkl", "wb") as handle:
                pickle.dump(version_dict, handle)
            break
    return version_dict

base_prefix="https://biblehub.com"
next_suffix="/genesis/1-1.htm" 
version_dict=defaultdict(dict)
version_dict=crawl_link(base_prefix, next_suffix)


# In[73]:



version_dict["King James Bible"]["Revelation 22:7"]


# ## Sanity check
# 
# Make sure that the stuff is being brought in correctly.

# In[74]:


import numpy as np

print("(Revelation 22:21);", version_dict["King James Bible"]['Revelation 22:21'][0])
print("Realted Verses:\n")
for cf in version_dict["King James Bible"]['Revelation 22:21'][1]:
    print("(" + cf+ ")",version_dict["King James Bible"][cf][0])


# In[198]:


version_dict["King James Bible"]["Luke 9:12"]


# In[267]:


with open("bible_data.pkl", "rb") as handle:
    bible_data=pickle.load(handle)

print(len(bible_data.items())) #version count
len(bible_data["Revelation 21:7"])


# In[47]:


x=None
x=type(x)
x==NoneType


# In[41]:





# In[ ]:




