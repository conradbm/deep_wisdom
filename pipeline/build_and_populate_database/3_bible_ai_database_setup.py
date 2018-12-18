
# coding: utf-8

# In[1]:


import pickle
import sqlite3


# In[6]:


with open("data/bible_data_with_commentary_2018-12-01_21_37_26.pkl", "rb") as handle:
    commentary=pickle.load(handle)


# In[7]:


with open("data/bible_data_20181123_update.pkl", "rb") as handle:
    bible=pickle.load(handle)


# In[8]:


print("Commentary")
print(commentary[1]["Genesis 1:1"][0])
print("References")
print(commentary[1]["Genesis 1:1"][1])


# In[9]:


bible["King James Bible"]["Genesis 1:1"]


# ## Append Commentary to KJV

# In[10]:


import string
import re
import nltk
verses=set(bible["King James Bible"].keys())
def clean(text):
    text = "".join([ch for ch in text if ch not in string.punctuation and ch in string.printable])
    tokens = " ".join([re.sub(r'[0-9]','',w) for w in nltk.word_tokenize(text)])
    return tokens


for k,v in commentary[1].items():
    
    #Add commentary
    cleaner=clean(v[0]).replace("\n", "")
    bible["King James Bible"][k].append(cleaner)
    
    #Refs
    refs=list(filter(lambda x:x[0] in verses, v[1]))
    refs=list(set(map(lambda x:x[0], refs)))
    for r in refs:
        bible["King James Bible"][k][1].append(r)
    bible["King James Bible"][k][1]=",".join(i for i in list(set(bible["King James Bible"][k][1])))
    #Update refs
    print(k)
    #print(cleaner)
    #print(refs)
    #print(bible["King James Bible"][k][1])
    #input("...")


# In[11]:


bible["King James Bible"]["Genesis 1:1"]


# In[12]:


# Build SQLite table
# 1 simple table
# Max commentary length
max_comment_len=max(map(lambda x:len(x[1][2]), bible["King James Bible"].items()))
print(max_comment_len)
max_verse_len=max(map(lambda x:len(x[1][0]), bible["King James Bible"].items()))
print(max_verse_len)


# In[60]:





# In[13]:


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

        
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

create_table_statement="""CREATE TABLE IF NOT EXISTS T_Bible (
                         id integer PRIMARY KEY, 
                         book text,
                         chapter_verse text,
                         content text,
                         cross_references text,
                         commentary text
                        );"""

# create a database connection
conn=create_connection("data/bible.db")
if conn is not None:
    # create projects table
    create_table(conn, create_table_statement)
else:
    print("Error! cannot create the database connection.")
    
print("Table created.")
conn.commit()
input("...")

# In[15]:


def insert_stuff(conn, stuff):

    sql = ''' INSERT INTO T_BIBLE(book,chapter_verse,content,cross_references,commentary)
              VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, stuff)
    
for k,v in bible["King James Bible"].items():
    book=" ".join(i for i in k.split(" ")[:-1])
    chapter_verse=k.split(" ")[-1]
    content=v[0]
    cross_refs=v[1]
    comment=v[2]
    stuff=(book, chapter_verse, content, cross_refs, comment)
    #print(stuff)
    #input("...")
    insert_stuff(conn, stuff)
    
print("Inserted all data.")
conn.commit()
print(conn.cursor().execute("SELECT * FROM T_Bible where T_Bible.book=? and T_Bible.chapter_verse=?",("Genesis","1:1")).fetchall()[:5])
input("...")

# connect
conn=sqlite3.connect("data/bible.db")
if conn is not None:
    pass
else:
    print("Error! cannot create the database connection.")

print("Reconnecting to bible.db")
input("....")
print(conn.cursor().execute("SELECT * FROM T_Bible where T_Bible.book=? and T_Bible.chapter_verse=?",("Genesis","1:1")).fetchall()[:5])
conn.commit()
conn.close()
