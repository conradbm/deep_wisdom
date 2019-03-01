#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import sqlite3


# In[102]:


with open("../data/bible_data_with_commentary_2018-12-01_21_37_26.pkl", "rb") as handle:
    commentary=pickle.load(handle)


# In[103]:


with open("../data/bible_data_20181123_update.pkl", "rb") as handle:
    bible=pickle.load(handle)


# In[104]:


print("Commentary")
print(commentary[1]["Genesis 1:1"][0])
print("References")
print(commentary[1]["Genesis 1:1"][1])


# In[105]:


bible["King James Bible"]["Genesis 1:1"]


# ## Append Commentary to KJV

# In[106]:


import string

verses=set(bible["King James Bible"].keys())
def clean(text):
    text = "".join([ch for ch in text if ch not in string.punctuation and ch in string.printable])
    #tokens = nltk.word_tokenize(text)
    return text


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


# In[79]:


bible["King James Bible"]["Genesis 1:1"]


# In[78]:


# Build SQLite table
# 1 simple table
# Max commentary length
max_comment_len=max(map(lambda x:len(x[1][2]), bible["King James Bible"].items()))
print(max_comment_len)
max_verse_len=max(map(lambda x:len(x[1][0]), bible["King James Bible"].items()))
print(max_verse_len)


# In[60]:





# In[120]:


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
conn=create_connection("bible.db")
if conn is not None:
    # create projects table
    create_table(conn, create_table_statement)
else:
    print("Error! cannot create the database connection.")


# In[122]:


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
    
conn.commit()


# In[123]:


def select_all(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_Bible")
 
    rows = cur.fetchall()
 
    for row in rows[:5]:
        print(row)
select_all(conn)


# In[124]:


def select_book_chapter_verse(conn,params):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_Bible where T_Bible.book=? and T_Bible.chapter_verse=?", params)
 
    rows = cur.fetchall()
 
    for row in rows:
        print(row)
select_book_chapter_verse(conn, ("Genesis", "1:1"))


# ## Database Established
# 
# So now all you should need to do is the following when you want exact data

# In[127]:


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
conn=sqlite3.connect("bible.db")
if conn is not None:
    #query
    results=select_book_chapter_verse(conn, ("Genesis", "1:1"))
    print(results)
else:
    print("Error! cannot create the database connection.")
    


# In[128]:





# In[ ]:




