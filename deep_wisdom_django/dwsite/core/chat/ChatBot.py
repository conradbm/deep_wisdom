import numpy as np
import re
import sys
from collections import defaultdict
import random
import nltk
import string

class ChatBot:
    
    """
    Members we expect PeterChatBot to have after proper construction.
    """
    bible_data=None
    pattern=None
    corpus=None
    corpusClean=None
    word2idx = None
    idx2word = None
    vocab_size = None
    bigram_counts = None
    bigram_possibilities = None
    P = None
    
    options={'Prophets':r"""(Isaiah|Jeremiah|Lamentations|
                              Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|
                              Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|
                              Malachi)""",
             'Apostles':r"""(Romans|.*Corinthians.*|Galatians|Philippians|
                              Ephesians|Colossians|.*Thessalonians.*|Hebrews|
                              .*Timothy.*|Titus|Philemon|
                              .*John.*|Revelation|
                              .*Peter.*|James)""",
             'Old Testament':r"""(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|
                              Joshua|Judges|Ruth|
                              .*Saumuel.*|.*Kings.*|.*Chronicles.*|
                              Ezra|Nehemiah|Esther|.*Psalm.*|.*Proverb.*|
                              Ecclesiastes|.*Song.*|Isaiah|Jeremiah|Lamentations|
                              Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|
                              Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|
                              Malachi)""",
             'New Testament':r"""(Matthew|Mark|Luke|Acts|
                              Romans|.*Corinthians.*|Galatians|Philippians|
                              Ephesians|Colossians|.*Thessalonians.*|Hebrews|
                              .*Timothy.*|Titus|Philemon|.*John.*|Revelation|
                              .*Peter.*|Jude|James
                                )""",
             'Paul':r"(Romans|.*Corinthians.*|Galatians|Philippians|Ephesians|Colossians|.*Thessalonians.*|Hebrews|.*Timothy.*|Titus|Philemon)",
             'John':r"(.*John.*|Revelation)",
             'James':r"(James)",
             'Peter':r"(.*Peter.*)",
             'Jesus':r"(Matthew|Mark|Luke|John)",
             'Moses':r"(Genesis|Exodus|Leviticus|Numbers|Deuteronomy)",
             'Joshua':r"(Joshua)",
             'Job':r"(Job)",
             'David':r"(.*Samuel.*|.*Psalm.*)",
             'Solomon':r"(.*Proverb.*|.*Ecclesiastes.*|.*Song.*)",
             'Daniel':r"(Daniel)",
             'Isaiah':r"(Isaiah)",
             'Jeremiah':r"(Jeremiah|Lamentations)",
             'Ezekiel':r"(Ezekiel)",
             'Ruth':r"(Ruth)",
             'Nehemiah':r"(Nehemiah)",
             'Esther':r"(Esther)",
             'Hosea':r"(Hosea)",
             'Joel':r"(Joel)",
             'Jonah':r"(Jonah)",
             'Micah':r"(Micah)",
             'Nahum':r"(Nahum)",
             'Habakkuk':r"(Habakkuk)",
             'Zephaniah':r"(Zephaniah)",
             'Haggai':r"(Haggai)",
             'Zechariah':r"(Zechariah)",
             'Malachi':r"(Malachi)"
            }
    
    
    def __init__(self, who, verbose=True):
        print("Building {} Chat Bot".format(who))
        assert(who in list(self.options.keys()), "{} is not a valid character, try someone else.".format(who))
        self.who=who
        
    def getCorpus(self, pattern, dw_bible_mapping):
        
        """
        pattern should be a regular expression that will parse down any books containing
        some pattern of interest. For Peter this should just be (.*Peter.*).
        """
        self.corpus=[]
        
        # Get corpus of hits within this character's pattern
        for i in list(dw_bible_mapping.keys()):
            result=re.match(pattern,i)
            if not result is None:
                #print(result)
                self.corpus.append(dw_bible_mapping[i])
        
        self.corpusClean=[]
        for i in self.corpus:
            newVerse=[]
            for j in nltk.word_tokenize(i):
                newVerse.append(j.lower())
            self.corpusClean.append(newVerse)
        return self.corpusClean
    
    def getLookupTables(self, corpusClean):
        w2i={}
        i2w={}
        allWords = []
        for i in corpusClean:
            allWords = allWords + i
        for i,word in enumerate(allWords, 1):
            #print(i)
            w2i[word]=i
            i2w[i]=word
        self.word2idx={'UNK':0}
        self.idx2word={0:'UNK'}
        counter=1
        for k,v in w2i.items():
            self.word2idx[k]=counter
            self.idx2word[counter]=k
            counter+=1
        self.vocab_size=len(list(self.word2idx.items()))
        return self.word2idx, self.idx2word, self.vocab_size
    
    def getBigrams(self, corpusClean):
        """
        Usage:
            @param: sentences = ["this is a sentence1", "this is a sentence2", ..."this is a sentenceN"]
        """
        self.bigram_counts={}
        for sentence in self.corpusClean:
            N=len(sentence)
            for j in range(0,N-1):
                k=(sentence[j], sentence[j+1])
                try:
                    self.bigram_counts[k]+=1
                except Exception as e:
                    self.bigram_counts[k]=1
        return self.bigram_counts
    
    def getBigramPossibilities(self, bigram_counts, corpusClean):
        # Get list of bigram possibilities
        self.bigrams_possibilities=defaultdict(dict) #{k:[(w2j:count), w2, ..., wN]}
        found=set()
        for sentence in corpusClean:
            N=len(sentence)
            for j in range(0,N-1):
                # Get bigram lists
                w1=sentence[j]
                w2=sentence[j+1]
                bg_count = bigram_counts[(w1,w2)]    
                if bg_count > 0:
                    self.bigrams_possibilities[w1][w2] = bg_count
                else:
                    print("Skipping {} {}".format(w1, w2))
        return self.bigrams_possibilities
    
    def getMarkovMatrix(self, bigram_possibilities, vocab_size, w2i, i2w, smooth=0.01):
        self.P = np.ones((vocab_size,vocab_size), dtype=np.float)*smooth
        #print(self.P.shape)
        for w1,v in bigram_possibilities.items():
            for w2, c in v.items():
                #print("{}({}) {}({})".format(w1, word2idx[w1], w2, word2idx[w2]))
                self.P[w2i[w1],w2i[w2]]+=c
                #if w1 == 'suffering' or w2 == 'suffering':
                #    print(P[word2idx[w1],word2idx[w2]])
        self.P = self.P**1.75
        self.P /= self.P.sum(axis=1, keepdims=True)

        return self.P
    
    # INTERFACE WITH BOT
    def getNextWord(self, word, perfect=False, pure_chance=False, perfect_threshold=0.5):
        #np.random.choice(numpy.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
        assert(not self.word2idx is None, "Please construct Chat Bot first.")
        assert(not self.idx2word is None, "Please construct Chat Bot first.")
        assert(not self.P is None, "Please construct Chat Bot first.")


        # IF THE USERS TYPE IN A SERIES OF WORDS
        words=nltk.word_tokenize(word)
        if len(words)>1:
            word=words[-1].lower()

        # START CONNECTING IN THE MARKOV CHAINS
        try:
            i=self.word2idx[word]
        except:
            i=self.word2idx['UNK']
        #print(i)
        if perfect:
            j=self.P[i,:].argmax()
            # If UNK token, just choose randomly.
            if j == 0:
                j=np.random.choice(np.arange(0, self.P.shape[1]), p=list(self.P[i,:]))
        else:
            if pure_chance:
                j=np.random.choice(np.arange(0, self.P.shape[1]), p=list(self.P[i,:]))
            else:
                # 50/50 explore exploit
                if random.uniform(0, 1) < perfect_threshold:
                    j=np.random.choice(np.arange(0, self.P.shape[1]), p=list(self.P[i,:]))
                else:
                    j=self.P[i,:].argmax()
        word=self.idx2word[j]
        return word

    # HELPER FUNCTION FOR INTERFACE
    def makeSentenceFromJumble(self, sent):
        if not isinstance(sent, str):
            sent = " ".join([i for i in sent])
        w0=""
        for word in nltk.sent_tokenize(sent):
            for w in nltk.word_tokenize(word):
                if w in string.punctuation:
                    w0+=w
                else:
                    w0+=" "
                    w0+=w
        return w0
    
    # INTERFACE WITH USER
    def askTopic(self, word, results, sequenceLength=50, perfectFitLength=5, perfectThreshold=0.5, verbose=False):
        
        """
        Idea ...
        1. Get topic results from DW
        2. Sort through just pauls take on this topic
        3. Construct MC from this
        4. Let paul start talking from within this chain
        """
        #self.dw_bible_mapping=dw.query(word, K=100)
        self.dw_bible_mapping=results
        self.pattern=self.options[self.who]
        self.corpusClean = self.getCorpus(self.pattern, self.dw_bible_mapping)
        
        # Determine if this speaker is proficient on the topic
        if len(self.corpusClean) < 1:
            if verbose:
                print("Speaker: {}\nTopic: {}\nCorpus Size: {}".format(self.who, word, len(self.corpusClean)))
                print("{} NOT PROFICIENT ENOUGH ON {} TO EXPOUND.".format(self.who, word))
            return self.who, len(self.corpusClean), None

        
        # Otherwise build Markov Model for this speaker on the relevant texts returned from deepwisdom
        self.word2idx, self.idx2word, self.vocab_size = self.getLookupTables(self.corpusClean)
        self.bigram_counts = self.getBigrams(self.corpusClean)
        self.bigrams_possibilities = self.getBigramPossibilities(self.bigram_counts, self.corpusClean)
        self.P = self.getMarkovMatrix(self.bigrams_possibilities, self.vocab_size, self.word2idx, self.idx2word, smooth=0.01)
        
        if verbose:
            print("Speaker: {}\nTopic: {}\nCorpus Size: {}".format(self.who, word, len(self.corpusClean)))
        sentence=[]

        # Start off perfectly
        for _ in range(perfectFitLength):
            sentence.append(word)
            word=self.getNextWord(word, perfect=True, perfect_threshold=perfectThreshold)
        
        # Continue probabilistically
        for _ in range(sequenceLength):
            sentence.append(word)
            word=self.getNextWord(word, pure_chance=True)
        
        if verbose:
            for w in self.makeSentenceFromJumble(sentence):
                sys.stdout.write(w)
            print("\n")
        return self.who, len(self.corpusClean), self.makeSentenceFromJumble(sentence)

def chat_responses_reshape(searchText, data_dict, verbose=False):
    keys=set(list(ChatBot.options.keys()))
    bots=[]
    for k in keys:
        cb=ChatBot(who=k)
        bots.append(cb)
   
    #results=dw.query(t, K=50)
    results_agg={}
    for b in bots:
        who, size, sent = b.askTopic(searchText, data_dict)
        if sent is None:
            continue
        #if who in ["Old Testament", "New Testament", "Prophets", "Apostles"]:
        results_agg[who]=sent
        #else:
        #    results_specific[who]=sent
        if verbose:
            print("{}-{}-{}".format(who, size, sent))
    

    return results_agg

if __name__ == "__main__":
    def test_chatbot():
        bots=[]
        #s=set(["Old", "New"])
        keys=set(list(ChatBot.options.keys()))
        #d=keys.difference(s)

        for k in keys:
            cb=ChatBot(who=k)
            bots.append(cb)
        topics=['hope', 'life', 'sheep', 'resurrection']
        for t in topics:
            #print("Topic {}".format(t))
            results=dw.query(t, K=50)
            for b in bots:
                who, size, sent = b.askTopic(t, results)
                if sent is None:
                    continue
                print("{}-{}-{}".format(who, size, sent))
            input("Ready for next topic?")        
    test_chatbot()
