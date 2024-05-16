import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer
import re

train=pd.read_csv('../Data/multimodal_train.csv',encoding="cp437")
test=pd.read_csv('../Data/multimodal_test.csv',encoding="cp437")
valid=pd.read_csv('../Data/multimodal_valid.csv',encoding="cp437")


def Remove_column(df):
    df = df.drop_duplicates()
    df = df.drop(columns=["hasImage", "created_utc","linked_submission_id", "3_way_label"])
    df = df.fillna(0)
    return df

def Remove_empty_URL(df):
    for index, row in df.iterrows():
      if row["image_url"] == 0:
          df= df.drop(index)
    return df

def Remove_empty_txt(df):
    for index, row in df.iterrows():
      if row["clean_title"] == '':
          df= df.drop(index)
    return df
New_train = Remove_column(train)
New_train = Remove_empty_URL(New_train)

New_test = Remove_column(test)
New_test = Remove_empty_URL(New_test)

New_valid = Remove_column(valid)
New_valid = Remove_empty_URL(New_valid)
'''
Analysis of Title :
Converting text to lower case

Removing numbers from the text corpus

Removing punctuation from the text corpus

Removing special characters such as ‘<’, ‘…’ from the text corpus

Removing english stopwords

Stemming words to root words

Removing extra whitespaces from the text corpus

'''
print("Processing  ..............")
stopWords = set(stopwords.words('english'))
ps = PorterStemmer() # For word stemming
wst= WhitespaceTokenizer() 

global c
c =0

def preprocess_text(text):
    global c 
    c =c+1
    print(c)
    stem_sentence=[]
    text =str(text)
    text = text.lower() #Converting text to lower case
    text = re.sub(r"([.,!?])", r" \1 ", text) #Removing punctuation from the text corpus
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    text_tokens  = word_tokenize(text)
    for w in text_tokens:
        #Removing stop words
        if w not in stopWords and w.isalpha()==True and len(w) >=2 and w in words.words():
            stem_sentence.append(ps.stem(w))
            stem_sentence.append(" ")
    
    return "".join(stem_sentence)

#
New_train.clean_title = New_train.clean_title.apply(lambda x: preprocess_text(x))
New_train = Remove_empty_txt(New_train)
New_train.to_csv('../Data/multimodal_train.csv')

# test
New_test.clean_title = New_test.clean_title.apply(lambda x: preprocess_text(x))
New_test = Remove_empty_txt(New_test)
New_test.to_csv('../Data/multimodal_valid.csv')

#valid
New_valid.clean_title = New_valid.clean_title.apply(lambda x: preprocess_text(x))
New_valid = Remove_empty_txt(New_valid)
New_valid.to_csv('../Data/multimodal_test.csv')