
   
# importing modules

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sn
from sklearn.metrics import confusion_matrix


# data import

df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')

# data exploration
print(df.head())
 
# data cleaning

df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
df = df.rename(columns={"v1":"label","v2":"text"})
print(df.head())

# label encoding

df["label"] = df["label"].map({"ham":1,"spam":0})


# -->removing punctuation.

import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["text"] = df["text"].apply(lambda text: remove_punctuation(text))
print(df['text'])

# -->removing stopwords

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('subject')
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["text"] = df["text"].apply(lambda text: remove_stopwords(text))
print(df['text'])

# --> lemmatization

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
df["text"] = df["text"].apply(lambda text: lemmatize_words(text))
print(df.head())

X = df['text']
y = df['label']

# spiltting the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# vectorization of text

vectorizer = TfidfVectorizer()
print(X_train[1])
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train[1])

# model creation

clf = MultinomialNB()
clf.fit(X_train, y_train)

# prediction on the training set using the created model

y_pred = clf.predict(X_test)
y_pred1 = clf.predict(X_train)

# metrics for the predictions on training set
print()
trainacc = accuracy_score(y_train, y_pred1)
trainf1 = f1_score(y_train, y_pred1)
print("Metrics for training set")
print("accuracy score : " ,trainacc)
print("F1 score : ",trainf1)
cm = confusion_matrix(y_train, y_pred1)
sn.heatmap(cm, annot=True)
print()
# metrics for the predictions on training set

y_pred2 = clf.predict(X_test)
trainacc2 = accuracy_score(y_test, y_pred2)
trainf2 = f1_score(y_test, y_pred2)
print("Metrics for test set")
print("accuracy score : " ,trainacc2)
print("F1 score : ",trainf2)
cm = confusion_matrix(y_test, y_pred2)
sn.heatmap(cm, annot=True)
print()

# label prediction for a foreign dataset

test=pd.read_csv("test.csv")
test["text"] = test["text"].apply(lambda text: remove_punctuation(text))
test["text"] = test["text"].apply(lambda text: remove_stopwords(text))
test["text"] = test["text"].apply(lambda text:  lemmatize_words(text))
x_test = test["text"]
x_test = vectorizer.transform(x_test)
y_pred = clf.predict(x_test)
test["label"] = y_pred
test["label"]=df["label"].map({1:"ham",0:"spam"})
print("Label prediction for foreign dataset")
print(test)

