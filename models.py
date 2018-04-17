import nltk
import os, sys, email,re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.regexp import RegexpTokenizer

from subprocess import check_output

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

eng_stopwords = set(stopwords.words('english'))
def clean_text(text):
    """
    This Function was taken from one of the notebooks on kaggle where we got the Enron Data Set
    https://www.kaggle.com/jaykrishna/topic-modeling-enron-email-dataset.
    From looking it over this function takes some text and 
    """
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)


def svm_preprocess(df):
    """
    Does preprocessing and train/test split for to train a SVM
    """
    df = pd.read_csv('recipient_data.csv')
    df = df[:][~df.content.isnull()]
    df = df[:][~df.gender.isnull()]
    df.content = df.content.str.replace('\n'," ")
    df["clean_content"] = df.content.apply(clean)
    wordvector = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.4, min_df=5)
    wordvector_fit = wordvector.fit_transform(df.clean_content)
    X_train, X_test, y_train, y_test = train_test_split(wordvector_fit, df.gender, test_size=0.25)
    return X_train, X_test, y_train, y_test

def SVMregulizer(X_train, X_test, y_train, y_test, Cs):
    """
    Runs multiple SVMs at different values of C(reciprocal of regulariazation) and returns a list of accuracies
    """
    accs = []
    for c in Cs:
        model = LinearSVC(C = c)
        model = model.fit(X_train, y_train)
        pred_y = model.predict(X_test)
        acc = sum(y_test.values == pred_y)/len(pred_y)
        accs.append(acc)
    return accs