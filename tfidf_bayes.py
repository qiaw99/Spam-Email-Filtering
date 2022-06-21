import copy
import string
import numpy as np

from ipynb.fs.full.data_analysis import readData

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

if __name__ == "__main__":
    X, _ = readData("./data/Spam_Emails.csv")
    y = np.load("./data/classes.npy")

    new_words = []
    for word in X:
        for letter in word:
            if letter in string.punctuation:
                word = word.replace(letter,"")   
        new_words.append(word)


    bigram_vect = TfidfVectorizer(ngram_range=(1,1))
    temp = bigram_vect.fit_transform(new_words)
    new_X = temp.toarray()

    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=42)
    del new_X

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"% (len(X_test), (y_test != y_pred).sum()))
    print("f1 score: ", f1_score(y_test, y_pred, average='macro'))