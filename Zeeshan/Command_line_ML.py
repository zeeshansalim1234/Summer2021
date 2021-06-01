from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score,confusion_matrix,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from csv import writer,reader

import pickle

def predict(val) :

    loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    predict_vector = vectorizer.transform(val)
    result_prediction = loaded_model.predict(predict_vector)
    return result_prediction

def main():


    print()
    input1=input("Enter Sentence 1 : ")
    input2=input("Enter Sentence 2 : ")
    val=[input1,input2]
    prediction=predict(val)

    result=[]

    for i in range(0,len(prediction)):

        if(prediction[i]==0):
            result.append("Negative")
        else:
            result.append("Positive")

    print("\n")
    print(input1+" : "+result[0])
    print(input2+" : "+result[1])




if __name__ == "__main__":
  main()
