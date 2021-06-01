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

labels=['positive','negative']


def extract_database(address):

    df = pd.read_csv(address)
    return df


def pre_process_tweets(tweets):

    for i in range(0, len(tweets)):

        if (tweets[i] is not None):
            tweets[i] = tweets[i].lower()  # To lower case
            tweets[i] = tweets[i].replace('@', '')  # remove @
            tweets[i] = tweets[i].replace('#', '')  # remove #
            tweets[i] = remove_urls(tweets[i])  # remove URL
            tweets[i] = remove_emojis(tweets[i])  # remove emojis
            tweets[i] = "".join(j for j in tweets[i] if j not in (
            "?", ".", ";", ":", "!", "-", ",", "[", "]", "(", ")", "’", "‘", '"', "$", "'", "“", "”", "•", "=", "+",
            "%", "/", "&", "|", "~"))  # remove punctuations

    return tweets


def remove_urls (str):

    str = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str, flags=re.MULTILINE)
    return(str)


def remove_emojis(data):

    emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoji, '', data)


def bernoulli_model(tweets,corresponding_labels,predict_tweets,status):

    X_train, X_test, y_train, y_test = train_test_split(tweets, corresponding_labels)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, decode_error="ignore")
    train_x_vectors = vectorizer.fit_transform(X_train)
    test_x_vectors = vectorizer.transform(X_test)
    predict_vectors=vectorizer.transform(predict_tweets)
    model = BernoulliNB()
    model.fit(train_x_vectors, y_train)  # trained the model
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))

    y_prediction = model.predict(test_x_vectors)
    result_prediction=model.predict(predict_vectors)

    print("\nStatistic of ML model :")
    print("\nBernoulliNB Accuracy (" +status+" data) : "+str(metrics.accuracy_score(y_test, y_prediction)))
    print("BernoulliNB F1 Score (" +status+" data) : "+str(metrics.f1_score(y_test, y_prediction, average='weighted', labels=np.unique(y_prediction))))
    generate_heatmap(y_test, y_prediction,status)

    return result_prediction


def logistic_regression(tweets,corresponding_labels,status):

    X_train, X_test, y_train, y_test = train_test_split(tweets, corresponding_labels)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, decode_error="ignore")
    train_x_vectors = vectorizer.fit_transform(X_train)
    test_x_vectors = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(train_x_vectors, y_train)  # trained the model
    y_prediction=model.predict(test_x_vectors)


    print("\nLogistic Classifier Accuracy (" + status + " data) : " + str(metrics.accuracy_score(y_test, y_prediction)))
    print("logistic Classifier F1 Score (" + status + " data) : " + str(metrics.f1_score(y_test, y_prediction, average='weighted', labels=np.unique(y_prediction))))
    #y_prediction = model.decision_function(test_x_vectors)
    #generate_ROC_curve(y_test,y_prediction)


# decision function output tells you how far the pointis from the classification plane

def generate_ROC_curve(y_test,y_prediction):

    logistic_fpr, logistic_tpr, threshold = roc_curve(y_test,y_prediction,pos_label=['positive','negative'])
    auc1=auc(logistic_fpr,logistic_tpr)

    plt.figure(figsize=(5,5),dpi=100)
    plt.plot(logistic_fpr,logistic_tpr,marker='.',label='Logistic (auc=0.3%f)' % auc1)

    plt.xlabel("False positive rate -->")
    plt.ylabel("True positive rate -->")
    plt.legend()


def generate_heatmap(y_test,y_prediction,status):

    f = plt.figure()
    f.set_figwidth(25)
    f.set_figheight(25)

    cm = confusion_matrix(y_test, y_prediction, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.title("Heatmap for "+status+" dataset",fontsize=12)
    plt.xlabel("Prediction",fontsize=12)
    plt.ylabel("Actual",fontsize=12)
    plt.show()

def add_column_in_csv(input_file, output_file, transform_row):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)


def prepend(list, str):
    # Using format()
    str += '{0}'
    list = ((map(str.format, list)))
    return list

def main():


    df_train=extract_database("C:\\Users\\hp\\PycharmProjects\\SEC\\NLP\\IMDB Dataset.csv")
    df_predict=extract_database("C:\\Users\\hp\\PycharmProjects\\SEC\\NLP\\tweets.csv")
    train_tweets=df_train['review'].to_list()
    train_corresponding_labels= df_train['sentiment'].to_list()
    predict_tweets=df_predict['text'].to_list()

    train_pre_processed_tweets=pre_process_tweets(train_tweets)

    for i in range(0,len(predict_tweets)):
        predict_tweets[i]=str(predict_tweets[i])

    predict_pre_processed_tweets=pre_process_tweets(predict_tweets)
    indices=[]
    #bernoulli_model(binary_tweets,binary_labels,"raw") #raw dataset
    prediction=bernoulli_model(train_pre_processed_tweets,train_corresponding_labels,predict_pre_processed_tweets,"pre-processed") #pre-processed dataset

    counter=0
    print("\nNumber of tweets :"+str(len(prediction)))
    for i in range(0,len(prediction)):
        if(prediction[i]=="negative"):
            indices.append(i)
            counter+=1

    print("Number of negative tweets: "+str(counter))
    print("\n\n\n\n")
    print(indices)

    for i in range(0,len(predict_tweets)):
        for j in range(0,len(indices)):
            if(indices[j]==i):
                print(predict_tweets[i])

    #add_column_in_csv('C:\\Users\\hp\\PycharmProjects\\SEC\\NLP\\tweets.csv', 'output.csv', lambda row, line_num: row.append(prediction))

    #logistic_regression(binary_tweets,binary_labels,"raw")
    #logistic_regression(pre_processed_tweets, binary_labels, "pre-processed")


if __name__ == "__main__":
  main()
