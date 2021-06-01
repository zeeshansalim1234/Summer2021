import nltk,numpy
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

"""lemmatizer = WordNetLemmatizer()

nltk.download()

stemmer = SnowballStemmer("english")

sentence = "My name is Zeehsan Salim Chougle ....."

print(stemmer.stem("generously"))     #Gives the root word

tokens = nltk.word_tokenize(sentence)   #Divides sentence into tokens in a list
print(tokens)

tags = nltk.pos_tag(tokens)
print(tags)

entities = nltk.chunk.ne_chunk(tags)    #Name entity recognition
print(entities)"""

import csv,re,emoji,spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
import sys,os
#sp=spacy.load('en_core_web_sm')

def extract_tweets():

    tweets=[]
    with open("C:\\Users\\hp\\Desktop\\inventory\\tweets.csv","r",encoding='UTF-8') as file:
        reader = csv.DictReader(file)

        for row in reader:

            if(row is not None):
                tweets.append(row['text'])
    return tweets


def get_tweet_sizes(tweets) :

    sizes=[]
    for i in range(0, len(tweets)):

        if (tweets[i]):
            sizes.append(len(tweets[i]))

    return sizes

def calculate_average(tweets,sizes):

    sum = 0

    for i in range(0, len(sizes)):
        sum += sizes[i]

    return (int(sum / len(sizes)))

def pre_process_tweets(tweets) :

    for i in range (0,len(tweets)) :

        if(tweets[i] is not None)  :

            tweets[i]=tweets[i].lower()            #To lower case
            tweets[i]=tweets[i].replace('@','')     # remove @
            tweets[i]=tweets[i].replace('#','')     # remove #
            tweets[i]=remove_urls(tweets[i])        # remove URL
            tweets[i]=remove_emojis(tweets[i])      # remove emojis
            tweets[i] = "".join(j for j in tweets[i] if j not in ("?", ".", ";", ":", "!","-",",","[","]","(",")","’","‘",'"',"$","'","“","”","•","=","+","%","/","&","|","~")) # remove punctuations

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

def get_tokens(positive_tweets) :

    temp_str=""

    for i in range(0, len(positive_tweets)):
        if(positive_tweets[i] is not None) :
            temp_str+=str(positive_tweets[i])

    tokens = word_tokenize(temp_str)

    return tokens

def remove_stopwords(tokens) :

    all_stopwords = set(stopwords.words('english')) #sp.Defaults.stop_words
    tokens_without_sw = [word for word in tokens if not word in all_stopwords]
    return tokens_without_sw

def get_POStags(tokens_without_sw) :

    return nltk.pos_tag(tokens_without_sw)

def remove_negative_tweets(tweets) :

    temp_str_array=[]

    for i in range(0,len(tweets)) :
        temp_str=TextBlob(str(tweets[i]))
        polarity=temp_str.sentiment.polarity
        if(polarity>=0):        # appending only positive or neutral sentiment tweets
            temp_str_array.append(temp_str)

    return temp_str_array


def write_preprocessed_tweets(positive_tweets) :

    f = open("tweets.txt", "w", encoding="utf-8")

    for i in range (0,len(positive_tweets)) :

        if(positive_tweets[i] is not None) :
            f.write(str(positive_tweets[i]))
            f.write("\n --------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

    f.close()

def write_keywords(str) :

    f = open("entities.txt", "w",encoding="utf-8")

    for i in range(0,len(str)) :
        f.write(str[i]+"\n")

    f.close()

def display_tweets(tweets) :

    for i in range (0,len(tweets)) :

        print(tweets[i])
        print("\n --------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

def generate_wordcloud(arg) :


    wc = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        height=600,
        width=400
    )

    str=""

    for i in range(0,len(arg)) :

        str+=arg[i]+" "

    wc.generate(str)
    wc.to_file('wordcloud.PNG')

def generate_histogram_processed(sizes_before_process,sizes_after_process) :

    plt.title("Number of words in tweets (before and after pre-processing)")
    plt.hist([sizes_before_process, sizes_after_process], bins=[0, 50, 100, 150, 200, 250, 300], rwidth=0.95,
             color=['dodgerblue', 'orchid'], label=['before pre-processing', 'after pre-processing'])
    plt.xlabel("Number of words in the tweet")
    plt.ylabel("Number of tweets")
    plt.legend()
    plt.show()

def generate_histogram_frequency(keywords) :

    counts = dict(Counter(keywords).most_common(23))

    labels, values = zip(*counts.items())

    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]

    indexes = np.arange(len(labels))

    bar_width = 0.05

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(15)

    plt.bar(indexes, values, color='orange')

    # add labels
    plt.title("Frequency of keywords in tweets")
    plt.xlabel("Keywords ->")
    plt.ylabel("Number of occurrences (Frequency) ->")
    plt.xticks(indexes + bar_width, labels)
    plt.show()

def remove_redundant(keywords):

    result=[]

    for keyword in keywords :

        keyword=keyword.lower()
        if(keyword =="calgary" or keyword =="charity" or keyword =="donation" or keyword =="alberta" or keyword =="donations"):
            keyword=""

    return keywords


def main() :

    tweets=extract_tweets()
    sizes_before_process = get_tweet_sizes(tweets)
    average_before=calculate_average(tweets,sizes_before_process)
    tweets=pre_process_tweets(tweets)
    positive_tweets = remove_negative_tweets(tweets)
    sizes_after_process = get_tweet_sizes(positive_tweets)
    average_after=calculate_average(positive_tweets,sizes_after_process)
    write_preprocessed_tweets(positive_tweets)
    num_tweets=len(tweets)
    tokens=get_tokens(positive_tweets)
    tokens_without_sw = remove_stopwords(tokens)
    tags=nltk.pos_tag(tokens_without_sw)
    chunks=nltk.ne_chunk(tags,binary=False)


    meaningful_keywords= [word for word,pos in tags if (pos=='NN')]
    meaningful_keywords=remove_redundant(meaningful_keywords)
    write_keywords(meaningful_keywords)

    print("\n\nStatistics of the analysis : \n\n")
    print("Number of tweets : " + str(num_tweets))
    print("Number of positive tweets : "+str(len(positive_tweets)))
    print("Average size of tweets (before processing) : "+ str(average_before) + " words")
    print("Average size of tweets (after processing) : "+str(average_after)+" words")
    print("Number of tokens : "+str(len(tokens)))
    print("Number of tokens without stopwords : "+str(len(tokens_without_sw)))
    print("Number of meaningful keywords : "+str(len(meaningful_keywords)))

    #generate_wordcloud(meaningful_keywords)
    #generate_histogram_processed(sizes_before_process,sizes_after_process)
    #generate_histogram_frequency(meaningful_keywords)


if __name__ == "__main__":
  main()
