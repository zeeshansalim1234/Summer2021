from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def get_title_from_index(index,df):

    return df[df.index == index]["OrganizationName"].values[0]

def get_index_from_title(title,df):

    return df[df.OrganizationName == title]["index"].values[0]

def extract_database():

    df = pd.read_csv("ccc-organizations-2011_1.csv")
    return df

def pre_process(tweets):

    for i in range(0, len(tweets)):

        if (tweets[i] is not None):

            if(tweets[i]!=tweets[i]):
                tweets[i]=""

            tweets[i] = tweets[i].lower()  # To lower case
            tweets[i] = tweets[i].replace('@','')  # remove @
            tweets[i] = tweets[i].replace('#','')  # remove #
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


def generate_movie_data(description,actors):

    result=[]
    for i in range(0,len(description)):
        result.append(description[i]+" "+actors[i])

    return result

def print_similar(sorted_movies,movie_index,df):

    print("\nRecommendations : \n")

    i = 0
    for element in sorted_movies:

        if (element[0] != movie_index):
            print(get_title_from_index(element[0], df))
        i = i + 1
        if i > 50:
            break


def main():

    features = ['Description', 'City']
    df=extract_database()
    df['index']=df.index
    description = df['Description'].to_list()
    city=df['City'].to_list()
    description=pre_process(description)
    city=pre_process(city)
    charity_data=generate_movie_data(description,city)

    cv=CountVectorizer()
    count_matrix=cv.fit_transform(charity_data)


    cos_sim=cosine_similarity(count_matrix)
    sample_movie=input("Enter the charity user had previously donated to: ")

    chairty_index=get_index_from_title(sample_movie,df)
    similar_charities=list(enumerate(cos_sim[chairty_index]))
    sorted_charities=sorted(similar_charities,key=lambda x:x[1],reverse=True)

    print_similar(sorted_charities,chairty_index,df)

if __name__ == "__main__":

  main()


