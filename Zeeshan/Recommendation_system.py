from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def get_title_from_index(index,df):

    return df[df.index == index]["title"].values[0]

def get_index_from_title(title,df):

    return df[df.title == title]["index"].values[0]

def extract_database():

    df = pd.read_csv("C:\\Users\\hp\\Downloads\\archive\\IMDb movies.csv")
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


def generate_movie_data(description,actors,genre,director):

    result=[]
    for i in range(0,len(description)):
        result.append(description[i]+" "+actors[i]+" "+genre[i]+" "+director[i])

    return result


def main():

    features = ['description', 'actors', 'genre', 'director']
    df=extract_database()
    description = df['description'].to_list()
    actors=df['actors'].to_list()
    genre=df['genre'].to_list()
    director=df['director'].to_list()
    description=pre_process(description)
    actors=pre_process(actors)
    genre=pre_process(genre)
    director=pre_process(director)
    movie_data=generate_movie_data(description,actors,genre,director)

    cv=CountVectorizer()
    count_matrix=cv.fit_transform(movie_data)

    cos_sim=cosine_similarity(count_matrix)
    sample_movie="Avatar"

    movie_index=get_index_from_title(sample_movie)
    similar_movies=list(enumerate(cos_sim[movie_index]))
    sorted_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

    i = 0
    for element in sorted_movies:
        print(get_title_from_index(element[0]))
        i = i + 1
        if i > 50:
            break


if __name__ == "__main__":

  main()


