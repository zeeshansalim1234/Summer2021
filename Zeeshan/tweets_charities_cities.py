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


def generate_charity_data(result,description,organization,city):


    for i in range(0,len(description)):
        result.append(organization[i]+" "+description[i]+" "+city[i])

    return result

def print_similar(sorted_charities,charity_index,df):

    print("\nRecommendations : \n")

    i = 0
    for element in sorted_charities:

        if (element[0] != charity_index):
            print(get_title_from_index(element[0], df))

        i = i + 1
        if i > 5 :
            break


def main():

    features = ['Description', 'OrganizationName','City']
    df=extract_database()
    df['index']=df.index
    description = df['Description'].to_list()
    organization_name = df['OrganizationName'].to_list()
    city=df['City'].to_list()
    description=pre_process(description)
    organization_name=pre_process(organization_name)
    city=pre_process(city)

    sample_description = input("Enter a sentence : ")
    user_location = input("Enter your city of residence : ")

    charity_data=[]
    charity_data=generate_charity_data(charity_data,description,city,organization_name)
    charity_data.append(sample_description + " " + user_location)
    charity_data = pre_process(charity_data)

    cv=CountVectorizer()
    count_matrix=cv.fit_transform(charity_data)

    cos_sim=cosine_similarity(count_matrix)

    similar_charities=list(enumerate(cos_sim[len(charity_data)-1]))
    sorted_charities=sorted(similar_charities,key=lambda x:x[1],reverse=True)

    print_similar(sorted_charities,len(charity_data)-1,df)

    #During covid-19 pandemic children are more prone to child abuse at home

if __name__ == "__main__":

  main()


