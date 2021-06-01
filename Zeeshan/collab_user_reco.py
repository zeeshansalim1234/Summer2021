from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

""" User based collaborative filtering"""

def extract_database():

    ratings = pd.read_csv("ratings.csv")
    ratings = ratings.fillna(0)
    movies = pd.read_csv("movies.csv")
    ratings = pd.merge(movies, ratings).drop(['genres','timestamp'],axis=1)
    return ratings


def standardize(row):

    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row

def get_similar_movies(movie_name,user_rating,item_similarity_df) :

    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

def generate_recommendations(user_similarity_df,sample_user) :

    user_similarity = user_similarity_df.values

    results = []
    indices = []

    for i in range(0, len(user_similarity)):
        results.append(user_similarity[sample_user-1][i])

    for i in range(0, len(user_similarity)):
        indices.append(i+1)

    users_tuple = list(zip(indices,results))
    users_tuple=Sort_Tuple(users_tuple)
    users_ids = [a_tuple[0] for a_tuple in users_tuple]
    users_ids_reversed=users_ids[::-1]

    return users_ids_reversed

def Sort_Tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):

        for j in range(0, lst - i - 1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup



def print_reco(recommendations,sample_user) :

    n=input("Enter the number of potential friends to suggest : ")
    print("\nRecommendations : \n")
    for i in range(0,(int(n)+1)):
        if(recommendations[i]!=sample_user):
            print(recommendations[i])

def main() :

    ratings=extract_database()
    users_ratings=ratings.pivot_table(index=['title'],columns=['userId'],values='rating')
    users_ratings=users_ratings.dropna(thresh=10,axis=1).fillna(0)
    user_similarity_df = users_ratings.corr(method='pearson')

    sample_user = int(input("Enter User's id : "))
    reco=generate_recommendations(user_similarity_df,sample_user)
    print_reco(reco,sample_user)


if __name__ == "__main__":

  main()