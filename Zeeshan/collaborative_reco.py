from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

""" Item based collaborative filtering"""

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

def generate_recommendations(user,item_similarity_df) :

    similar_movies = pd.DataFrame()

    for movie, rating in user:
        similar_movies = similar_movies.append(get_similar_movies(movie, rating, item_similarity_df), ignore_index=True)

    similar_movies = similar_movies.sum().sort_values(ascending=False)

    return similar_movies

def print_reco(recommendations) :

    n=input("Enter the number of recommendation you want : ")
    print("\nRecommendations : \n")
    for i in range(0,int(n)):
        print(recommendations[i])

def main() :

    ratings=extract_database()
    users_ratings=ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
    users_ratings=users_ratings.dropna(thresh=10,axis=1).fillna(0)
    item_similarity_df = users_ratings.corr(method='pearson')
    user=[("Inception (2010)",5),("Captain America: The First Avenger (2011)",4),("Rise of the Planet of the Apes (2011)",1)]

    similar_movies=generate_recommendations(user,item_similarity_df)
    print_reco(similar_movies.index.values)


if __name__ == "__main__":

  main()