from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

""" Item based collaborative filtering"""

def extract_database() :

    ratings = pd.read_csv(
        "https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Collaborative%20Filtering/dataset/toy_dataset.csv",
        index_col=0)
    ratings = ratings.fillna(0)
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

def main() :

    ratings=extract_database()
    ratings_std = ratings.apply(standardize)
    item_similarity=cosine_similarity(ratings_std.T)
    item_similarity_df = pd.DataFrame(item_similarity,index=ratings.columns,columns=ratings.columns) #shows relationship bertween items
    user=[("action1",5),("romantic1",1),("romantic2",1)]

    similar_movies=generate_recommendations(user,item_similarity_df)
    print(similar_movies)


if __name__ == "__main__":

  main()