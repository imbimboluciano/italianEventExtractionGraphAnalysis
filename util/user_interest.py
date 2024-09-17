import pandas as pd
import pathlib
import random

def create_previuos_timestap_values(value):
    return value - random.randint(0,value)


#dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/keyword_english_tweets.csv"
dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "graphbasedeventdetection/dataset/keyword_italian_tweets.csv"
dataset = pd.read_csv(dataset_path, sep=';')
likes = dataset['favorite_count']
retweets = dataset['retweet_count']

likes_previuos = likes.apply(create_previuos_timestap_values)
retweets_previuos = retweets.apply(create_previuos_timestap_values)

likes_previuos.name = 'favorite_count_previous'
retweets_previuos.name = 'retweet_count_previous'

dataset = pd.concat([dataset, likes_previuos], axis=1)
dataset = pd.concat([dataset, retweets_previuos], axis=1)
dataset.to_csv('graphbasedeventdetection/dataset/keyword_italian_tweets.csv',sep=';')

