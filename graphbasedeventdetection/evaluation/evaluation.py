import pandas as pd
import pathlib
import random

dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/keyword_italian_tweets.csv"
dataset = pd.read_csv(dataset_path, sep=';')
real_news = dataset['real_news']


print(real_news.value_counts().size)

