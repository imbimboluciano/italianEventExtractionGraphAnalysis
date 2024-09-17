import pandas as pd
import pathlib



def filter_tweet_text(text):
    text = ' '.join(text.splitlines())
    return text

dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/sample_dataset/covid19_tweets.csv"
tweets = pd.read_csv(dataset_path, sep=',').head(100)

tweets = tweets[['date','text']]
tweets = tweets.sample(20)

final_dataset = pd.DataFrame(tweets)


dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/sample_dataset/fifa_world_cup_2022_tweets.csv"
tweets = pd.read_csv(dataset_path, sep=',').head(100)

tweets = tweets.sample(20)
tweets = tweets[['Date Created', 'Tweet']]
tweets.rename(columns={'Date Created': 'date','Tweet':'text'}, inplace=True)
final_dataset = pd.concat([final_dataset, tweets])

dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/sample_dataset/iranprotest_tweets.csv"
tweets = pd.read_csv(dataset_path, sep=',').head(100)

tweets = tweets[['date','text']]
tweets = tweets.sample(20)

final_dataset = pd.concat([final_dataset, tweets])

dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/sample_dataset/vaccination_tweets.csv"
tweets = pd.read_csv(dataset_path, sep=',').head(100)

tweets = tweets[['date','text']]
tweets = tweets.sample(20)

final_dataset = pd.concat([final_dataset, tweets])
final_dataset['text'] = final_dataset['text'].apply(filter_tweet_text)

dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/english_tweets_v2.csv"
final_dataset.to_csv(dataset_path,sep=';')
