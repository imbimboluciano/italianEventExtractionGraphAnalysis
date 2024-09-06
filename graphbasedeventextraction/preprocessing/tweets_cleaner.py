import pandas as pd
import re, pathlib, string
from wordsegment import load, segment

# Load wordsegment
load()

def clean_tweet(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')

    tweet = re.sub(r'@\w+', '', tweet)
    
    emoticon_pattern = r'[:;=][oO\-]?[D\)\]\(\]/\\OpP]'
    tweet = re.sub(emoticon_pattern, '', tweet)
    
    tweet = re.sub(r'#(\w+)', lambda m: ' '.join(segment(m.group(1))), tweet)

    return tweet

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def trim_spaces_in_middle(text):
    trimmed_text = ' '.join(text.split())
    return trimmed_text



# dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/italian_tweets.csv"
dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/english_tweets.csv"


df = pd.read_csv(dataset_path, sep=';')
df['cleaned_tweets'] = df['full_text'].apply(clean_tweet)
df['cleaned_tweets'] = df['cleaned_tweets'].apply(remove_punctuation)
df['cleaned_tweets'] = df['cleaned_tweets'].apply(trim_spaces_in_middle)

dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/cleaned_english_tweets.csv"
df.drop(['favorite_count','view_count','retweet_count','reply_count', 'full_text'], axis=1, inplace=True)
df.to_csv(dataset_path, sep=';')
