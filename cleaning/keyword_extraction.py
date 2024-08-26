import pandas as pd
import pathlib
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from string import punctuation

def remove_special_chars_and_urls(text):

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[@#$%^&*()_+=\[\]{}|\\<>/~`]', '', text)
    text = re.sub(r'[^\w\s.,\'"!?]', '', text)
    return text.strip()

def remove_stop_words(text):

    stop_words = set(stopwords.words('italian'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return ' '.join(filtered_sentence)

def extract_keyword(text):

    pos_tag = ['ADJ', 'NOUN']
    nlp = spacy.load("it_core_news_sm")
    doc = nlp(text.lower())
    keywords = [w.text for w in doc if w.pos_ in pos_tag]
    return keywords

def cleaning_data(df):

    df.drop_duplicates()
    tweets = df['full_text']
    tweets = tweets.apply(remove_special_chars_and_urls)

    return tweets

def keyword_extraction(tweets):


    tweets = tweets.apply(remove_stop_words)
    tweets = tweets.apply(extract_keyword)

    return tweets

if __name__ == "__main__":

    dataset_path = "dataset/right_tweets.csv"
    dataset = pd.read_csv(dataset_path, sep=';')
    cleaned_tweets = cleaning_data(dataset)

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

    keyword_tweets = keyword_extraction(cleaned_tweets)
    keyword_dataset = pd.concat([dataset, keyword_tweets], axis=1)
    keyword_dataset.to_csv('dataset/keyword_tweets.csv')