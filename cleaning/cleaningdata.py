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

    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    nlp = spacy.load("it_core_news_lg")
    doc = nlp(text.lower())
    keywords = [w.text for w in doc if w.pos_ in pos_tag]
    print(keywords)
    return keywords


def cleaning_data(df):

    df.drop_duplicates()
    tweets = df['full_text']
    tweets = tweets.apply(remove_special_chars_and_urls)

    return tweets

def keyword_extraction(tweets):
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    

    tweets = tweets.apply(remove_stop_words)
    tweets = tweets.apply(extract_keyword)

    return tweets
    


if __name__ == "__main__":

    dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/tweets.csv"
    dataset = pd.read_csv(dataset_path, sep=';')
    cleaned_tweets = cleaning_data(dataset.head(100))

    keyword_tweets = keyword_extraction(cleaned_tweets)
    print(keyword_tweets)
   