import pandas as pd
import re
import nltk
import yake
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def remove_special_chars_and_urls(text):

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[@#$%^&*()_+=\[\]{}|\\<>/~`]', '', text)
    text = re.sub(r'[^\w\s.,\'"!?]', '', text)
    return text.strip()

def lemmatize_keywords(text):

    lemmatizer = WordNetLemmatizer()
    words = text.split(',')
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ','.join(lemmatized_words)

def remove_stop_words(text):

    stop_words = set(stopwords.words('italian'))
    #stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return ' '.join(filtered_sentence)

def extract_keyword(text):

    kw_extractor = yake.KeywordExtractor()
    language = "it"
    max_ngram_size = 1
    deduplication_threshold = 0.9
    numOfKeywords = 10
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = kw_extractor.extract_keywords(text)
    keywords = [keyword[0] for keyword in keywords]
    return ','.join(keywords)

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

    dataset_path = "dataset/italian_tweets.csv"
    #dataset_path = "dataset/english_tweets.csv"
    dataset = pd.read_csv(dataset_path, sep=';')
    cleaned_tweets = cleaning_data(dataset)

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

    keyword_tweets = keyword_extraction(cleaned_tweets)
    keyword_tweets = keyword_tweets.apply(lemmatize_keywords)
    keyword_tweets.name = 'keyword'
    keyword_dataset = pd.concat([dataset, keyword_tweets], axis=1)
    keyword_dataset.to_csv('dataset/keyword_italian_tweets.csv',sep=";")