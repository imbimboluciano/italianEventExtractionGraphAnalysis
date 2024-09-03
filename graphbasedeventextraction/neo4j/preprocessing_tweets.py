import spacy
import re

nlp = spacy.load("en_core_web_sm")

def preprocess_tweet(tweet):
    # Remove URLs, mentions, non-ASCII characters, etc.
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    return tweet

def extract_entities_and_context(tweet):
    doc = nlp(tweet)
    entities = [ent.text for ent in doc.ents]
    context = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return {'entities': entities, 'context': context}

# Example
tweets = ["@user1 Check out the latest news on the hurricane in Florida http://example.com"]
processed_tweets = [extract_entities_and_context(preprocess_tweet(tweet)) for tweet in tweets]
