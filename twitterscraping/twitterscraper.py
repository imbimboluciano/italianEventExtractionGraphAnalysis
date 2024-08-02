from twikit import Client
import pandas as pd
import asyncio
import re
import configparser
import pathlib


config_path = pathlib.Path(__file__).parent.absolute() / "config.ini"
config = configparser.ConfigParser()
config.read(config_path)

USERNAME = config['TwitterCredentials']['USERNAME']
EMAIL = config['TwitterCredentials']['EMAIL']
PASSWORD = config['TwitterCredentials']['PASSWORD']

# Initialize client
client = Client('it-IT')

users = ["Agenzia_ansa", "repubblica", "Corriere", "fattoquotidiano", "libero_it", "LaVeritaWeb", "ilfoglio_it", "LaStampa", "qn_carlino","ilmessaggeroit","sole24ore"]

def filter_tweet_text(text):
    text = ' '.join(text.splitlines())
    return text

async def retrieve_tweets():
    await client.login(auth_info_1=USERNAME, auth_info_2=EMAIL, password=PASSWORD)
    client.save_cookies('cookies.json')
    client.load_cookies(path='cookies.json')

    tweets_to_store = []

    for user in users:

        user_retrieved = await client.get_user_by_screen_name(user)
        tweets = await client.get_user_tweets(user_retrieved.id, count=40, tweet_type='Tweets') 
        for i in range(2):
            
            if(i != 0):
                tweets = await tweets.next()

            for tweet in tweets:
                text = filter_tweet_text(tweet.text)

                tweets_to_store.append({
                    'author': user,
                    'created_at': tweet.created_at,
                    'favorite_count': tweet.favorite_count,
                    'view_count': tweet.view_count,
                    'full_text': text,
                })

            
    dataset_old = pd.read_csv('../dataset/tweet.csv')
    df = pd.DataFrame(tweets_to_store)
    new_dataset = pd.concat([dataset_old,df])
    new_dataset.to_csv('tweets.csv', index=False, sep=';')
    

asyncio.run(retrieve_tweets())




