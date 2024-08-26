from twikit import Client
import pandas as pd
import asyncio
import configparser
import pathlib


USERNAME = ''
EMAIL = ''
PASSWORD = ''

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
                    'retweet_count': tweet.retweet_count,
                    'reply_count': tweet.reply_count,
                    'full_text': text,
                })

    dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/tweets.csv"
    try:
        dataset_old = pd.read_csv(dataset_path, sep=';')
    except pd.errors.EmptyDataError:
        print('CSV file is empty')
        dataset_old = pd.DataFrame()
    except FileNotFoundError:
        print('CSV file not found')

    
    df = pd.DataFrame(tweets_to_store)
    new_dataset = pd.concat([dataset_old,df])
    new_dataset.to_csv(dataset_path, index=False, sep=';')
    

asyncio.run(retrieve_tweets())




