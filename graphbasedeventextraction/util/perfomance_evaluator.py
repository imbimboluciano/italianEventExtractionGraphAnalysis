import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def string_to_list(s):
    if pd.isna(s) or s == '':
        return []
    try:
        return ast.literal_eval(s)
    except:
        return s.split(', ')


class PerformanceEvaluator:

    def __init__(self,language,dataset_path):
        self.language = language
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(self.dataset_path, sep=';')
        self.cleaned_tweets = self.dataset["cleaned_tweets"]
        self.len_events_classified = 0

    def convert_csv_to_original_form(self,csv_file):
        df = pd.read_csv(csv_file,on_bad_lines='skip')
    
        for col in ['what', 'who', 'where']:
            df[col] = df[col].apply(string_to_list)
    
        df['when'] = pd.to_datetime(df['when'])
        result = df.to_dict('records')
    
        return result

    def get_all_events(self, event_dataset_path):
        return self.convert_csv_to_original_form(event_dataset_path)
    

    def sentence_similarity(self,sentence1, sentence2):
    # Create a vocabulary from both sentences
        vectorizer = CountVectorizer().fit([sentence1, sentence2])
    
    # Create vectors
        vector1 = vectorizer.transform([sentence1]).toarray()
        vector2 = vectorizer.transform([sentence2]).toarray()
    
    # Calculate cosine similarity
        similarity = cosine_similarity(vector1, vector2)[0][0]
    
        return similarity

    def check_classification(self,indices):
        filter_df = self.dataset.loc[indices]
        news = filter_df['news']
        if len(news.value_counts()) == 1:
            is_correct = True
        else:
            is_correct = False
        return is_correct

    def accuracy(self,all_events):
        self.len_events_classified = len(all_events)
        events_correctly_classified = []

        similarity_threshold = 0.2

        for event in all_events:
            words = []
            words.extend(event['what'])
            words.extend(event['who'])
            words.extend(event['where'])

            event_similarities = []
            for idx,tweet in enumerate(self.cleaned_tweets):
                similarity = self.sentence_similarity(tweet,' '.join(words))
                if similarity >= similarity_threshold:
                    event_similarities.append({
                        'tweet':idx,
                        'similarity': similarity})
                        
            tweet_keys = []
            for event_similarity in event_similarities:
                tweet_keys.append(event_similarity['tweet'])
            is_correct = self.check_classification(tweet_keys)
            print(f'{tweet_keys} and {is_correct}')
            if is_correct:
                events_correctly_classified.append(event)


        return len(events_correctly_classified) / self.len_events_classified
    


    def get_n_events_classified(self):
        return self.len_events_classified
    

    def get_n_events_groundtruth(self):
        return len(self.dataset['news'].value_counts())
    

    def check_named_entity(self,all_events):
        n_what = 0
        n_who = 0
        n_where = 0
        n_when = 0

        for event in all_events:
            if event["what"]:
                n_what += 1
            
            if event["where"]:
                n_where += 1
            
            if event["who"]:
                n_who += 1

            if event["when"]:
                n_when += 1

        return n_what/self.len_events_classified, n_who / self.len_events_classified, n_where / self.len_events_classified, n_when / self.len_events_classified