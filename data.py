import pandas as pd
from depression_model import DepressionModel 
from bias_detection import BiasDetector


def predict_depression(df):
    import codecs
    model = DepressionModel()
    with codecs.open('depression_prediction.csv', 'a', encoding='utf-8') as f:
        for user_id in df['user_id'].unique():
            tweets = df[df['user_id'] == user_id]['post_text'].tolist()
            predictions = model.predict(tweets)
            for tweet, prediction in zip(tweets, predictions):
                f.write(f'{user_id};;{tweet};;{prediction}\n')

def predict_cognitive_distortion(df):
    import codecs
    model = BiasDetector()
    with codecs.open('cognitive_distortion_prediction.csv', 'a', encoding='utf-8') as f:
        for user_id in df['user_id'].unique():
            tweets = df[df['user_id'] == user_id]['post_text'].tolist()
            predictions = model.predict(tweets)
            for tweet, prediction in zip(tweets, predictions):
                f.write(f'{user_id};;{tweet};;{prediction}\n')

def merge_data():
    depression_prediction = pd.read_csv('depression_prediction.csv', sep=';;', names=['user_id', 'post_text', 'prediction'], engine ='python')
    cognitive_distortion_prediction = pd.read_csv('cognitive_distortion_prediction.csv', sep=';;', names=['user_id', 'post_text', 'prediction'], engine ='python')
    depression_prediction['user_id'] = depression_prediction['user_id'].astype(str)
    cognitive_distortion_prediction['user_id'] = cognitive_distortion_prediction['user_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    df = pd.concat([df, depression_prediction.drop('post_text', axis=1), cognitive_distortion_prediction.drop('post_text', axis=1)], axis=1)
    with open('Mental-Health-Twitter-Analysis_merged.csv', 'w', encoding='utf-8') as f:
        df.to_csv(f, index=False)

def transform_data():
    df = pd.read_csv('Mental-Health-Twitter-Analysis_merged.csv')
    df['prediction'] = df['prediction'].astype(str) # convert to string type
    df['prediction'] = df['prediction'].apply(lambda x: x[1:-1]) # remove brackets
    df['prediction'] = df['prediction'].apply(lambda x: ''.join(x)) # join list of strings into a single string
    df['prediction'] = df['prediction'].apply(lambda x: x.split(',')) # split the string into a list
    df['prediction'] = df['prediction'].apply(lambda x: [i.split(':') for i in x if len(i.split(':'))==2]) # split each element of the list into a list of two elements
    df['prediction'] = df['prediction'].apply(lambda x: [i[1] for i in x]) # get the second element of each list
    df['prediction'] = df['prediction'].apply(lambda x: [i.replace("'", "") for i in x]) # remove single quotes
    df['prediction'] = df['prediction'].apply(lambda x: [i.replace(" ", "") for i in x]) # remove spaces
    df['prediction'] = df['prediction'].apply(lambda x: [i.replace("}", "") for i in x]) # remove curly brackets
    df['prediction'] = df['prediction'].apply(lambda x: [i.replace("]", "") for i in x]) # remove square brackets
    df['prediction'] = df['prediction'].apply(lambda x: [i.replace('"', "") for i in x]) # remove double quotes
    df['prediction'] = df['prediction'].apply(lambda x: [i.replace('label', "") for i in x]) # remove label
    df['prediction'] = df['prediction'].apply(lambda x: dict(zip(x[::2], x[1::2])))
    df['prediction.1'] = df['prediction.1'].astype(str) # convert to string type
    df['prediction.1'] = df['prediction.1'].apply(lambda x: x[1:-1]) # remove brackets
    df['prediction.1'] = df['prediction.1'].apply(lambda x: ''.join(x)) # join list of strings into a single string
    df['prediction.1'] = df['prediction.1'].apply(lambda x: x.split(',')) # split the string into a list
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i.split(':') for i in x if len(i.split(':'))==2]) # split each element of the list into a list of two elements
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i[1] for i in x]) # get the second element of each list
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i.replace("'", "") for i in x]) # remove single quotes
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i.replace(" ", "") for i in x]) # remove spaces
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i.replace("}", "") for i in x]) # remove curly brackets
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i.replace("]", "") for i in x]) # remove square brackets
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i.replace('"', "") for i in x]) # remove double quotes
    df['prediction.1'] = df['prediction.1'].apply(lambda x: [i.replace('label', "") for i in x]) # remove label
    df['prediction.1'] = df['prediction.1'].apply(lambda x: dict(zip(x[::2], x[1::2])))
    df.drop(['user_id.1', 'user_id.2'], axis=1, inplace=True)
    df = pd.concat([df.drop(['prediction'], axis=1), df['prediction'].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop(['prediction.1'], axis=1), df['prediction.1'].apply(pd.Series, dtype='object')], axis=1)
    return df

