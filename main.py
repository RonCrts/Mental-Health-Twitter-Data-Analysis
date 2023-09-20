import streamlit as st
import pandas as pd
from data import transform_data
from plotly import graph_objs as go


df = pd.read_csv('Mental-Health-Twitter-Analysis_merged.csv')
st.title('Mental Health Twitter Data Analysis')
st.image('https://images.everydayhealth.com/images/coping-with-depression-a-guide-to-good-treatment-1440x810.jpg')
st.sidebar.title('Mental Health Twitter Data Analysis')
st.sidebar.write('This application is a Streamlit dashboard used to analyze Mental Health Twitter Data.')
st.sidebar.write("The author of this application is <a href='https://www.linkedin.com/in/ronaldo-cort%C3%A9s-641a76227/'>Ronaldo Cortes Duran David.</a>", unsafe_allow_html=True)
st.sidebar.image("https://media.licdn.com/dms/image/D4E03AQGzSCd1Pz2AWA/profile-displayphoto-shrink_800_800/0/1667582997166?e=1700697600&v=beta&t=ycOaTMxmqLyIDzWszkSbsziXWjmsotL3w5hb_-eUwZI")
df = pd.read_csv('Mental-Health-Twitter-Analysis_merged.csv')
st.subheader('Context of the data')
st.info('The data is in uncleaned format and is collected using Twitter API. The Tweets has been filtered to keep only the English context. It targets mental health classification of the user at Tweet-level. i limited the data to 9000 cases to be more agile in the analysis with the resources at my disposal.')
st.write("<span style='color:red'>We approach the data from an analytical perspective that considers the relationship of the various variables that make it possible for us to consider a Tweet as depressive, such as the presence of bias in the statements.</span>", unsafe_allow_html=True)
st.subheader('Exploratory Analysis for Mental Health Twitter Data understanding')
st.write(df.head())
st.header('Average Number of Followers per User')
#plot the average number of followers per user(user_id, followers)
fig = go.Figure()
fig.add_trace(go.Histogram(x=df['followers']))
st.plotly_chart(fig)
st.header('Average Number of Favourites per User')
#plot the average number of favourites per user(user_id, favourites)
fig = go.Figure()
fig.add_trace(go.Histogram(x=df['favourites']))
st.plotly_chart(fig)
st.header('Average Number of Statuses per User')
#plot the average number of statuses per user(user_id, statuses)
fig = go.Figure()
fig.add_trace(go.Histogram(x=df['statuses']))
st.plotly_chart(fig)
#median table
st.header('Median of Followers, Favourites and Statuses per User')
st.write(df[['followers', 'favourites', 'statuses']].median())
st.title('Preparing the data for analysis')
st.info("First of all, we will apply an artificial intelligence model called Fine-tuned DepRoBERTa model, for detecting the level of depression as not depression, moderate or severe, based on social media posts in English.")
st.write("""Model was part of the winning solution for the Shared Task on Detecting Signs of Depression from Social Media Text at LT-EDI-ACL2022.
More information can be found in the following paper: OPI@LT-EDI-ACL2022: Detecting Signs of Depression from Social Media Text using RoBERTa Pre-trained Language Models.""")
st.subheader('Citiation')
st.write("""@inproceedings{poswiata-perelkiewicz-2022-opi,
    title = "{OPI}@{LT}-{EDI}-{ACL}2022: Detecting Signs of Depression from Social Media Text using {R}o{BERT}a Pre-trained Language Models",
    author = "Po{\'s}wiata, Rafa{\l} and Pere{\l}kiewicz, Micha{\l}",
    booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.ltedi-1.40",
    doi = "10.18653/v1/2022.ltedi-1.40",
    pages = "276--282",
}
""")
st.subheader('Depression Model from HuggingFace using inference API')
st.code('''
        import requests

class DepressionModel:
    def __init__(self):
        self.API_URL = "https://api-inference.huggingface.co/models/rafalposwiata/deproberta-large-depression"
        self.headers = {"Authorization": "Your API key"}

    def predict(self, text):
        payload = {"inputs": text}
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()

def predict_depression(df):
    import codecs
    model = DepressionModel()
    with codecs.open('depression_prediction.csv', 'a', encoding='utf-8') as f:
        for user_id in df['user_id'].unique():
            tweets = df[df['user_id'] == user_id]['post_text'].tolist()
            predictions = model.predict(tweets)
            for tweet, prediction in zip(tweets, predictions):
                f.write(f'{user_id};;{tweet};;{prediction}\n')
        ''')
st.write('The following code is used to predict depression using the depression model from HuggingFace using inference API')
st.subheader("""Model Description
             
This is one of the smaller BERT variants, pretrained model on English language using a masked language modeling objective.
             

Cognitive distortion refers to patterns of biased or distorted thinking that can lead to negative emotions, behaviors, and beliefs. These distortions are often automatic and unconscious, and can affect a person's perception of reality and their ability to make sound judgments.

Some common types of cognitive distortions include:

Personalization: Blaming oneself for things that are outside of one's control.
Examples:

She looked at me funny, she must be judging me.
I can't believe I made that mistake, I'm such a screw up.
Emotional Reasoning: Believing that feelings are facts, and letting emotions drive one's behavior.
Examples:

I feel like I'm not good enough, so I must be inadequate.
They never invite me out, so they must not like me.
Overgeneralizing: Drawing broad conclusions based on a single incident or piece of evidence.
Examples:

He never listens to me, he just talks over me.
Everyone always ignores my needs.
Labeling: Attaching negative or extreme labels to oneself or others based on specific behaviors or traits.
Examples:

I'm such a disappointment.
He's a total jerk.
Should Statements: Rigid, inflexible thinking that is based on unrealistic or unattainable expectations of oneself or others.
Examples:

I must never fail at anything.
They have to always put others' needs before their own.
Catastrophizing: Assuming the worst possible outcome in a situation and blowing it out of proportion.
Examples:

It's all going to be a waste of time, they're never going to succeed.
If I don't get the promotion, my entire career is over.
Reward Fallacy: Belief that one should be rewarded or recognized for every positive action or achievement.
Examples:

If I work hard enough, they will give me the pay raise I want.
If they don't appreciate my contributions, I'll start slacking off.""")
st.code('''
        import requests

class BiasDetector:
    def __init__(self):
        self.API_URL = "https://api-inference.huggingface.co/models/amedvedev/bert-tiny-cognitive-bias"
        self.headers = {"Authorization": "Your API key"}

    def predict(self, text):
        payload = {"inputs": text}
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()


def predict_cognitive_distortion(df):
    import codecs
    model = BiasDetector()
    with codecs.open('cognitive_distortion_prediction.csv', 'a', encoding='utf-8') as f:
        for user_id in df['user_id'].unique():
            tweets = df[df['user_id'] == user_id]['post_text'].tolist()
            predictions = model.predict(tweets)
            for tweet, prediction in zip(tweets, predictions):
                f.write(f'{user_id};;{tweet};;{prediction}\n')
        ''')
st.write('The following code is used to predict cognitive distortion using the cognitive distortion model from HuggingFace using inference API')
st.title('How we understand depression and cognitive distortion?')
st.subheader('Depression definition')
st.image('https://domf5oio6qrcr.cloudfront.net/medialibrary/7813/a83db567-4c93-4ad0-af6f-72b57af7675d.jpg')
st.info('Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. Also called major depressive disorder or clinical depression, it affects how you feel, think and behave and can lead to a variety of emotional and physical problems. You may have trouble doing normal day-to-day activities, and sometimes you may feel as if life isn\'t worth living.')
st.subheader('Cognitive Distortion definition')
st.image('https://anxietyandbehaviornj.com/wp-content/uploads/2018/02/Cognitive-Distortions.png')
st.info('Cognitive distortions are simply ways that our mind convinces us of something that isn\'t really true. These inaccurate thoughts are usually used to reinforce negative thinking or emotions — telling ourselves things that sound rational and accurate, but really only serve to keep us feeling bad about ourselves.')
st.title('Data preparation for analysis')
st.subheader('Merging the data')
st.code('''
def merge_data():
    depression_prediction = pd.read_csv('depression_prediction.csv', sep=';;', names=['user_id', 'post_text', 'prediction'], engine ='python')
    cognitive_distortion_prediction = pd.read_csv('cognitive_distortion_prediction.csv', sep=';;', names=['user_id', 'post_text', 'prediction'], engine ='python')
    depression_prediction['user_id'] = depression_prediction['user_id'].astype(str)
    cognitive_distortion_prediction['user_id'] = cognitive_distortion_prediction['user_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    df = pd.concat([df, depression_prediction.drop('post_text', axis=1), cognitive_distortion_prediction.drop('post_text', axis=1)], axis=1)
    with open('Mental-Health-Twitter-Analysis_merged.csv', 'w', encoding='utf-8') as f:
        df.to_csv(f, index=False)
''')
st.write('The following code is used to merge the data')
st.subheader('Transforming the data')
st.code('''
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
''')
st.title('Data Analysis and extraction of insights')
st.subheader('Depression and Cognitive Distortion over time')
st.info('The following graph shows the number of users classified as depressed and having cognitive distortion over time.')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['post_created'], y=df['severe'], name='Severe Depression'))
fig.add_trace(go.Scatter(x=df['post_created'], y=df['moderate'], name='Moderate Depression'))
st.plotly_chart(fig)
st.subheader('Severe Depression and Cognitive Distortion over time')
st.info('The following graph shows the number of users classified as severely depressed and having cognitive distortion over time. I limited the data to 500 cases to be more precise.')
severe = df['severe']
personalization = df['PERSONALIZATION']
emotiona_reasoning = df['EMOTIONALREASONING']
overgeneralizing = df['OVERGENERALIZING']
labeling = df['LABELING']
should_statements = df['SHOULDSTATEMENTS']
catastrophizing = df['CATASTROPHIZING']
reward_fallacy = df['REWARDFALLACY']
post_created = df['post_created']
df = df.head(500)
df['distortion'] = df[['PERSONALIZATION', 'EMOTIONALREASONING', 'OVERGENERALIZING', 'LABELING', 'SHOULDSTATEMENTS', 'CATASTROPHIZING', 'REWARDFALLACY']].idxmax(axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['post_created'], y=df['distortion'], name='Cognitive Distortion over time'))
st.plotly_chart(fig)
st.title('Non-depressed users with no cognitive distortion')
st.info('The following graph shows the number of users classified as non-depressed and having no cognitive distortion over time.')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['post_created'], y=df['notdepression'], name='Non-depressed'))
fig.add_trace(go.Scatter(x=df['post_created'], y=df['NODISTORTION'], name='No Cognitive Distortion'))
st.plotly_chart(fig)
st.title('Conclusion')
st.write('The data shows that there is a relationship between depression and cognitive distortion. The data shows that the number of users classified as severely depressed and having cognitive distortion is increasing over time.')
