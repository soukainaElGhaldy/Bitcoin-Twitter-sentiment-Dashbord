
# 1. Import librairies
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

st.title("The Bitcoin Twitter Sentiment")
# 2. Functions
def TweetsHandler(df):
    """ Handles the twitter data after import. """
    df.date = pd.to_datetime(df.date) #convert to datetime object
    df.set_index(['date'],inplace=True)# Set date column as index
    df['Nb_words']= df.clean_twt.apply(len)# Add column with nb of words in tweets
    df.rename(columns={"Sentiment2": "Sentiments"},inplace=True)
    return df

def CryptoHandler(df):
    """Handles the features data after import."""
    df['close time'] = pd.to_datetime(df['close time']) #convert to datetime object
    df.set_index(['close time'],inplace=True) #set index
    df = df.fillna(method="pad") #Fill Nan values forward
    df = df.shift(periods = -1) # lagging data to t-1
    df.drop(df.tail(1).index,inplace=True) #delete the last row
    return df

def WingMan(df1,df2):
    """ Joins 2 dataframes."""
    return df1.join(df2)

def get_dom(date):
    """ Returns the day of the month."""
    return date.day

def get_weekdayName(date):
    """Returns the name of the day of the week."""
    return date.strftime('%A')

def get_hour(date):
    """ Returns the hour of the day."""
    return date.hour

def MinMaxNormalize(series):
    """ Returns the serie with a min-max normalization."""
    return (series - min(series))/(max(series) - min(series))


def BagOfWords(series):
    """This function's input is a series of strings and it merges all the strings in one big string."""
    return series.str.cat(sep=' ')

def price(column,columnName):
    """Returns a tuple."""
    return (columnName,str(column.values[-1])+"$",str(round((column.values[-1]- column.values.mean())/column.values[-1],2)*100)+"%")

# 3. Code
## Import csv files

tweets = pd.read_csv("./data/DATA.csv", lineterminator='\n')
features = pd.read_csv('./data/FEATURES.csv')
## Total number of collected tweets (extracted from previous study)
Total = 3861148
# Apply Functions
twt = TweetsHandler(tweets)
crypto = CryptoHandler(features)
data = WingMan(twt,crypto)
data['Day'] = data.index.map(get_dom)
data['Hour'] = data.index.map(get_hour)
data['Day_Name'] = data.index.map(get_weekdayName)

st.header("Discovering the data")
## Show the final Dataframe
st.dataframe(data.head(3))
## Show important stats
col1, col2, col3 = st.columns(3)
col1.metric("Collected Tweets", Total, "+1%")
col2.metric("Number of Words", str(data.Nb_words.sum()), "-1%")
col3.metric("Number of Rows", len(data), "-1%")

col4, col5 = st.columns(2)

col4.metric(price(crypto.BTC,"BTC price")[0],price(crypto.BTC,"BTC price")[1],price(crypto.BTC,"BTC price")[2])
col5.metric(price(crypto.ETH,"ETH price")[0],price(crypto.ETH,"ETH price")[1],price(crypto.ETH,"ETH price")[2])

col6, col7 = st.columns(2)

col6.metric(price(crypto.SOL,"SOL price")[0],price(crypto.SOL,"SOL price")[1],price(crypto.SOL,"SOL price")[2])
col7.metric(price(crypto.ADA,"ADA price")[0],price(crypto.ADA,"ADA price")[1],price(crypto.ADA,"ADA price")[2])



## First Chart
#Code Credits :  https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners
trace1 = go.Scatter(
                    x = data.index,
                    y = MinMaxNormalize(data.Nb_words),
                    mode = "lines",
                    name = "Number of tweeted words Normalized",
                    marker = dict(color = 'rgba(255,69,0, 0.8)'),
                    text= data.Sentiment1)
trace2 = go.Scatter(
                    x = data.index,
                    y = MinMaxNormalize(data.BTC),
                    mode = "lines",
                    name = "BTC lagged t-1 Normalized",
                    marker = dict(color = 'rgba(152,251,152, 0.8)'),
                    text= data.Sentiment1)
traces = [trace1, trace2]
layout = dict(title = 'Number of words Vs Lagged BTC price',
              xaxis= dict(title= 'time',ticklen= 5,zeroline= False)
             )
fig = dict(data = traces, layout = layout)
st.plotly_chart(fig)

st.caption('We can see here how Twitter sentiment has an impact on the bitcoin price. In order, to improve accuracy the student has created a data dictionnary with famous crypto community slangs and added it to the VADER Lexicon.')
## Second Chart
#Code Credits :  https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners
trace3 = go.Scatter(
                    x = data.index,
                    y = data.Comp_diff,
                    mode = "markers",
                    name = "Difference between Coumpound Scores of tweets",
                    marker = dict(size=2,color = 'rgb(65,105,225, 0.5)',symbol='square-open'),
                    text= data.Sentiments)

layout = dict(title = 'Effect of the added Data Dictionary on sentiment Detection',
              xaxis= dict(title= 'time',ticklen= 5,zeroline= False))
fig = dict(data = trace3, layout = layout)
st.plotly_chart(fig)

st.caption('Here, we can observe the difference between the scores acquired with the help of the data dictionnary and the scores with only the native VADER Lexicon.')

# Third Chart
# create trace 4 that is 3d scatter
st.subheader("Sentiment Scores for 1 minute in Twitter.")
trace4 = go.Scatter3d(
    x=data.Positive2,
    y=data.Negative2,
    z=data.Neutral2,
    name='Tweets',
    mode='markers',
    marker=dict(
        size=1,
        color='rgb(85,107,47)'))# set color to an array/list of desired values


layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0),

    scene = dict(
    xaxis = dict(title='Positive Score',
                backgroundcolor="black",
                gridcolor="white",
                showbackground=True),
    yaxis = dict(title='Negative Score',
                 backgroundcolor="black",
                 gridcolor="white",
                 showbackground=True),
    zaxis = dict(title='Neutral Score',
                 backgroundcolor="black",
                 gridcolor="white",
                 showbackground=True)),


)
fig = go.Figure(data=trace4, layout=layout)
st.plotly_chart(fig)

st.caption("Each data point represents the sentiment of 1 min gap of tweets.")

#Fourth chart
fig = px.histogram(data, x=data.Day_Name, color="Sentiments",
                   facet_col="Sentiments",
                   color_discrete_sequence=px.colors.qualitative.G10,
                   title="Tweeting frequency by Day of the week and Sentiment")
fig.update_layout(
    font_size = 15,
    title_font_size = 20,
    yaxis_title="count")

st.plotly_chart(fig)

#Fifth chart
fig = px.histogram(data, x=data.Hour, color="Sentiments",
                   facet_col="Sentiments",
                   color_discrete_sequence=px.colors.qualitative.G10,
                   title="Tweeting frequency by Hour of the day and Sentiment")
fig.update_layout(
    font_size = 15,
    title_font_size = 20,
    yaxis_title="count")

st.plotly_chart(fig)

#Sixth chart
fig = px.histogram(data, x=data.Day, color="Sentiments",
                   facet_col="Sentiments",
                   color_discrete_sequence=px.colors.qualitative.G10,
                   title="Tweeting frequency by Day of the month and Sentiment",
                   template = 'plotly_dark')
fig.update_layout(
    font_size = 15,
    title_font_size = 20,
    yaxis_title="count")
st.plotly_chart(fig)


st.caption("The twitter data is extremely noisy. And eventhough we cleaned it there is still the obvious fact that only people who are interested in the bitcoin will post about it. Which explains the extremely positive vibe in twitter about the bitcoin.")

#Seventh chart
myBigBOW = BagOfWords(data.clean_twt)
my_stopwords = set(STOPWORDS)
my_stopwords = my_stopwords.union(ENGLISH_STOP_WORDS)
list_stopwords = ["triangle","pointed","car","light","amp","usd","police","car","bitcoin", "Bitcoin",
                  "el","salvador","s","president","btc","tender","week","eth","day"," ","fet"
                  "project","btc","usd","etf","floor","index","ethereum", "mining","skin","tone",
                  "tweet","code","tx","backhand","pointing","face","people","think","today","check","mark",
                  "point","projector","addr","orders","guy","use","thing","triangular","year","day","eth",
                  "going","attraction","make","week","rolling","come","button","time","say","cryptocurrencie"
                  ,"look","thats", "said","happen","world","miner","follow","transaction","hash","raising",
                  "hand","United","State","States","attractions","s","legal","tender",
                  "ll", "nft","month","lets","government","country","vault","location","team","goe","user"]
my_stopwords = my_stopwords.union(list_stopwords)
st.subheader('The Most Frequent words Posted with the "#bitcoin"')
my_cloud = WordCloud(mode = "RGBA", background_color=None,stopwords=my_stopwords).generate(myBigBOW)
plt.imshow(my_cloud, interpolation='bilinear')
plt.axis('off')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
