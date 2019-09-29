import requests
from bs4 import BeautifulSoup
import pandas as pd


headers = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
    
from datetime import date
today = date.today()
  
   
#Aftonbladet    
aftonbladet = "http://aftonbladet.se"
aftonbladet_pageTree = requests.get(aftonbladet, headers=headers)
aftonbladet_pageSoup = BeautifulSoup(aftonbladet_pageTree.content, 'html.parser')
af_hl = aftonbladet_pageSoup.find_all("h3", {"class": "_1Qq8L"})


af_headline_list = []
len_af_hl = len(af_hl)

for a in range(0,len_af_hl):
    af_headline_list.append(af_hl[a].text)

af_df = pd.DataFrame({"Headlines":af_headline_list})
af_df['Date'] = af_df.apply(lambda row: today, axis=1)
af_df['Publisher'] = af_df.apply(lambda row: 'Aftonbladet', axis=1)


#Dagens Nyheter
dn = "https://www.dn.se"
dn_pageTree = requests.get(dn, headers=headers)
dn_pageSoup = BeautifulSoup(dn_pageTree.content, 'html.parser')
dn_headlines = dn_pageSoup.find_all("h1", {"class": "teaser__title"})


len_dn_hl = len(dn_headlines)
dn_headline_list = []


for b in range(0,len_dn_hl):
    dn_headline_list.append(dn_headlines[b].text)
    
dn_df = pd.DataFrame({"Headlines":dn_headline_list})
dn_df['Date'] = dn_df.apply(lambda row: today, axis=1)
dn_df['Publisher'] = dn_df.apply(lambda row: 'Dagens Nyheter', axis=1)


#Expressen
ex = "https://www.expressen.se"
ex_pageTree = requests.get(dn, headers=headers)
ex_pageSoup = BeautifulSoup(ex_pageTree.content, 'html.parser')
ex_headlines = ex_pageSoup.find_all("h2")

len_ex_hl = len(ex_headlines)
ex_headline_list = []

for c in range(0,len_ex_hl):
    ex_headline_list.append(ex_headlines[c].text)

    
ex_df = pd.DataFrame({"Headlines":ex_headline_list})
ex_df['Date'] = ex_df.apply(lambda row: today, axis=1)
ex_df['Publisher'] = ex_df.apply(lambda row: 'Expressen', axis=1)

#Svenska dagbladet

svd = "https://www.svd.se"
svd_pageTree = requests.get(svd, headers=headers)
svd_pageSoup = BeautifulSoup(svd_pageTree.content, 'html.parser')
svd_headlines = svd_pageSoup.find_all("h2")


len_svd_hl = len(svd_headlines)
svd_headline_list = []

for d in range(0,len_svd_hl):
    svd_headline_list.append(svd_headlines[d].text)

svd_df = pd.DataFrame({"Headlines":svd_headline_list})
svd_df['Date'] = svd_df.apply(lambda row: today, axis=1)
svd_df['Publisher'] = svd_df.apply(lambda row: 'Svenska Dagbladet', axis=1)


#Metro
metro = "https://www.metro.se/nyheter"
metro_pageTree = requests.get(metro, headers=headers)
metro_pageSoup = BeautifulSoup(metro_pageTree.content, 'html.parser')
metro_headlines = metro_pageSoup.find_all("h2")


len_metro_hl = len(metro_headlines)
metro_headline_list = []

for d in range(1,len_metro_hl):
    metro_headline_list.append(metro_headlines[d].text)


metro_df = pd.DataFrame({"Headlines":metro_headline_list}).iloc[1:,:]
metro_df['Date'] = metro_df.apply(lambda row: today, axis=1)
metro_df['Publisher'] = metro_df.apply(lambda row: 'Metro', axis=1)



#Analysing the sentiment   

#Concat all the headlines
all_headlines = pd.concat([svd_df, ex_df, dn_df,af_df,metro_df]).reset_index().drop('index', axis=1)

#Translating the Headlines
from googletrans import Translator
lang = Translator()
eng_all_hl = []

for text in all_headlines.iloc[:,0]:
    eng_all_hl.append(lang.translate(text, dest='en').text)

df_eng_all_hl = pd.DataFrame({"Headlines":eng_all_hl, "Date":all_headlines.iloc[:,1], 'Publisher':all_headlines.iloc[:,2]})

#Overview of the Sentiment
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

all_hl_lists = svd_headline_list + ex_headline_list + dn_headline_list +af_headline_list + metro_headline_list
joint_headlines = ','.join(eng_all_hl)

my_stop_words = ENGLISH_STOP_WORDS.union(['sweden','swedish', 'new', 'best', 'want', 'does', 'dn'])

my_cloud = WordCloud(background_color='white', stopwords=my_stop_words).generate(joint_headlines)
plt.imshow(my_cloud, interpolation='bilinear') 
plt.axis("off")
plt.show()
'''
#Tokenizing
from nltk import word_tokenize
import nltk
nltk.download('punkt')
word_tokens = [word_tokenize(review) for review in df_eng_all_hl.headlines]
cleaned_tokens = [[word for word in item if word.isalpha()] for item in word_tokens]

list_cleaned_tokens = []
for i in cleaned_tokens:
    list_cleaned_tokens.append(','.join(i))
    
    
#Swedish Stopwords
import nltk
 
from nltk.corpus import stopwords
 
print (stopwords.words('swedish'))    
'''
#Vectorizing the headlines
   
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


vect = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, max_features=5000)

vect.fit(df_eng_all_hl.headlines)
vect_bow = vect.transform(df_eng_all_hl.headlines)

X_df = pd.DataFrame(vect_bow.toarray(), columns=vect.get_feature_names())


#Predicting the sentiment

#TextBlob Predict
from textblob import TextBlob

all_sentiment_polarity = []
all_sentiment_subjectivity = []
for i in eng_all_hl:
    all_sentiment_polarity.append(TextBlob(i).sentiment.polarity)
    all_sentiment_subjectivity.append(TextBlob(i).sentiment.subjectivity)

df_all_sentiment_polarity = pd.DataFrame(all_sentiment_polarity)  
df_all_sentiment_polarity.columns = ['polarity']
df_all_sentiment_subjectivity = pd.DataFrame(all_sentiment_subjectivity)    
df_all_sentiment_subjectivity.columns = ['subjectivity']

#Vader Predict
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

vader_sentiment = []
for i in eng_all_hl:
    vader_sentiment.append(sid.polarity_scores(i))

df_vader_sentiment = pd.DataFrame(vader_sentiment)
df_vader_sentiment.columns = ['neg', 'neu', 'pos', 'compound']
    
    
    
#Flair Predict


#Combining Predection Metrics
all_sentiment = pd.concat([df_all_sentiment_polarity, df_all_sentiment_subjectivity, df_vader_sentiment],axis=1)
headlines_and_sentiment = pd.concat([all_sentiment, all_headlines],axis=1)

#Analyzing the sentiment

#Viz the Metrics
import seaborn as sns
sns.set()

plt.subplot(2,3,1)
plt.hist(all_sentiment.polarity)
plt.xlabel('Polarity')
plt.subplot(2,3,2)
plt.hist(all_sentiment.subjectivity)
plt.xlabel('subjectivity')
plt.subplot(2,3,3)
plt.hist(all_sentiment.neg)
plt.xlabel('neg')
plt.subplot(2,3,4)
plt.hist(all_sentiment.neu)
plt.xlabel('neu')
plt.subplot(2,3,5)
plt.hist(all_sentiment.pos)
plt.xlabel('pos')
'''
plt.subplot(2,3,6)

plt.hist(all_sentiment.compound)
plt.xlabel('compound')
'''
plt.show()



all_sentiment.describe()

#Unsupervised Classification
'''
#Predicting the Sentiment
from sklearn.cluster import KMeans
xs = X_df.iloc[:250,:]
ys = X_df.iloc[250:,:]
kmeans = KMeans(n_clusters=2)
kmeans.fit(xs)
preds = kmeans.predict(ys)
'''
