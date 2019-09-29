# Sentiment_Analysis_Newspappers_Headlines

Used the BeautifulSoup package to create a webscraper to gather the headlines of the biggest Swedish Newspappers.  

Currently there are no pre-trained swedish NLP models to predict sentiment on texts. So I had to use a Google translate API to translate the headlines into english so the model could predict the sentiment. This is ofcourse not the optimal way, but it did a good job of translating.

Then I used a Wordcloud to get a sense of what all the newspappers was writing about for that day. Because I need some way to evaluate to see if the models did a good job of predicting the headlines

To the sentiment analysis part I used the TextBlob- and Vader model to predict the Sentiment. Unfortunately the models did not do a quite good job of predicting the sentiment. Although it's hard to evaluate around 300-350 headlines, that was my impression of the performance. 

