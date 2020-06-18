# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:54:49 2020

@author: Rohith
"""


import numpy as np   
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  

##.................Importing train & tst data................############

dataset= pd.read_csv("E:\\Data Science\\Project\\Data\\train.csv")
data= pd.read_csv("E:\\Data Science\\Project\\Data\\test.csv")


## FINDING NULL VALUES...##
dataset.isnull().sum() ## there is 219 null values in condition column which is 0.005% of entire value#
data.isnull().sum()

## droping all NA values from Train  data..##
dataset.dropna(inplace=True)

#############......Pre-processing on train data...###########################

dataset.describe()

dataset['rating'].value_counts()

##removing the na values from train data##
dataset.dropna(inplace=True)

## checking percentage wise distribution of rating..##
dataset['rating'].value_counts(normalize=True)*100

dataset['rating'].value_counts()
dataset['output'].value_counts()
## plotting the ratings to get a clear picture of rating distribution..##     
sns.distplot(dataset['rating'] ,hist=True, bins=100)

dataset['usefulCount'].max()
dataset['usefulCount'].min()

## Finding the Correlation Matrix...##
corrmat = dataset.corr() 
print(corrmat)

# removing the date column as date has not that significance in output##
dataset.drop(["date"],axis=1,inplace=True)

dataset.head()

## cleaning the data..##
## Cleaning the text input for betting understanding of Machine..##

##Converting all review into Lowercase..###
dataset['review']= dataset['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))

## removing punctuation from review..#
import string
dataset['review']=dataset['review'].apply(lambda x:''.join([i for i in x  if i not in string.punctuation]))
                                                 

## Remove Numbers from review...##
dataset['review']=dataset['review'].str.replace('[0-9]','')


## removing all stopwords(english)....###
from nltk.corpus import stopwords

stop_words=stopwords.words('english')

dataset['review']=dataset['review'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

dataset.head(2)

# Lemmatization
from textblob import Word
dataset['review']= dataset['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


## Removing specific words which dont have much significance and higher frequency..##
n_req= ['one','first','effect','side','taking','day', 'month','year','week','im','ive','mg','time','hour','could','lb','two','sideeffect','started','still']

dataset['review']=dataset['review'].apply(lambda x: " ".join(word for word in x.split() if word not in n_req))




## subjectvity & polarity of each review rows...##
from textblob import TextBlob

dataset['polarity'] = dataset['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
dataset['subjectivity'] = dataset['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

## Finding sentiment through VADER sentiment Analyzer..##
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(dataset.iloc[4]['review'])

dataset['vad_scores'] = dataset['review'].apply(lambda review:sid.polarity_scores(review))
dataset['vad_compound'] = dataset['vad_scores'].apply(lambda d:d['compound'])


########...finding Correlation in the data....###
corrmat = dataset.corr() 
print(corrmat)


## ....Finding most common occuring words in Corpus...##
review_str=" ".join(dataset.review)
text=review_str.split()

from collections import Counter
counter= Counter(text)
top_100= counter.most_common(100)
print(top_100)

###############.....Finding Unique Words from the entire corpus...##################
len(set(counter))


###### WordCloud formation for better understanding of the data...##

from wordcloud import WordCloud
from PIL import Image
apple = np.array(Image.open( "C:/Users/Sayan Mondal/Desktop/apple.jpg"))

wordcloud= WordCloud(width= 3000,
                     height=3000,mask=apple,
                     background_color='black'
                     ).generate(review_str)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
