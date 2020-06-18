import pandas as pd
import numpy as np
import nltk
import re
import seaborn as sns
from numpy import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# In[6]:


df=pd.read_csv("train1.csv")
df.head()


# In[3]:


#------------------EDA-----------------------#


# In[4]:


df.shape


# In[5]:


# Statistics Summary
df.describe()


# In[6]:


# check types
df.dtypes


# In[6]:


# check for missing data
df.isna().sum()


# In[7]:


sns.pairplot(df)


# In[8]:


#Top 10 conditions
conditions = df.condition.value_counts().sort_values(ascending=False)
conditions[:10]


# In[9]:


plt.rcParams['figure.figsize'] = [13, 8]
conditions[:10].plot(kind='bar',color='indigo')
plt.title('Top 10 Most Common Conditions')
plt.xlabel('Condition')
plt.ylabel('Count');


# In[10]:


#Number of drugs per condition

con_dn = df.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
con_dn[0:20].plot(kind="bar", figsize = (13,6), fontsize = 10,color="orange")
plt.xlabel("Condition", fontsize = 16)
plt.ylabel("Number of drugs", fontsize = 16)
plt.title("Top20 : The number of drugs per condition.", fontsize = 20)


# In[12]:


#Top 10 drugs with 10/10 rating

# Setting the Parameter
sns.set(font_scale = 1.2, style = 'darkgrid')
plt.rcParams['figure.figsize'] = [15, 8]

rating = dict(df.loc[df.rating == 10, "drugName"].value_counts())
drugname = list(rating.keys())
drug_rating = list(rating.values())

sns_rating = sns.barplot(x = drugname[0:10], y = drug_rating[0:10])

sns_rating.set_title('Top 10 drugs with 10/10 rating')
sns_rating.set_ylabel("Number of Ratings")
sns_rating.set_xlabel("Drug Names")
plt.setp(sns_rating.get_xticklabels(), rotation=90);


# In[13]:


#distribution of ratings

df.rating.hist(color='purple')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks([i for i in range(1, 11)]);


# In[19]:


from wordcloud import WordCloud, STOPWORDS


# In[20]:


def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(10,10), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(df["review"], title="Word Cloud of review")


# In[17]:


corelation_train=df.corr()
print(corelation_train)


# In[20]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
# Create list (cast to array) of compound polarity sentiment scores for reviews
sentiments = []

for i in df.review:
    sentiments.append(sid.polarity_scores(i).get('compound'))
    
sentiments = np.asarray(sentiments)


# In[23]:


df['sentiment'] = pd.Series(data=sentiments)
useful_train = df.reset_index(drop=True)
useful_train.iloc[:15]


# In[24]:


df['sentiment'].hist(color='skyblue', bins=30)
plt.title('Compound Sentiment Score Distribution')
plt.xlabel('Scores')
plt.ylabel('Count');


# In[ ]:


#---------------------Data cleaning----------------------#


# In[7]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
df['review'] = df['review'].apply(clean_text)


# In[8]:


#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

# import these modules 
from nltk.stem import WordNetLemmatizer 
lt = WordNetLemmatizer() 
df['review'] = df['review'].apply(lambda x: " ".join([lt.lemmatize(word) for word in x.split()]))
df['review'].head()


# In[9]:


nbs=CountVectorizer(max_features=20934)
X = nbs.fit_transform(df.review)
y=df.output


# In[10]:


#---------------------Model Building----------------------#


# In[11]:


#Splitting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)


# In[16]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

#nbs2 = LinearSVC()
nbs2 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,C=0.55, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=42, max_iter=1000)


nbs2.fit(X_train, y_train)


y_pred = nbs2.predict(X_test)

print('accuracy on train set %s' % accuracy_score(y_train, nbs2.predict(X_train)))
print('accuracy on test set %s' % accuracy_score(y_pred, y_test))


report = classification_report(y_test, y_pred)
print('Classification Report Logistic regression: \n', report)

print(confusion_matrix(y_pred,y_test))


# In[2]:


dt=pd.read_csv("test1.csv")


# In[3]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
dt['review'] = dt['review'].apply(clean_text)


# In[4]:


#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
dt['review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#Lemmetisation
from nltk.stem import WordNetLemmatizer 
lt = WordNetLemmatizer() 
dt['review'] = dt['review'].apply(lambda x: " ".join([lt.lemmatize(word) for word in x.split()]))
dt['review'].head()


# In[5]:


nbs=CountVectorizer()
Xt = nbs.fit_transform(dt.review)
Xt


# In[17]:


preds = nbs2.predict(Xt)
preds


# In[16]:


submission=pd.DataFrame({
                    'Id':dt['Id'],
                     'output':preds
})
submission.to_csv('submission.csv',index=False)


# In[18]:


import  pickle

pickle.dump(nbs2,open('model_l.pkl','wb'))


# In[ ]:




