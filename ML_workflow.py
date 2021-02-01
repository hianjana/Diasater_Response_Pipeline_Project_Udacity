#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries for data load
import pandas as pd
import re
from sqlalchemy import create_engine

# Library for data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Libraries for data cleaning and pre-processing
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator,TransformerMixin

# Libraries for pipeline and model building
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Libraries for model evaluation
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, make_scorer
import pickle

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def load_data(db, database_name, feature, pos):
    
    # load data from database
    engine = create_engine(db)
    
    sql = 'SELECT * FROM ' + database_name
    df = pd.read_sql(sql, engine)
    X = df[feature]
    y = df.iloc[:,pos:]
    
    return X, y


# In[5]:


def tokenize(text):
    '''
    To clean and pre-process the raw data. Here are the steps done by the function
    1) Clean the data to remove all HTML tags
    2) Normalize by converting the text to lowercase and removing punctuations
    3) Split text into tokens
    4) Remove English stop words
    
    '''
    
    # Replace url tags with the string 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Convert to lowercase
    text = text.lower() 
    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize each word to create clean tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='n').strip() for word in words]
    clean_tokens = [lemmatizer.lemmatize(token, pos='v').strip() for token in lemmatized_tokens]
    
    return clean_tokens


# In[59]:


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        '''
        # Debugging
        for sentence in sentence_list:
            len_sent = len(sentence)
            if len_sent < 10:
                print(len_sent, ' : ' ,sentence)
        '''        
        
        for sentence in sentence_list:
            print(sentence)
            
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            print(pos_tags)            
            
            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

            return False
            

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged)


# In[54]:


def model_pipeline():
    text_pipeline = Pipeline([
                            ('vect', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf', TfidfTransformer())
                            ])
    pipeline = Pipeline([
                        ('feature_union', FeatureUnion([('text_pipeline', text_pipeline), ('verb_extractor', StartingVerbExtractor())]))
                        #,('clf', RandomForestClassifier(n_estimators=10))
                        ]) 
    return pipeline


# In[31]:


def train(X_train, X_test, y_train, y_test, model):
    '''
    This function will train the model and predict on the test data.
    It returns the predicted y values
    '''   
    
    model.fit(X_train, y_train) # Train the model on train data
    
    y_pred = model.predict(X_test) #Predict on test data
    
    target_col_names = y_test.columns
    y_pred = pd.DataFrame(y_pred, columns = target_col_names) #Convert y_pred to a dataframe
    
    return y_pred


# In[58]:


def main():
    X, y = load_data('sqlite:///DisasterMessages.db', 'DisasterMessages', 'message', 4)
    
    y.drop(['child_alone'], axis=1, inplace=True) # Drop this column from the target as it has only 0s
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 40)

    verb_extractor = StartingVerbExtractor()
    verb_extractor.fit_transform(X_train)
    
    #model = model_pipeline()    
    #y_pred = train(X_train, X_test, y_train, y_test, model)


# In[60]:


main()


# In[ ]:




