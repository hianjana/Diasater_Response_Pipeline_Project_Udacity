import sys
# Libraries for data load
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Libraries for model evaluation
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, make_scorer
import pickle

import warnings
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    '''
    This function will load the data from the SQL database.
    Also it will create X and y.
    X is the text feature.
    y is the target and a few columns will be excluded from the original data.
    '''
    
    db = 'sqlite:///' + database_filepath
    engine = create_engine(db)
    
    # Extract database name from the database path
    pos = database_filepath.rindex('/') # Find the position of the last occurance of "/" 
    db_nm = database_filepath[pos+1:]
    db_name = db_nm.replace('.db', '')    
    
    sql = 'SELECT * FROM ' + db_name
    df = pd.read_sql(sql, engine)
    X = df['message']
    y = df.iloc[:,4:]
    
    return X, y


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


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    The purpose of this class is to create a feature which indicates whether or not a sentence starts with a verb.
    It returns a True if the sentence starts with a verb or it is a re-tweet and False otherwise.
    '''

    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)      
        
        for sentence in sentence_list:
            
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            if len(pos_tags) > 1:
                # index pos_tags to get the first word and part of speech tag
                first_word, first_tag = pos_tags[0]
            
                # return true if the first word is an appropriate verb or RT for retweet
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
                else:
                    return 0
            else:
                return 0
            

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        X_tagged = pd.DataFrame(X_tagged)
        X_tagged = X_tagged.replace(np.nan, 0)
        
        return X_tagged


def build_model():
    '''
    This functions creates a pipeline of various features: CountVectorizer, TfidfTransformer.
    For model building, SGDClassifier is used. 
    The function returns a pipeline model using all of the above.
    '''

    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                        ('tfidf',TfidfTransformer()),
                        ('classifier', MultiOutputClassifier(SGDClassifier())) 
                        ])
                        
    parameters = {'classifier__estimator__loss' : ['hinge', 'log'],
                  'classifier__estimator__alpha' : [0.00005, 0.001]
                  }
                  
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2)
    
    return model
    
def train(X_train, X_test, y_train, y_test, model):
    '''
    This function will train the model and predict on the test data.
    It returns the predicted y values
    '''   
    
    print('Model training started')
    model.fit(X_train, y_train) # Train the model on train data
    print('Model training completed')
    
    print('Prediction started')
    y_pred = model.predict(X_test) #Predict on test data
    print('Prediction complete')
    
    target_col_names = y_test.columns
    y_pred = pd.DataFrame(y_pred, columns = target_col_names) #Convert y_pred to a dataframe
    
    return y_pred


def model_evaluation(y_true, y_pred):
    '''
    This function will loop through each target column and calculate the accuracy, precision, recall and F1-score.
    The resulting dataframe will be returned.
    '''    
    
    eval_measures = {} # Dictionary to store the performance measures
    target_col_names = y_true.columns # Get all the column names present in the target
    
    for col in target_col_names:
        eval_measures[col] = {}
        precision, recall, f1_score, support = precision_recall_fscore_support(y_true.loc[:,col], y_pred.loc[:,col], average='macro')
        accuracy = accuracy_score(y_true.loc[:,col], y_pred.loc[:,col])
        
        eval_measures[col]['f1_score'] = f1_score
        eval_measures[col]['precision'] = precision
        eval_measures[col]['recall'] = recall        
        eval_measures[col]['accuracy'] = accuracy 
    
    df_eval_measures = pd.DataFrame(eval_measures)
    df_eval_measures = df_eval_measures.transpose()
    df_eval_measures = df_eval_measures.sort_values(by=['f1_score', 'precision', 'recall', 'accuracy'], ascending=False)
    
    print(df_eval_measures)


def save_model(model, model_filepath):
	pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    This is the main function which does the following steps:
    1) Loads the data
    2) Provides data visualization
    3) Create train and test datasets.
    4) Model building: a basic model, hyperparamater tuned model, an improved model with additional feature.
    5) Model evaluation
    6) Storing the models as pickles.
    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        print('Data load complete')
        
        # Drop this column from the target as it has only 0s    
        y.drop(['child_alone'], axis=1, inplace=True) 
    
        # Split the data into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)
    
        # Model building
        print('Building model...')
        model = build_model() 
    
        # Model training
        print('Training model...')
        y_pred = train(X_train, X_test, y_train, y_test, model)
    
        # Model evaluation
        print('Evaluating model...')
        model_evaluation(y_test, y_pred)
    
        # Save the model as a pickle
        #save_model(model, model_filepath)
        print('Trained model saved!')
        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')    


if __name__ == '__main__':
    main()
