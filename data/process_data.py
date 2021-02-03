#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline
# 
# The intent of this notebook is to load the 2 files which have disaster response data: messages.csv, categories.csv, clean them, combine them and load it into a SQL database.

# **Sections:**
# 1. Load csv files 
# 2. Data cleaning
# 3. Data merge
# 4. Load data to SQL database

# #### Import required libraries

# In[56]:

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# ### 1. Load csv files

# In[57]:


def load_data(filepath):
    '''
    This fucntion will load data from CSV files
    '''
    df = pd.read_csv(filepath)
    return df


# ### 2. Data cleaning

# In[58]:


def categories_cleaning(categories):
    '''
    This function will do the following:
    Split the values in the categories column on the ; character so that each value becomes a separate column. 
    Use the first row of categories dataframe to create column names for the categories data.
    Rename columns of categories with new column names.
    '''
    
    # create a dataframe of the 36 individual category columns
    categories_split = categories['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories_split.iloc[0]
    
    # use this row to extract a list of new column names for categories
    category_colnames = [each[:-2] for each in row]
    #print(category_colnames)
    
    # rename the columns
    categories_split.columns = category_colnames
          
    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].apply(lambda x: x[-1:])
    
        # convert column from string to numeric
        categories_split[column] = categories_split[column].astype('int')
    
    categories_split['related'].replace({2: 1}, inplace=True)
    
    categories_clean = categories.merge(categories_split, left_index=True, right_index=True, how='inner')
    
    # drop the original categories column
    categories_clean.drop(['categories'], axis=1,inplace=True)
    
    return categories_clean


# ### 3. Data merge

# In[59]:


def data_merge(messages, categories_clean):
    '''
    This function will merge the messages and cleaned categories datasets
    '''
    df = pd.merge(messages, categories_clean,on='id')
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


# ### 4. Load data to SQL database

# In[60]:


def load_into_sql(df, database_filepath):
    '''
    This function will load the datafame to a SQL database
    '''
    
    db = 'sqlite:///' + database_filepath
    
    # Extract database name from the database path
    pos = database_filepath.rindex('/') # Find the position of the last occurance of "/" 
    db_nm = database_filepath[pos+1:]
    db_name = db_nm.replace('.db', '') 
    print('DB path: ', db)
    print('DB name: ', db_name)
    
    engine = create_engine(db)
    df.to_sql(db_name, engine, index=False, if_exists='replace')
    print('Data load complete!')


# In[61]:


def main():
    '''
    This is the main function which calls all other functions to load data from the CSV files, clean them, merge them and
    load it back to a SQL database.
    '''
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
    
        # Load the messages dataset
        messages = load_data(messages_filepath) 
        # Load the categories dataset
        categories = load_data(categories_filepath) 
    
        # Clean the categories dataset and split the 'categories' column into individual columns per category
        categories_clean = categories_cleaning(categories)
    
        # Merge the cleaned categories dataset and messages dataset
        df = data_merge(messages, categories_clean)
    
        # Load to a SQL database
        load_into_sql(df, database_filepath)
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

# In[63]:


if __name__ == '__main__':
    main()


# In[ ]:




