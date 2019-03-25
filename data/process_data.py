import sys

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
   
   """ Loads data into a Pandas dataframee
   Parameters:

   disaster_messages.csv
   messages_filepath (string): Name and filepath of the messages to load
   categories_filepath (string): Name filepath of the categories to load
   Returns: df (Pandas.DataFrame) : Dataframe containing loaded data
   """
   messages = pd.read_csv(messages_filepath)
   categories = pd.read_csv(categories_filepath)
   return messages.merge(categories, on = 'id')

def clean_data(df):
   
   """ Cleans dataframe
   Parameters:

   df (Pandas.DataFrame) : dataframe to clean

   Returns (Pandas.DataFrame) : Cleaned dataframe
   
   """

   # select the first row of the categories dataframe
   categories = df['categories'].str.split(';', expand = True)
   
   # use this row to extract a list of new column names for categories.
   row = categories.iloc[0, :]

   category_colnames = row.apply(lambda cat : cat.split('-')[0]).tolist()

   # rename the columns of `categories`
   categories.columns = category_colnames
   
   for column in categories:
      # set each value to be the last character of the string
      categories[column] = categories[column].apply(lambda val : val.split('-')[1])

      # convert column from string to numeric
      categories[column] = pd.to_numeric(categories[column])
      
   # drop the original categories column from `df`
   df.drop(labels = ['categories'], axis = 1, inplace = True)
   
   # concatenate the original dataframe with the new `categories` dataframe
   df = pd.concat([df, categories], axis=1)
   
   # Drop column that only has 0 values
   df = df.drop(labels = 'child_alone', axis=1)

   # Drop rows related equal 2
   df = df[df.related != 2]

   # Drop duplicates
   df = df[np.logical_not(df.duplicated())]

   # Drop any row where messages is null
   df = df.dropna(axis=0, subset=['message'])

   return df


def save_data(df, database_filename):
   
    """ Saves dataframe to SQL database

    Parameters:
    
    df (Pandas.DataFrame) : dataframe to save
    database_filiename (string) : Name of database saved to
    
    Returns : Nothing
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponses', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:
        
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
