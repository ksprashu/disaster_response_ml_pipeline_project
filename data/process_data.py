"""Extracts messages and categories from CSVs and returns clean merged data

The file is meant to help with ETL of data related to disaster response messages
It processes the raw CSVs of messages and categories data and extracts the
categories data as columns and merges with the messages data based on the 
common id. The data is then cleaned and saved as a sqllite db for future use.

    Usage Example: 
    
    python process_data.py messages_filepath categories_filepath database_name
"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the data from the provided CSVs for messages and categories

    This method will read the CSVs provided for messages and categories
    and load it into a dataframe. Finally the method will merge the
    two dataframe together into a single dataframe along the id column so that 
    all messages have their categories associated in the same row.
    
    Args:
        messages_filepath: Path to the messages CSV
        categories_filepath: Path to the categories CSV

    Returns: 
        Combined dataframe of messages and associated categories
        columns: id, messages, original, genre, categories
    """

    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='outer', on='id')

    return df


def clean_data(df):
    """Wrangles and cleans the categories data

    This method will take in a dataframe consisting of messages and 
    associated categories as strings. The method will then wrangle the data
    so that each category is in its own column. 

    Args:
        df: Dataframe of messages and categories

    Returns: 
        Combined dataframe of messages and cleaned categories
        columns: id, messages, original, genre, [list of categories]
    """

    # create a dataframe of the 36 individual category columns
    # by extracting the first part of the category name value
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    # and use this row to extract a list of new column names for categories.
    row = categories.iloc[1]    
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert categories columns to values 0 or 1
    # by setting each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """Saves the dataframe as a sqllite db

    Args:
        df: A pandas dataframe with merged messages and categories data
        database_filename: The name to be used while saving the sqllite db file
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:4]

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