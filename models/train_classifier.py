"""Builds a prediction model to determine disaster response against tweets

This file will read in the cleaned messages and categorization data and use that
to train a prediction model. Once trained and evaluated on the validation data, 
the model is saved so that it can be used in further predictions.
"""

# Core modules
import sys
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine

# SKLearn modules for text processing 
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# SKLearn modules for feature extraction and classification
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def load_data(database_filepath):
    """Loads the messages from the database

    Args:
        database_filepath: The path to the SQLlite database

    Returns:
        X: The vector of messages
        Y: The vector of labels for the messages
        category_names: An array of category names that the columns of Y
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('messages', engine)

    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()

    return X, Y, category_names

    
def tokenize(text):
    """Defines the tokenizer for the text messages

    The method first tokenizes the text into words using nltk.
    Then we will remove all the words that contains only special characters
    or punctuations. Next we will remove all the stop words based on wordnet
    Finally we will lemmatize the words to its root for both nouns and verbs
    
    Args:
        text: A tweet or message that has to be analyzed

    Returns:
        The tokens associated with the given text
    """
    
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if re.search('[A-Za-z0-9]', tok) != None]
    tokens = [tok.lower().strip() for tok in tokens 
                    if tok not in stopwords.words('english')]
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(wnl.lemmatize(tok), pos='v') for tok in tokens]            

    return tokens


def build_model():
    """Creates and returns a machine learning classifier 

    The method uses a pipeline to chain togther text processing, 
    feature extraction, and classifiers into a single model for prediction.

    The best params to be used were identified using GridSearchCV to be
        {'clf__estimator__max_features': 'sqrt',
        'cv__max_features': 10000,
        'cv__ngram_range': (1, 1),
        'cv__tokenizer': <function __main__.tokenize_lemmatize(text)>}

    Also comparing RandomForestClassifier, SVM classifier, and Multinomial
    Naive Bayes classifier, the best accuracy and f1-score was obtained by
    using Random Forest Classifier. Hence we'll use that in the pipeline.

    Returns: 
        A pipeline that can be used to fit and predict on data
    """

    pipeline = Pipeline([
        ('cv', CountVectorizer(
            tokenizer=tokenize, ngram_range=(1,1), max_features=10000)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, max_features='sqrt')))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Predicts on the test data using the provided model and prints scores

    Args:
        model: The prediction model that will be used
        X_test: An array of features to predict on
        y_test: A multi-label array of expected prediction results
        category_names: The actual label names
    """

    Y_pred = model.predict(X_test)
    Y_trans = Y_pred.transpose()

    for i in range(0,Y_test.columns.size):
        print(classification_report(Y_test.iloc[:,i].tolist(), Y_trans[i]))


def save_model(model, model_filepath):
    """Saves the model to the filesystem

    Args:
        model: the model used to fit training data
        model_filepath: the filesystem path where to save the model to
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:3]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()