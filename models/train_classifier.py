import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import nltk
from nltk import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
   
   """ Cleans and tokenizes text
    Parameters: 
    database_filepath (string): The name and filepath of the database to load
    
    Returns: X (list) : List of disaster response messages - each entry in list is represents a single message
             Y (list) : List of 1's and 0's indicating whether a given message fell into a specific disaster response category
             category_names (list): List of names of each of the disaster response categories identified by Y
    """
   # load data from database
   engine = create_engine('sqlite:///' + database_filepath)
   df = pd.read_sql_table(database_filepath, engine)
   X = df.loc[:, 'message'].values
   Y = df.iloc[:, 4:].values
   category_names = df.columns.values[4:]
   return X, Y, category_names

def tokenize(text):
   
   """ Cleans and tokenizes text
   Parameters: 
   text (string):  The text to tokenize
   
   Returns: List of words representing the tokenized string
   """
   
    lemmatizer = WordNetLemmatizer()
    
    # Lower case
    text = str.lower(text)
    
    # Remove punctuation
    text = re.sub(r'[^a-zA-z0-9]',' ', text)
    
    # Tokenize the text
    text = word_tokenize(text)
    
    # Remove stop words
    text = [lemmatizer.lemmatize(w).strip() for w in text if w not in stopwords.words('english')]
    return text


def build_model():
   
   """ Builds a classifier pipeline to identify the disaster response categories a given message falls into

   Returns: Classifier model
   """
   
   
   pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(estimator = RandomForestClassifier()))])
   
   return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
   
   """ Evaluate a classifier for accuracy, precision and recall
   
   Parameters:
   
    model (object):  Model to evaluate
    X_test (array):  Test data to evaluate
    Y_test (array):  Target values for prediction
    category_names (list): List of strings indicating each disaster response category to predict / evaluate

    Returns: Nothing
    """
   # Predict on the test data
   y_pred = model.predict(X_test)
   
   for index in np.arange(0, y_pred.shape[1]) :
      accuracy = accuracy_score(Y_test[:, index], y_pred[:, index])
      precision = precision_score(Y_test[:, index], y_pred[:, index])
      recall = recall_score(Y_test[:, index], y_pred[:, index])

      accuracies.append(accuracy)
      precisions.append(precision)
      recalls.append(recall)
      
      print('\n', str.upper(category_names[index]))
      print('Accuracy: ', accuracy)
      print('Precision: ', precision)
      print('Recall: ', recall)

   print('Average accuracy: ', np.mean(accuracies))
   print('Median accuracy: ', np.median(accuracies))
   print('Average precision: ', np.mean(precisions))
   print('Median precision: ', np.median(precisions))
   print('Average recall: ', np.mean(recalls))
   print('Median recall: ', np.median(recalls))
   
   return


def save_model(model, model_filepath):
   
    """ Saves a trained model for later use
    Parameters:
    
    model (object):  Model to save
    model_filepath (string): Filepath and name to save file to

    Returns: Nothing
    """
    
    pickle.dump(model, open(filename, model_filepath='wb'))
    
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
