import json
import plotly
import pandas as pd


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re

stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

def tokenize(text):
   
   """ Cleans and tokenizes text
   Parameters: 
   text (string):  The text to tokenize
   
   Returns: List of words representing the tokenized string
   """
   
   # Remove punctuation
   text = re.sub(r'[^a-zA-z0-9]',' ', text.lower())

   # Tokenize the text
   tokenized = word_tokenize(text)
   
   # Remove stop words
   tokenized = [lemmatizer.lemmatize(w).strip() for w in tokenized if w not in stopwords]
   return tokenized

def create_graph(genre_name, count_tuples) :
    
    """ Finds five most frequent words and counts per genre in the dataframe
    Parameters:
    genre_name (string) :  Message genre for which words were counted
    count_tuples (List (tuples)) : List of tuples where each tuple is a word and the count of that word for the genre
    Returns: Dict representing a Plotly graph
    """
    
    words = [count_tup[0] for count_tup in count_tuples]
    word_counts = [count_tup[1] for count_tup in count_tuples]
    
    graph = {
        'data': [
            Bar(
                x=words,
                y=word_counts
                )
            ],
        'layout': {
            'title': 'Most Frequent Words in ' + genre_name + ' Messages',
            'yaxis': {
                'title': 'Counts'
                },
            'xaxis': {
                'title': 'Words'
                }
            }
        }
    return graph


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponses', engine)

# Create the counts_dict outside of index function so it doesn't have to recreate every time main page is entered
#counts_dict = get_most_freq_words(df)
# load count_dict that was saved off during data processing
counts_dict = joblib.load("../data/counts_dict.pkl")

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    
    # create visuals
    
    graphs = []
    
    for genre_name, count_tuples in counts_dict.items() :
        graphs.append(create_graph(genre_name, count_tuples))
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
