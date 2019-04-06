# Disaster Response Pipeline Project

Train a disaster response pipeline to classify messages into a a variety of genres. 
Serve a website with visualizations of the most frequent words per genre and functionality to classify a user entered message

### Files

0. README.md : This file
1. data/disaster_messages.csv : Disaster response messages and their genre
2. data/disaster_categories.csv: The disaster reponse message categories a given message is categorized under
3. data/process_data.py : Cleans data from data/disaster_messages.csv and data/disaster_categories.csv and saves cleaned dataframe to data/DisasterResponse.db
4. models/train_classifier.py : Trains a disaster response pipeline to classify the disaster response messages into categories using  the data in DisasterResponse.db. 
							Saves trained classifier a pickle file
5. app/run.py : Launches the interactive web application that allows users to enter their own messages for classification
6. app/master.html : Main page of the web application
7. app/go.html : Page to display the classifications of user entered message

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
