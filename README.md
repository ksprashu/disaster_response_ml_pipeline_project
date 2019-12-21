# Disaster Response Pipeline Project

### Overview
This projects helps set up a machine learning pipeline on top of messages pertaining to disaster response messages. We will first wrangle and clean the raw data and then train our machine learning models using the cleaned data. Once the models are built, we will be able to serve the model via a Flask webapp in order to predict on user input messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### License
Feel free to use all resources here for educational and non-commercial purposes