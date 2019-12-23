# Disaster Response Pipeline Project

### Overview
This projects helps set up a machine learning pipeline on top of messages pertaining to disaster response messages. We will first wrangle and clean the raw data and then train our machine learning models using the cleaned data. Once the models are built, we will be able to serve the model via a Flask webapp in order to predict on user input messages.

### Pre-requisites
The jupyter notebooks will work on a default installation of anaconda. However in order to get the python webapp to run, you will have to run the custom conda environment provided. Please run the following command to set up the right environment.

```bash
conda env create -f environment.yml
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project contents
The project contains the following

- `jupyter` - This folder contains the jupyter notebooks where we do prelim data wrangling and build the right models for predcition. These files can be run individually in order to get better understanding of the data and the models / hyperparams used.

- `data` - This contains the python code to wrange and clean the CSV files and save this in a sqllite database
- `models` - The contains the python code that is used to create a classifier to fit on the training data and generate a trained model. The model is then saved to the filesystem so that it can be easily loaded up and used when required to make predictions
- `app` - This contains a flask webapp that can be used to predict on user input data.

### License
Feel free to use all resources here for educational and non-commercial purposes