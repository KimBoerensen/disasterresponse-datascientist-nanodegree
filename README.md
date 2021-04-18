# disasterresponse-datascientist-nanodegree
___

## Installations
* pandas
* numpy
* sqlalchemy
* sklearn
* nltk
* flask
* plotly

___

## Summary and Project Motivation
The project was part of the Udacity Data Scientist Nanodegree. The goal was to develop a ETL and machine learning pipeline to classify incoming messages during a natural disaster into different categories to efficiently support disaster response management. The end product is a web app where an emergency worker can input a new message and get classification results in several categories so that they can send the messages to an appropriate disaster relief agency.

___

## File Descriptions
The repositary contains XYZ files with XYZ folders next to this README file.

A) app folder
* templates: go.html + master.html (html templates for web app)
* run.py (code to run the web app)

B) data folder
* disaster_category.csv (target data)
* disaster_messages.csv (predictor data)
* process_data.py (ETL pipeline to pre-process text data for model training)
* DisasterResponse.db (database to store processed data for model training)

C) models folder
* train_classifier.py (ML pipeline to train classification model)
* classifier.pkl (trained classifier used in web app)

___

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.
    * To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    * To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

___
## Acknowledgments
* [Udacity](https://www.udacity.com) for providing the [Data Sciencist Nanodegree Programme](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
* [Figure Eight](https://www.appen.com) for providing the data to train the model. 
