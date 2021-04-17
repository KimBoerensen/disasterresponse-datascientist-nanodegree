import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    ''' load data from sql database and
    return feature dataframe, label-data DataFrame, labels as list'''
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    
    # Split data into features and target
    X = df.message.values 
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    # return features and target
    return X, Y, category_names

def tokenize(text):
    '''clean and tokenize input messages'''
    
    # replace urls with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # tokenize text
    tokens = word_tokenize(text)
        
    # process text further in loop
    clean_tokens = []
    for tok in tokens:
        
        # Remove stop words
        if tok in stopwords.words("english"):
            continue
            
        # Reduce words to their stems
        tok = PorterStemmer().stem(tok)
        
        # Reduce words to their root form
        lemmatizer = WordNetLemmatizer()
        tok = lemmatizer.lemmatize(tok).lower().strip()
        
        # append to list
        clean_tokens.append(tok)
        
    # Remove all non alphabet characters
    clean_tokens = [tok for tok in clean_tokens if tok.isalpha()]
    
    # return clean and tokenized text
    return clean_tokens


def build_model():
    '''build pipeland, set parameter, do Gridsearch and return model'''
    
    # build pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # set parameters
    parameters = {
        'clf__estimator__learning_rate': [0.01, 0.02],
        'clf__estimator__n_estimators': [20, 40]
    }
    
    # Grid Search
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=10)
    
    # return model
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate model with classification_report'''
    
    # run predictions
    Y_pred = model.predict(X_test)
    
    # create classification report
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,i]))

    # Compute overall model accuracy
    accuracy = (Y_pred == Y_test).mean().mean()
    print("Accuracy:")
    print(accuracy)
    

def save_model(model, model_filepath):
    '''save model as pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))


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
