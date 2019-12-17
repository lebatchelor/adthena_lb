import pickle
import pandas as pd
import numpy as np
import sys
import re
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import datetime
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


def preprocess_data(file_path):
    try:
        if '.csv' in file_path:
            data = pd.read_csv(file_path, header=None)
        else:
            data = pd.read_csv(file_path, sep='/n', header=None)

        print('Data loaded successfully')
        data['term_processed'] = data[0].apply(clean_text)
        print('Text cleaned successfully')
        return data
    except Exception as err:
        raise err


def clean_text(text):
    try:
        replace_punctuation = re.compile('[/(){}\[\]\|@,;]')
        replace_symbols = re.compile('[^A-Za-z\s]')
        stopwrds = set(stopwords.words('english'))

        text_processed = text.lower()
        text_processed = replace_punctuation.sub(' ', text_processed)
        text_processed = replace_symbols.sub('', text_processed)
        text_processed = ' '.join(word for word in text_processed.split() if word not in stopwrds)
        return text_processed
    except Exception as err:
        raise err


def train_model(X_train, y_train):
    print('Training model...')
    try:
        clf = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
                       ('svc', LinearSVC(penalty='l2', multi_class='ovr', C=0.4, max_iter=5000))
                       ])

        clf.fit(X_train, y_train)
        print('Model trained successfully')
        return clf
    except Exception as err:
        raise err


def serialize(model, filename):
    print(f'Serializing model to file: {filename}')
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            print('Model saved')
    except Exception as err:
        raise err


def deserialize(filename):
    print(f'De-serializing model from file: {filename}')
    try:
        with open(filename, 'rb') as f:
            pickle_model = pickle.load(f)
            print('Model de-serialized')
            return pickle_model
    except Exception as err:
        raise err


def get_predictions(clf, X_test):
    print('Retrieving model predictions..')
    try:
        ypred = clf.predict(X_test)
        ypred = pd.DataFrame(ypred).to_csv(f'predictions_{str(datetime.date.today())}.csv')
        print('Predictions received and saved to file')
        return ypred
    except Exception as err:
        raise err


def get_test_stats(y_test, y_pred):
    try:
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        precision = np.diag(cm) / np.sum(cm, axis=0)
        precision = np.nan_to_num(precision)
        print('accuracy: ', accuracy)
        print('Precision: ', np.mean(precision))
        print('Recall: ', np.mean(recall))
        print('Macro_Precision: ', precision_score(y_test, y_pred, average='macro'))
        print('Micro_Precision: ', precision_score(y_test, y_pred, average='micro'))
        print('macro_f1: ', f1_score(y_test, y_pred, average='macro'))
        print('micro_f1: ', f1_score(y_test, y_pred, average='micro'))
        print('avg_f1: ', f1_score(y_test, y_pred, average='weighted'))
    except Exception as err:
        raise err
