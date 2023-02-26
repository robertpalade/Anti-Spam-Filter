import os
import pickle

import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer


def read_message(filename):
    f = open(filename, "r", encoding="latin-1")
    return str(f.readlines())


def train_model(saved_list_file, test_dataset, output_file):
    training_dataframe = []

    for filename in os.scandir("Clean"):
        clean_dataframe = {"Category": "cln", "Message": read_message(filename.path)}
        training_dataframe.append(clean_dataframe)
    for filename in os.scandir("Spam"):
        spam_dataframe = {"Category": "inf", "Message": read_message(filename.path)}
        training_dataframe.append(spam_dataframe)

    trained_dataframe = pd.DataFrame(training_dataframe)
    category_training = trained_dataframe["Category"]
    message_training = trained_dataframe["Message"]

    cv = CountVectorizer()
    features = cv.fit_transform(message_training)
    model = svm.SVC()
    model.fit(features, category_training)

    saved_list = [cv, model]
    pickle.dump(saved_list, open(saved_list_file, 'wb'))

    for file_message in test_dataset:
        message_text = [file_message[1]]
        features_test_dataset = cv.transform(message_text)
        filename = str(file_message[0].name)
        prediction = str(model.predict(features_test_dataset[0])[0])
        output_file.write(filename + "|" + prediction + "\n")
        output_file.close()
