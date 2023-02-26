import os
import pickle
import sys

from train_model import read_message, train_model

if __name__ == '__main__':
    arg = sys.argv
    clean = []
    spam = []
    if arg[1] == "-info":
        output_file = open(arg[2], "w")
        info = "Anti Spam Filter\n" \
               "Palade Robert-Eusebiu\n" \
               "Robert\n" \
               "Version 3.0"
        output_file.write(info)
        output_file.close()
    elif arg[1] == "-scan":
        path = os.getcwd()
        output_file = open(arg[3], "w")

        test_dataset = []
        for file in os.scandir(arg[2]):
            test_dataset.append([file, read_message(file.path)])

        saved_list_file = "trained_model.sav"

        # if model is trained
        if os.path.isfile(saved_list_file):
            loaded_list = pickle.load(open(saved_list_file, 'rb'))

            for file_message in test_dataset:
                message_text = [file_message[1]]
                features_test_dataset = loaded_list[0].transform(message_text)
                filename = str(file_message[0].name)
                prediction = str(loaded_list[1].predict(features_test_dataset[0])[0])
                output_file.write(filename + "|" + prediction + "\n")
                output_file.close()
        # if model is not trained
        else:
            train_model(saved_list_file, test_dataset, output_file)
