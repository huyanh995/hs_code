"""
Tokenizer
Onehot encoder
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


enc_dir = 'data/encoders/'

FILES = ['tokenizer.pkl', 'enc_chapter.pkl', 
            'enc_heading.pkl', 'enc_sub_heading.pkl', 'enc_country_extension.pkl']

HEADERS = ['TEN_HANG', 'CHAPTER', 'HEADING', 'SUB_HEADING', 'COUNTRY_EXTENSION']

NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 200

def preprocess(train_path, records=True):
    """
    Apply preprocessing to train data, including:
    tokenizing and one-hot encoders for four labels
    """
    df = pd.read_csv(train_path, dtype=str)

    if not all(x in df.columns for x in HEADERS):
        raise AssertionError("Data is not in the right format!")

    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(df[HEADERS[0]])
    list_tokens = tokenizer.texts_to_sequences(df[HEADERS[0]])
    description_data = pad_sequences(list_tokens, MAX_SEQUENCE_LENGTH)
    data = [description_data]
    encoders = []
    print(type(data))
    for i in range(1, len(HEADERS)):
        label = np.expand_dims(df[HEADERS[i]].to_numpy(), axis=1)
        enc = OneHotEncoder()
        enc_data = enc.fit_transform(label)
        data.append(enc_data)
        encoders.append(enc)

    if records:
        new_folder_dir = get_folder_name()
        os.mkdir(new_folder_dir)
        for name, enc in zip(FILES, encoders):   
            file_path = os.path.join(new_folder_dir, name)

            with open(file_path, 'wb') as f:
                pickle.dump(enc, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("All encoders are saved into {}".format(new_folder_dir))

    return data, encoders

def load_preprocess(encoder_dir = 'data/encoders/default'):

    encoders = []

    for name in FILES:
        with open(os.path.join(encoder_dir, name), 'rb') as f:
            enc = pickle.load(f)
            encoders.append(enc)   

    return encoders

def get_folder_name():
    new_dir = 'encoders_{}'
    count = 0
    while True:
        temp_folder = enc_dir + new_dir.format(count)
        if not os.path.isdir(temp_folder):
            break
        count += 1

    return temp_folder
        



def apply_preprocess(test_df, encoders):

    if not all(x in test_df.columns for x in HEADERS):
        raise AssertionError("Data is not in the right format!")
    
    tokenizer = encoders[0]
    list_tokens = tokenizer.texts_to_sequences(test_df[HEADERS[0]])
    description_data = pad_sequences(list_tokens, MAX_SEQUENCE_LENGTH)
    data = [description_data]

    for i in range(1,len(HEADERS)):
        enc = encoders[i]
        data.append(enc.transform(test_df[HEADERS[i]]))
        
    return data


if __name__ == '__main__':
    preprocess('train.csv')