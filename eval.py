"""
Inference model on provided test data.
Written by Huy Anh Nguyen (anh.h.nguyen@stonybrook.edu)

Last modified by Huy Anh Nguyen
Date: May 20, 2021
"""

import argparse
import os 
import json
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from utils import apply_preprocess, get_folder_name, load_preprocess, transform_result
from model import HierarchicalModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser(description="Arguments to train model")

parser.add_argument("-d", "--data", type=str, default='test.csv', help="Path to train data")
parser.add_argument("-e", "--encoders", type=str, default='data/encoders/default/', help="Path to encoders")
parser.add_argument("-w", "--weights", type=str, default='data/model_weights/default/', help="Path to saved/pretrained model weights and parameters folder")
parser.add_argument("-b", "--batch", type=int, default=512, help="Number of examples per batch")

args = parser.parse_args()

# Load model parameters and encoders
with open(os.path.join(args.weights, 'model_parameters.json'), 'rb') as f:
    model_parameters = json.load(f)

n_chapter_classes = model_parameters['n_chapter_classes']
n_heading_classes = model_parameters['n_heading_classes']
n_sub_heading_classes = model_parameters['n_sub_heading_classes']
n_country_extension_classes = model_parameters['n_country_extension_classes']

encoders = load_preprocess(args.encoders)

# Preprocess data
print("Preprocess data ...")
tokenizer = encoders[0]
df = pd.read_csv(args.data, dtype=str)
list_tokens = tokenizer.texts_to_sequences(df['TEN_HANG'])
description_data = pad_sequences(list_tokens, model_parameters['max_sequence_length'])
print("Done", "-"*20, "", sep="\n")


# Initiate model and load saved weights
print("Load saved weights ... ")
model = HierarchicalModel(n_chapter_classes, n_heading_classes, n_sub_heading_classes, n_country_extension_classes)
weight_path = os.path.join(args.weights, 'model_weight.ckpt')
model.load_weights(weight_path)
print("Done", "-"*20, "", sep="\n")


# Inference 
print("Inference ...")
raw_pred = model.predict(description_data, batch_size=args.batch)
onehot_encoders = encoders[1:]
pred = transform_result(raw_pred, onehot_encoders)

print("Saving result ...")
df["PRED"] = pred
out_name = args.data.split("/")[-1].split(".")[0] + "_result.csv"
df.to_csv(out_name)

print("Done", "-"*20, "", sep="\n")