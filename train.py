"""
Code to train model on provided train data.
Written by Huy Anh Nguyen (anh.h.nguyen@stonybrook.edu)

Last modified by Huy Anh Nguyen
Date: May 20, 2021
"""

import argparse
import os 
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from utils import preprocess, get_folder_name
from model import HierarchicalModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop

parser = argparse.ArgumentParser(description="Arguments to train model")

parser.add_argument("-d", "--data", type=str, default='train.csv', help="Path to train data")
parser.add_argument("-ep", "--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("-b", "--batch", type=int, default=1024, help="Number of examples per batch")
parser.add_argument("-lr", "--learning_rate", type=float, default=5e-4, help="Model learning rate")
parser.add_argument("--save_encoders", type=bool, default=True, help="Choose to save the encoders or not")
parser.add_argument("-v", "--verbose", type=int, default=1)
args = parser.parse_args()


print("Preprocessing data ...")
data = preprocess(args.data)

print("-"*20, "\n")

code_declaration = data[0]
chapter_label = data[1]
heading_label = data[2]
sub_heading_label = data[3]
country_extension_label = data[4]

n_chapter_classes = chapter_label.shape[1]
n_heading_classes = heading_label.shape[1]
n_sub_heading_classes = sub_heading_label.shape[1]
n_country_extension_classes = country_extension_label.shape[1]



model = HierarchicalModel(n_chapter_classes, n_heading_classes, n_sub_heading_classes, n_country_extension_classes)

weight_folder_dir = get_folder_name('model_weights/model_{}')
os.mkdir(weight_folder_dir)
weight_path = weight_folder_dir + "/model_weight.ckpt"
model_parameters = model.get_parameters()
with open(os.path.join(weight_folder_dir, 'model_parameters.json'), 'w') as outfile:
    json.dump(model_parameters, outfile)
    
checkpoint = ModelCheckpoint(weight_path, save_weights_only=True, monitor='loss', verbose=2, save_best_only=True, mode='min')

optimizer = Adam(args.learning_rate)
model.compile(loss='categorical_crossentropy',optimizer = optimizer, metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=2, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=2,min_lr=1e-5)

print("Traning model with batch size {} and {} epochs".format(args.batch, args.epochs))

model.fit(code_declaration,
    [chapter_label, heading_label, sub_heading_label, country_extension_label],
    epochs=args.epochs,
    batch_size=args.batch,validation_split=0.1, callbacks=[reduce_lr, early_stop, checkpoint], verbose=args.verbose)

print("-"*20, "\n")
print("Model weights and parameters are saved into {}".format(weight_folder_dir))