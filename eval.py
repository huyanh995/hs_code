import argparse
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from utils import apply_preprocess, get_folder_name, load_preprocess
from model import HierarchicalModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop


parser = argparse.ArgumentParser(description="Arguments to train model")

parser.add_argument("-d", "--data", type=str, default='train.csv', help="Path to train data")
parser.add_argument("-e", "--encoders", type=str, default='data/encoders/default/', help="Path to encoders")
parser.add_argument("-w", "--weights", type=str, default='data/model_weights/default/', help="Path to saved/pretrained model weights and parameters folder")
parser.add_argument("-b", "--batch", type=int, default=1024, help="Number of examples per batch")

args = parser.parse_args()

with open(os.path.join(args.weights, 'model_parameters.json'), 'rb') as f:
    model_parameters = json.load(f)


encoders = load_preprocess(args.encoders)

tokenizer = encoders[0]
df = pd.read_csv(args.data, dtype=str)
list_tokens = tokenizer.texts_to_sequences(df['TEN_HANG'])
description_data = pad_sequences(list_tokens, model_parameters['max_sequence_length'])

# Load saved model parameters
n_chapter_classes = model_parameters['n_chapter_classes']
n_heading_classes = model_parameters['n_heading_classes']
n_sub_heading_classes = model_parameters['n_sub_heading_classes']
n_country_extension_classes = model_parameters['n_country_extension_classes']


model = HierarchicalModel(n_chapter_classes, n_heading_classes, n_sub_heading_classes, n_country_extension_classes)

pred = model.predict(description_data, batch_size=args.batch_size)
print(pred.shape)