"""

Created: 18 May
Last modified: 18 May by Huy Anh Nguyen (anh.h.nguyen@stonybrook.edu)
"""

from tensorflow.keras.preprocessing.text import one_hot,text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input,Bidirectional,LSTM,Dropout,Permute,Activation,Lambda,Reshape
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow.keras.backend as K


class CustomAttention(keras.layers.Layer):
    def __init__(self, n_classes):
        # Creating Attention Layer
        super(CustomAttention, self).__init__()
        self.dense = Dense(n_classes, activation='linear', use_bias=False)
        self.permute = Permute((2,1))
        self.activation = Activation('softmax')
        self.attention = Lambda(lambda x: K.batch_dot(x[0], x[1]))
        
    def call(self, inputs, n_classes):
        x = self.dense(inputs)
        x = self.permute(x)
        alpha = self.activation(x)
        return self.attention([alpha, inputs])


class HierarchicalModel(keras.Model):
    def __init__(
                self,  
                n_chapter_classes, 
                n_heading_classes, 
                n_sub_heading_classes, 
                n_country_extension_classes,
                num_words=5000,
                embed_size = 512,
                max_sequence_length=200, 
                n_classes=100,
                name = 'hierarchical_clf'):

        super(HierarchicalModel, self).__init__(name=name, **kwargs)

        #self.input = Input(shape=(max_sequence_length,))
        self.embedding = Embedding(num_words, embed_size)
        self.bilstm = Bidirectional(LSTM(max_sequence_length, return_sequences=True), merge_mode='concat')
        self.attention = CustomAttention(n_classes)
        self.fc = Dense(1, activation='linear')
        self.reshape = Reshape((n_classes,))

        self.fc_1 = Dense(n_classes, activation='relu')
        self.fc_2 = Dense(n_classes, activation='relu')
        self.fc_3 = Dense(n_classes, activation='relu')
        self.fc_4 = Dense(n_classes, activation='relu')

        self.chapter_output = Dense(n_chapter_classes, activation='sigmoid')
        self.heading_output = Dense(n_heading_classes, activation='relu')
        self.sub_heading_output = Dense(n_sub_heading_classes, activation='sigmoid')
        self.country_extension_output = Dense(n_country_extension_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.bilstm(x)
        x = self.attention(x)
        x = self.reshape(self.fc(x))
        fc1 = self.fc_1(x)
        fc2 = self.fc_2(fc1)
        fc3 = self.fc_3(fc2)
        fc4 = self.fc_4(fc3)
        chapter = self.chapter_output(fc1)
        heading = self.heading_output(fc2)
        sub_heading = self.sub_heading_output(fc3)
        country_extension = self.country_extension_output(fc4)

        return [chapter, heading, sub_heading, country_extension]

        

        