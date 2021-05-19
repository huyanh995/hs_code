import argparse


from utils import preprocess
from model import HierarchicalModel

parser = argparse.ArgumentParser(description="Arguments to train model")

parser.add_argument("-d", "--data", type=str, default='train.csv', help="Path to train data")
#parser.add_argument("-e", "--encoders", type=str, default='data/encoders/default/', help="Path to encoders"
parser.add_argument("-ep", "--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="Model learning rate")
parser.add_argument("--save_encoders", type=bool, default=True, help="Choose to save the encoders or not")
args = parser.parse_args()

data_dir = args.data
n_epochs = args.epochs
lr = args.learning_rate

data = preprocess(data_dir)

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

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
optimizer = Adam(lr = 5e-4)
model.compile(loss='categorical_crossentropy',optimizer = optimizer,metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_acc',patience=8,verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1,min_lr=1e-5)

model.fit(X_train,
    {"hs_chaper": chapter_train, "hs_heading": heading_train, "hs_sub_heading":sub_heading_train,"hs_country_extension":country_extension_train },
    epochs=20,
    batch_size=1024,validation_split=0.1, callbacks=[reduce_lr, early_stop, checkpoint]
)

if __name__ == '__main__':
    print(data_dir, n_epochs, lr, args.save_encoders)