import argparse


from utils import preprocess

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


if __name__ == '__main__':
    print(data_dir, n_epochs, lr, args.save_encoders)