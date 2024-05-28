import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv1D, Conv2D, LSTM
from tqdm import tqdm
from glob import glob
import argparse
import warnings
#allows the import of necessary libraries and modules. TensorFlow, Keras, NumPy, SciPy, scikit-learn 
#and some other helper libraries are imported here.

#This code defines the DataGenerator class, which is a custom data generator class. This class is used to process the dataset and return the data at each training step.
class DataGenerator(tf.keras.utils.Sequence):
    #This is the constructor method of the DataGenerator class. The constructor method runs when the data generator object is created and takes the required parameters.
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

#This method determines how many steps the generator produces during a cycle. This is calculated using the number of audio files in the dataset and the specified batch size.
    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

#This method returns data in a given index. The index represents the current batch number during the loop. Retrieves audio files and tags corresponding to this batch number.
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        #This code creates an empty array. X holds audio data and Y holds tags.
        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        #This loop takes the path of each audio file, reads the file and places it in data arrays.
        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

#This method is called at the end of a loop loop and shuffles the aggregated indexes. This is used to change the order of the data samples in each training cycle.
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

#This is the train function that performs the training operation. It takes the training parameters and creates the training dataset.
# This function is the main function that performs the training. Here are the operations this function performs:

# Takes the relevant values from the arguments and assigns them to the variables.
# Defines model parameters and model objects.
# Retrieves dataset paths and labels.
# Converts tags to numeric values.
# Creates training and validation datasets.
def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES': len(os.listdir(args.src_root)),
              'SR': sr,
              'DT': dt}
    models = {'conv1d': Conv1D(**params),
              'conv2d': Conv2D(**params),
              'lstm': LSTM(**params)}
    assert model_type in models.keys(), '{} not an available model'.format(model_type)
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                              labels,
                                                              test_size=0.2,
                                                              random_state=42)

    #This code block controls the class count of the training and validation datasets. 
    #If the number of classes is different from the expected number of classes, it gives a warning message.
    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(len(set(label_val)), params['N_CLASSES']))
    #In this part, he creates the data generators, defines the model and related callback functions, 
    #and performs the training of the model. Model training is performed on the training data generator (tg) 
    #and validation is performed on the validation data generator (vg).
    tg = DataGenerator(wav_train, label_train, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    model = models[model_type]
    cp = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(tg, validation_data=vg,
              epochs=30, verbose=1,
              callbacks=[csv_logger, cp])
#In this section, when the program is run directly, it processes command line arguments and calls the train() 
#function. Arguments include parameters such as model type, root directory of the dataset, batch size, 
#sampling time, and sampling rate.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='lstm',
                        help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--src_root', type=str, default='clean',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000,
                        help='sample rate of clean audio')
    args, _ = parser.parse_known_args()

    train(args)
