from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
# In this section, it is seen that the libraries and modules to be used are imported. 
# A model is loaded with tensorflow.keras, the downsample_mono and envelope functions are imported 
# from the clean module. Some classes and functions from the kapre.time_frequency module for audio processing, 
# LabelEncoder class from the sklearn.preprocessing module for classification, and other necessary modules are 
# also imported.



# This function is used to make an estimate. First, the specified model file (args.model_fn) is loaded. 
# The paths (wav_paths) of the audio files in the source directory specified by args.src_dir are obtained. 
# Then, the file paths are sorted and only those with the .wav extension are selected. Classes and labels are 
# obtained. Tags are derived from the directory names above the file paths. Next, labels are converted to 
# numeric values using LabelEncoder. A list (results) is created to keep the forecast results.
def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []


#This loop makes predictions for each audio file. First, the audio file sample is sampled according to the sample 
# rate (args.sr) with the downsample_mono function. Then, the envelope of the signal is calculated with the envelope 
# function. Enveloped cleaned sound (clean_wav) is obtained. Then, at a certain step (step), sampled audio tracks 
# are created on clean_wav and added to the batch list. These chunks, called x_batch, are converted to a numpy array.
# The prediction is made using the predict method of the model and the average of the prediction results (y_mean) is 
# calculated. The highest prediction class (y_pred) is determined and real class information is obtained. 
# The prediction results (y_mean) are added to the results list.
        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        results.append(y_mean)

    np.save(os.path.join('logs', args.pred_fn), np.array(results))



# In this section, an argument parser is created using argparse to parse the arguments. 
# The arguments entered from the command line are analyzed and the args object is created. 
# Then the make_prediction function is called and the args object is passed to this function.

# This piece of code enables loading a model and making predictions on specified audio files. 
# The --model_fn argument specifies the file path of the model to use for prediction. 
# The --pred_fn argument specifies the filename where the prediction results will be saved. The --src_dir 
# argument specifies the directory where the sound files to guess are in. The --dt argument specifies the time 
# to sample based on the sampling rate of the audio files. The --sr argument specifies the sample rate of the 
# cleaned audio. The --threshold argument specifies the threshold for the np.int16 data type.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_prediction(args)

