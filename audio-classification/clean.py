import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio
from librosa import resample
#In this section, it is seen that the necessary libraries and modules are imported. 
#This library and modules are used for processing and sampling audio files.


#This function is used to envelop an audio signal according to the given threshold value. 
# Using a rolling window on the audio signal, it calculates the maximum value for each window and compares it 
# with the threshold value. Values that exceed the threshold value are added to the masking list.
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean



#This function converts a given audio file to low resolution audio by changing the sampling rate. 
# First, the audio file is read and the original sampling rate is retrieved. Then the channels of the audio file 
# are checked and merged into a single channel. The sampling rate is converted to the target sampling rate (sr) by 
# the resampling process. Finally, the audio data is converted to the np.int16 data type, returning the processed 
# audio and sample rate.
def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc
    wav = resample(wav, orig_sr=rate, target_sr=sr)  # Güncelleme burada
    wav = wav.astype(np.int16)
    return sr, wav



# This function saves the sound samples obtained as a result of the sampling operation in the target directory. 
# It takes parameters sample data (sample), sample rate (rate), target directory (target_dir), filename (fn), and 
# an index (ix). The new file name is created by removing the ".wav" extension from the file name. The target file
# path is reached using the target directory and the new filename. If the target file already exists, 
# no action is taken. Otherwise, the audio sample is recorded using the wavfile.write() function.
def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)

# This function checks if a given directory exists. If the directory does not exist, a new directory is created.
def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)



# In this section, root directories are checked and target directories are created. First, the target root directory (dst_root) is checked and created if it does not exist. Next, the list of classes in the source root directory (src_root) is retrieved. The target directory is created for each class, and if it does not exist, it is created. Next, the list of files in src_dir is retrieved and the following operations are performed for each file:
# Get the full path of the file (src_fn).
# The audio file is downsampled (downsample_mono) to make it low resolution. Sample rate (rate) and processed audio data (wav) are obtained.
# A sequence (y_mean) is obtained from which masking and averages are calculated to create an envelope on the audio signal.
# By using masking, only sound samples that exceed the envelope are taken (wav).
# The delta sample count (delta_sample) is calculated.
# If the size of the processed audio is smaller than the delta sampling number, the audio sample size is padded with zeros to match the delta sampling number and saved in the target directory.
# Otherwise, it splits the processed audio sample by the number of delta samples and saves each segment in the target directory.
def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


#This function finds the audio file for a given sub-string and performs the envelope creation. 
# First, a list with the path of the audio files is created using the source root directory (src_root). 
# Then it finds the path (wav_path) of the audio file containing the particular substring. 
# If the number of files found is not 1, an error message is given and the function is terminated. 
# If the file is found, the audio file is downsampled (downsample_mono) to make it low resolution. 
# To create an envelope on the audio signal, an array (env) is obtained from which the masking and averages are calculated. 
# Finally, using the matplotlib library, a plot is made and displayed on the audio signal, envelope 
# and threshold.
def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    rate, wav = downsample_mono(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


#This section represents the beginning of the main code. Command line arguments are analyzed using the argument 
# parser (argparse). The src_root and dst_root arguments specify the source and destination directories. 
# The delta_time argument determines the time to split audio files. The sr argument determines the sampling rate 
# of the audio files.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='wavfiles',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='clean',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')

    parser.add_argument('--fn', type=str, default='3a3d0279',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    test_threshold(args)
    # split_wavs(args)
