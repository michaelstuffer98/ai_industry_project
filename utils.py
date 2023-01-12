"""
save and load numpy array
split them into multiple files to keep the file size below 100MB,
since GitHub does not support files >100MB
"""

import numpy as np
from pathlib import Path
import os
import re
import yaml
import pickle
from keras.models import load_model
import json


def save_numpy_array_sliced(filename, data, data_dir='data', max_size_mb=90):
    assert data.ndim == 2
    
    data_dir = Path(data_dir)

    max_size_bytes = max_size_mb*1024*1024
    size_of_row = data[0, :].nbytes
    max_rows = int(max_size_bytes/size_of_row)
    
    rows_processed = 0
    rows_total = data.shape[0]
    
    n_iter = 0

    print("Saved file to slices", filename)

    while rows_processed < rows_total:
        n_iter += 1
        n_rows = min(max_rows, data.shape[0])
        rows_processed += n_rows
        f_name = f'{filename}_{n_iter:03d}'
        print("   saved slice {n:03d} to ".format(n=n_iter), f_name)
        np.save(data_dir / f_name, data[:n_rows, :])
        data = data[n_rows:, :]

    if rows_total != rows_processed:
        raise IOError("Failed writing all rows of array to file")

def load_sliced_numpy_array(filename, data_dir = 'data'):
    """
    Load the slices back, not guaranteed to be in the order as the slices have been saved
    """
    data_dir = Path(data_dir)

    data = None
    regex_pattern = filename + '_[0-9]*.npy'

    files_loaded = []

    for file in filter(lambda d: os.path.isfile(data_dir/d), os.listdir(data_dir)):
        if re.match(regex_pattern, file):
            if data is None:
                data = np.load(data_dir/file)
            else:
                data = np.vstack([data, np.load(data_dir/file)])
            files_loaded.append(file)
    
    print(f"Loaded {len(files_loaded)} files:")
    [print("   ", file) for file in files_loaded]

    return data


def save_training(to_dump: dict, name:str):
    model = to_dump['model']
    model.save_weights(f'models/{name}_trained_weights')
    model.save(f'models/{name}_trained')

    with open(f'models/{name}_history', 'wb') as f:
        pickle.dump(to_dump['history'].history, f)

    with open(f'models/{name}_config.yaml', 'w') as f:
        yaml.dump(to_dump['config'], f, default_flow_style=False)


def load_history(name):
    with open(f'models/{name}_history', 'rb') as f:
        return pickle.load(f)

def load_model(name):
    return load_model(f'model/{name}_trained')


def get_class_names():
    with open('data/class_label_index_mapping.json', 'r') as f:
        return json.load(f).keys()

def audio_file_iterator(mute=False):
    """
    Iteratable object to iterate over all wav-audio files in the 'wav_data' directory
        mute: if true, all output is suppressed
        returns: each audio file with the relative file path in the project
    """
    wav_dir = Path('wav_data')
    for directory in filter(lambda d: os.path.isdir(wav_dir/d), os.listdir(wav_dir)):
        genre_directory = wav_dir / directory
        if not mute:
            print('Processing genre directory \'{dir}\''.format(dir=directory))
        file_counter = 0
        for audio_file in filter(lambda f: os.path.isfile(genre_directory/f) and f.endswith('.wav'), os.listdir(genre_directory)):
            file_counter += 1
            if not mute:
                print("   Processed {n} files          ".format(n=file_counter), end='\r')
            yield genre_directory/audio_file
        if not mute:
            print('')