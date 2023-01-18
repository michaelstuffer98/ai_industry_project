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
import keras.models
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

    return n_iter


def load_sliced_numpy_array(filename, data_dir = 'data'):
    """
    Load the slices back, not guaranteed to be in the order as the slices have been saved
    """
    data_dir = Path(data_dir)
    regex_pattern = filename + '_[0-9]*.npy'

    file_names = []
    for file in filter(lambda d: os.path.isfile(data_dir/d), os.listdir(data_dir)):
        if re.match(regex_pattern, file):
            file_names.append(file)
    
    data = None
    for file in sorted(file_names):
        if data is None:
            data = np.load(data_dir/file)
        else:
            data = np.vstack([data, np.load(data_dir/file)])
        print(f"Load slice from file {file}")
    
    print(f"Loaded {len(file_names)} files as slices, resulting shape: {data.shape}")

    return data


def test_load_save_routine():
    # Check if array after saving and loading is still equal
    array_before = np.random.randint(0, 100, (10000, 500))

    n_iter = save_numpy_array_sliced('tmp_array', array_before, max_size_mb=1)
    array_after = load_sliced_numpy_array('tmp_array')

    assert np.abs(array_after - array_before).sum() == 0

    for i in range(n_iter):
        os.remove(f'data/tmp_array_{(i+1):03}.npy')


def save_training(to_dump: dict, name:str, dir='models'):
    dir = Path(dir)
    
    model = to_dump['model']
    model.save_weights(dir/f'{name}_trained_weights')
    model.save(dir/f'{name}_trained')

    with open(dir/f'{name}_history', 'wb') as f:
        pickle.dump(to_dump['history'], f)

    with open(dir/f'{name}_config.yaml', 'w') as f:
        yaml.dump(to_dump['config'], f, default_flow_style=False)


def load_history(name, dir='models'):
    dir = Path(dir)
    with open(dir/f'{name}_history', 'rb') as f:
        return pickle.load(f)

def load_model(name, dir='models'):
    dir = Path(dir)
    return keras.models.load_model(dir/f'{name}_trained')


def get_class_names():
    with open('data/class_label_index_mapping.json', 'r') as f:
        return json.load(f).keys()


def get_class_mapping():
    with open('data/class_label_index_mapping.json', 'r') as f:
        return json.load(f)


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


def get_config(name):
    # read in the config
    with open(f'configs/{name}_config.yaml', 'r') as f:
        return yaml.safe_load(f)