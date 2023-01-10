import json
import yaml
import numpy as np
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
import save_utils


if __name__ == "__main__":
    with open('music_tag_transformer/transformer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    transformer_name = config['transformer_name']
    transformer_pretrained_name = config['pretrained_transformer']
    batch_size = config['batch_size']
    epochs = config['epochs']
    data_dir = Path(config['data_dir'])

    melspec_data = save_utils.load_sliced_numpy_array('melspec_features', data_dir=data_dir)
    labels = np.load(data_dir/'labels.npy')

    with open(data_dir/'class_label_index_mapping.json', 'r') as f:
        labels_to_id = json.load(f)

    print(labels_to_id)

    mel_train, mel_val, lab_train, lab_val = train_test_split(melspec_data, labels, test_size=config['test_set_size'], random_state=config['random_state'])

    assert mel_train.shape[0] == lab_train.shape[0] and mel_val.shape[0] == lab_val.shape[0] and mel_val.shape[0] == lab_val.shape[0]

    exit(0)


