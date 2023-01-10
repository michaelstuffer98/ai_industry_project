import yaml
import numpy as np

import save_utils


if __name__ == "__main__":
    with open('music_tag_transformer/transformer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    transformer_name = config['transformer_name']
    transformer_pretrained_name = config['pretrained_transformer']
    batch_size = config['batch_size']
    epochs = config['epochs']

    melspec_data = save_utils.load_sliced_numpy_array('melspec_features', data_dir='data')
    labels = np.load('data/labels.npy')

