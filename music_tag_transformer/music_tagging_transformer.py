import json
import yaml
import numpy as np
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.layers import (
    Input,
    GlobalAvgPool1D,
    Dense,
    Dropout,
)
from keras.models import Model
from keras.optimizers import Adam

from transformer import Encoder
def transformer_model(model_config, n_classes):
    num_layers = model_config['n_layers']
    d_model = model_config['d_model']
    num_heads = model_config['n_heads']
    dff = model_config['dff']
    maximum_position_encoding = model_config['max_pos_encoding']
    init_lr = model_config['init_learning_rate']
    dropout_rate = model_config['dropout_rate']
    activations = model_config['activations']

    input_layer = Input((None, d_model))

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=model_config['encoder_rate']
    )

    x = encoder(input_layer)
    x = Dropout(dropout_rate)(x)
    x = GlobalAvgPool1D()(x)
    x = Dense(4 * n_classes, activation=activations[0])(x)

    out = Dense(n_classes, activation=activations[1])(x)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer=Adam(init_lr), loss=custom_binary_crossentropy, metrics=[custom_binary_accuracy])
    model.summary()
    return model

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


