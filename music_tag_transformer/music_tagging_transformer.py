import json
import yaml
from pathlib import Path
import save_utils

import numpy as np

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Input,
    GlobalAvgPool1D,
    Dense,
    Dropout,
)
from keras.models import Model
from keras.optimizers import Adam

from transformer import Encoder

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops

def custom_binary_accuracy(y_true, y_pred, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_true.dtype)

    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


def custom_binary_crossentropy(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    epsilon_ = K._constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    output = clip_ops.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = 4 * y_true * math_ops.log(output + K.epsilon())
    bce += (1 - y_true) * math_ops.log(1 - output + K.epsilon())
    return K.sum(-bce, axis=-1)


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
    # read in the config
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

    mel_train, mel_test_val, lab_train, lab_test_val = train_test_split(melspec_data, labels, train_size=config['train_set_size'], random_state=config['random_state'])
    mel_val, mel_test, lab_val, lab_test             = train_test_split(mel_test_val, lab_test_val, test_size=(config['val_set_size']/(1-config['train_set_size'])), shuffle=False)

    # Check the shapes of the splitted sets
    assert mel_train.shape[0] == lab_train.shape[0] and mel_test.shape[0] == lab_test.shape[0] and mel_val.shape[0] == lab_val.shape[0]
    assert mel_train.shape[1] == mel_test.shape[1] == mel_val.shape[1] and lab_train.shape[1] == lab_test.shape[1] == lab_val.shape[1]

    model = transformer_model(config['model_structure'], n_classes=len(labels_to_id))

    # load pretrained model
    # if transformer_pretrained_name:
    #    model.load_weights(transformer_pretrained_name, by_name=True)

    train_config = config['training']

    checkpoint = ModelCheckpoint(
        transformer_name,
        monitor=train_config['monitor'],
        verbose=1,
        save_best_only=train_config['save_best_weights'],
        mode=train_config['monitor_mode'],
        save_weights_only=False
    )

    # Reduce learning rate when val_loss stopps improving
    lr_reduce_config = train_config['lr_reducing']
    lr_reducing_on_platteau = ReduceLROnPlateau(
        monitor=lr_reduce_config['monitor'], patience=lr_reduce_config['patience'], min_lr=lr_reduce_config['min_lr'], mode=lr_reduce_config['mode']
    )

    model.fit(
        x=mel_train,
        y=lab_train,
        validation_data=(mel_val, lab_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, lr_reducing_on_platteau],
        use_multiprocessing=True,
        verbose=2
    )
