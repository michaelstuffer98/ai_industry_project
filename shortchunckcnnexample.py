import numpy as np
from keras.models import Sequential, Input,Model, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, MaxPool1D, GaussianNoise, GlobalMaxPooling1D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os

# Reshaping Mel-Spectrogram
def reshape_melspectogram(mel_train, mel_test):
    maximum = np.amax(mel_train)
    mel_train = mel_train/np.amax(maximum)
    mel_test = mel_test/np.amax(maximum)
    mel_train = mel_train.astype(np.float32)
    mel_test = mel_test.astype(np.float32)
    N, row, col = mel_train.shape
    mel_train = mel_train.reshape((N, row, col, 1))
    N, row, col = mel_test.shape
    mel_test = mel_test.reshape((N, row, col, 1))
    return mel_train, mel_test

# Save Mel-Spectrogram train-test
# def savemelspectogram(mel_train, mel_test, y_train, y_test):
    # np.savez_compressed(os.getcwd()+"/new_mel_train_test.npz", mel_train= mel_train, mel_test= mel_test, y_train = y_train, y_test= y_test)

# Mel-Sprectrogram

# Load npz file of Mel-Spectrogram
# file = np.load(os.getcwd()+"/new_mel_train_test.npz")
# mel_train = file['mel_train']
# mel_test = file['mel_test']
# y_train = file['y_train']
# y_test = file['y_test']

# Define the model
def shortchunckcnn(mel_train):
    model = Sequential()
    model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= mel_train[0].shape, padding= 'same'))
    model.add(MaxPooling2D((4,4), padding= 'same'))
    model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
    model.add(MaxPooling2D((4,4), padding= 'same'))
    model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
    model.add(MaxPooling2D((4,4), padding= 'same'))
    model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
    model.add(MaxPooling2D((4,4), padding= 'same'))
    model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
    model.add(MaxPooling2D((4,4), padding= 'same'))
    model.add(Flatten())
    model.add(Dense(64, activation= 'relu'))
    model.add(Dense(10, activation= 'softmax'))

    model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')

    model.summary()


# Train Model

checkpoint = ModelCheckpoint(os.getcwd()+"/models/ensemble_model_melspectrogram1_{epoch:03d}.h5", period= 5)

model.fit(mel_train, y_train, epochs= 200, callbacks= [checkpoint], batch_size= 32, verbose= 1)
model.save(os.getcwd() + "/models/ensemble_model_melspectrogram1.h5")

# Load the model
model = load_model(os.getcwd() + "/models/ensemble_model_melspectrogram1.h5")


# Training Accuracy
y_pred = model.predict(mel_train)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_train, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100

print("Train Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

# Testing Accuracy
y_pred = model.predict(mel_test)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_test, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100
print("Testing Accuracy", acc)

class_names = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
conf_mat = confusion_matrix(y_true, y_pred, normalize= 'true')
conf_mat = np.round(conf_mat, 2)

conf_mat_df = pd.DataFrame(conf_mat, columns= class_names, index= class_names)

plt.figure(figsize = (10,7), dpi = 200)
sn.set(font_scale=1.4)
sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16}) # font size
plt.tight_layout()
plt.savefig(os.getcwd() + "/ensemble_mel_conf_mat1.png")