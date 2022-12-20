import matplotlib.pyplot as plt
import librosa.display as lrd
import numpy as np
import librosa
import seaborn as sn

def plot_waveform(amp, sr):
    """
    Plot a waveform of the given amplitudes
        amp: amplitudes as a numpy array
        sr: sample rate of the audio
    """
    plt.figure(figsize=(14, 5))
    lrd.waveshow(amp, sr=sr)

def plot_melspectogram(S, sr, fmax, save_fig=True):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    if save_fig:
        fig.savefig('Plots/melfrequency_spectogram with fmax{f}.jpg'.format(f=int(fmax)))

def plot_hist(history, keys, legends, title, y_label, x_label, save_to=None):
    max_size = 0
    fig, ax = plt.subplots()

    ax.set_facecolor('white')

    for key in keys:
        if max_size < len(history[key]):
            max_size = len(history[key])
        ax.plot(history[key])

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.grid(b=True, alpha=1.0, color='black')
    ax.legend(legends, loc='upper left')

    if not save_to is None:
        fig.savefig(save_to)
    
    plt.show()

def plot_conf_mat(conf_mat_df, save_to=None):
    plt.figure(figsize = (7,6))
    sn.set(font_scale=0.9)
    sn.heatmap(conf_mat_df, annot=True, cmap='Blues', annot_kws={"size": 12})
    plt.tight_layout()
    if not save_to is None:
        plt.savefig(save_to)
    plt.show()