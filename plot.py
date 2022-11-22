import matplotlib.pyplot as plt
import librosa.display as lrd
import numpy as np
import librosa
def plot_waveform(amp, sr):
    """
    Plot a waveform of the given amplitudes
        amp: amplitudes as a numpy array
        sr: sample rate of the audio
    """
    plt.figure(figsize=(14, 5))
    lrd.waveshow(amp, sr=sr)

def plot_melspectogram(S, sr, fmax):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    fig.savefig('Plots/melfrequency_spectogram with fmax{}'.format(fmax))