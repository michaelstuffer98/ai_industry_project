import matplotlib.pyplot as plt
import librosa.display as lrd

def plot_waveform(amp, sr):
    """
    Plot a waveform of the given amplitudes
        amp: amplitudes as a numpy array
        sr: sample rate of the audio
    """
    plt.figure(figsize=(14, 5))
    lrd.waveshow(amp, sr=sr)