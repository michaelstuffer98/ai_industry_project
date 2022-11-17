import matplotlib.pyplot as plt
import librosa.display as lrd

def plot_waveform(amp, sr):
    plt.figure(figsize=(14, 5))
    lrd.waveshow(amp, sr=sr)