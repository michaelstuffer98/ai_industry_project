from pydub import AudioSegment
import librosa as lr
import soundfile as sf
import numpy as np

"""
Introduces some methods to perform preprocessing on audio data
"""

def slice_to_length(audiofile, sr, length_ms, pad=True):
    length_index = int(sr*length_ms/1000)
    if pad and audiofile.shape[0] < length_index:
        return np.concatenate((audiofile, np.zeros((length_index - audiofile.shape[0], 2), dtype=np.float32)), axis=0)
    return audiofile[0:length_index, :]

def to_mono_channel(audiofile):
    return lr.to_mono(audiofile.T).T

def resample(audiofile, old_sample_rate, new_sample_rate):
    return lr.resample(audiofile.T, orig_sr=old_sample_rate, target_sr=new_sample_rate).T

def channels(audiofile):
    return 1 if len(audiofile.shape) <= 1 else audiofile.shape[1]

def duration(audiofile, sample_rate):
    return audiofile.shape[0] / sample_rate

def test():
    audio1, sr1 = sf.read('wav_data/06 Deep House/Bam Bam Beat.wav', dtype='float32')
    audio2, sr2 = sf.read('wav_data/06 Deep House/Underground States Chord Layers 02.wav', dtype='float32')

    print(audio1.shape[0]/sr1)
    audio1 = slice_to_length(audio1, sr1, 5000)
    print(audio1.shape[0]/sr1)

    print(audio2.shape[0]/sr2)
    audio2 = slice_to_length(audio2, sr2, 5000)
    print(audio2.shape[0]/sr2)

    assert audio1.shape == audio2.shape