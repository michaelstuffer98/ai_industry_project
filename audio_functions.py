import librosa as lr
import soundfile as sf
import numpy as np

"""
Introduces some methods to perform preprocessing on audio data
"""

def slice_to_length(audiofile, sr, length_ms, pad=True):
    """
    slices the audiofile to the desired length from the beginning, discards the rest of the audio
        Parameters:
            audiofile: n-dimensional numpy array
            sr: sample rate of the provided audio
            length_ms: length of the sliced audio array
            pad: if True, audio is zero padded at the end if the provided audio is shorter than the length_ms
        returns: first length_ms milliseconds of the passed audiofile
    """
    length_index = int(sr*length_ms/1000)
    dims = audiofile.ndim
    if dims > 2:
        raise IOError("Undefined bahaviour for arrays with dimension>2")
    if pad and audiofile.shape[0] < length_index:
        zero_pad_dims = (length_index - audiofile.shape[0], dims) if dims == 2 else (length_index - audiofile.shape[0])
        return np.concatenate((audiofile, np.zeros(zero_pad_dims, dtype=np.float32)), axis=0)
    return audiofile[:length_index, :] if dims==2 else audiofile[:length_index]

def multi_slice_to_length(audiofile, sr, length_ms, pad=True):
    """
    slices the audiofile to multiple snippests in the desired length from the beginning on to the end
    E.g. returns 5 snippets in 3 seconds length when passing an 15 second audio file length_ms=3000
        Parameters:
            audiofile: n-dimensional numpy array
            sr: sample rate of the provided audio
            length_ms: length of the sliced audio array
            pad: if True, audio is zero padded at the end if the provided audio is shorter than the length_ms
        returns: n-audio-snippets with length_ms of length
    """
    length_audio = audiofile.shape[0]
    slice_index = 0
    slice_interval = int(sr*length_ms/1000)
    dims = audiofile.ndim

    slices = []
    while slice_index <= length_audio:
        slices.append(((slice_to_length(audiofile[slice_index:, :] if dims==2 else audiofile[slice_index:], sr, length_ms)), sr))
        slice_index += slice_interval

    return slices
    #return np.array(slices, dtype=np.float32)

def to_mono_channel(audiofile):
    """
    converts a audiofile with multiple channels to a single (mono) channel audio file
        Parameters:
            audiofile: n-dimensional numpy array
        returns: 1-dimensional numpy array
    """
    return lr.to_mono(audiofile.T).T

def resample(audiofile, old_sample_rate, new_sample_rate):
    """
    resamples the audiofile from old_sample_rate to the new_sample_rate
        Parameters:
            audiofile: n-dimensional numpy array
            old_sample_rate: sample rate of the provided audio file
            new_sample_rate: desired sampling rate
        returns: 1-dimensional numpy array with the new sample rate
    """
    return lr.resample(audiofile.T, orig_sr=old_sample_rate, target_sr=new_sample_rate).T

def channels(audiofile) -> int:
    """
    Get the number of channels of the audio file
        Parameters:
            audiofile: n-dimensional numpy array
        returns: nr of channels of the audiofile
    """
    return 1 if len(audiofile.shape) <= 1 else audiofile.shape[1]

def duration(audiofile, sample_rate) -> float:
    """
    gets the duration of the audio file in seconds
        Parameters:
            audiofile: n-dimensional numpy array
            sr: sampling_rate
        returns: duration of the audiofile in seconds
    """
    return audiofile.shape[0] / sample_rate

def mean_stddev(audiofile, absolut=True) -> tuple[float, float]:
    """
    returns mean and std-dev of the audiofile
        Parameters:
            audiofile: n-dimensional numpy array
            absolut: set True to use absolut values for mean calculation
        returns: tuple of mean and std-dev
    """
    axis = len(audiofile.shape) -1
    assert axis in [0, 1]
    return (np.mean(np.abs(audiofile), axis=axis), np.std(audiofile, axis=axis))

def min_max(audiofile, absolut=True) -> tuple[float, float] | float:
    """
    returns min and max value of the audiofile
        Parameters:
            audiofile: n-dimensional numpy array
            absolut:
        returns: tuple of min and max value if absolut is false, a single float if absolut is true
    """
    axis = len(audiofile.shape) -1
    assert axis in [0, 1]
    return np.max(np.abs(audiofile)) if absolut else (np.min(audiofile, axis=axis), np.max(audiofile, axis=axis))


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