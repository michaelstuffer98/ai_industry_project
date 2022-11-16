from pydub import AudioSegment
import librosa as lr
import soundfile as sf

"""
Introduces some methods to perform preprocessing on audio data
"""

# Standard lenght of the snippets
UNIT_LENGTH_AUDIO_MILLIS = 3000


def test():
    path = 'wav_data/06 Deep House/'
    # 7 second sample
    filename = 'Bang Bang Bass 02.wav'
    u_length = to_unit_length(path + filename, export_to='test_unit_length.wav')
    mono_channel = to_mono_channel(path + filename, export_to='test_mono_channel.wav')
    resample(path + filename, 8000, export_to='test_downsample_8000.wav')

    assert len(u_length) == UNIT_LENGTH_AUDIO_MILLIS
    assert mono_channel.channels == 1


def to_unit_length(filename, export_to = None, length = UNIT_LENGTH_AUDIO_MILLIS):
    audio_snippet = AudioSegment.from_wav(filename)[0:UNIT_LENGTH_AUDIO_MILLIS]
    if not export_to is None: 
        audio_snippet.export(export_to, format="wav")
    return audio_snippet

def to_mono_channel(filename, export_to = None):
    audio = AudioSegment.from_wav(filename)
    audio = audio.set_channels(1)
    if not export_to is None: 
        audio.export(export_to, format="wav")
    return audio

def resample(filename, new_sampling_rate, export_to = None):
    data, sample_rate_original = sf.read(filename, dtype='float32')
    data = lr.resample(data.T, orig_sr=sample_rate_original, target_sr=new_sampling_rate)
    if not export_to is None:
        sf.write(export_to, data.T, new_sampling_rate)
    return data


if __name__ == '__main__':
    test()
