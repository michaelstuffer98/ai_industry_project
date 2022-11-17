from pathlib import Path
import os

def iter(mute=False):
    """
    Iteratable object to iterate over all wav-audio files in the 'wav_data' directory
        mute: if true, all output is suppressed
        returns: each audio file with the relative file path in the project
    """
    wav_dir = Path('wav_data')
    for directory in filter(lambda d: os.path.isdir(wav_dir/d), os.listdir(wav_dir)):
        genre_directory = wav_dir / directory
        if not mute:
            print('Processing genre directory \'{dir}\''.format(dir=directory))
        file_counter = 0
        for audio_file in filter(lambda f: os.path.isfile(genre_directory/f) and f.endswith('.wav'), os.listdir(genre_directory)):
            file_counter += 1
            if not mute:
                print("   Processed {n} files          ".format(n=file_counter), end='\r')
            yield genre_directory/audio_file
        if not mute:
            print('')