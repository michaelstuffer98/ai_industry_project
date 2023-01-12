from pydub import AudioSegment as AS
from pydub.playback import play
import os
from pathlib import Path

######################################################################################################
# Small script for the initlian conversion from mp3 to wav data, since python bindings work better 
# using .wav file format
######################################################################################################

genre_folders = ["06 Deep House", "09 Disco Funk", "12 Chinese Traditional", "15 Reggaeton Pop", "Keyboard Collection"]

wav_data_dir_name = "wav_data"

if not os.path.exists(wav_data_dir_name):
    os.mkdir(wav_data_dir_name)

for genre_path in genre_folders:
    genre_wav_dir = Path(wav_data_dir_name) / genre_path
    os.mkdir(genre_wav_dir)

    for filename in os.listdir("mp3_data/" + genre_path):
        f = genre_wav_dir / filename
        if os.path.isfile(f) and filename.endswith(".mp3"):
            sound = AS.from_mp3(f)
            sound.export(genre_wav_dir / str(filename).rstrip(".mp3") + ".wav", format="wav")
