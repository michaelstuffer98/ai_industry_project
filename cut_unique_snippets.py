from pydub import AudioSegment
import os
from pathlib import Path
from audio_file_iterator import iter as file_iterator

"""
DRAFT
Cuts all audio files in UNIT_LENGTH_AUDIO_MILLIS length snippets
"""

# Lenght of the snippets
UNIT_LENGTH_AUDIO_MILLIS = 3000

file_counter = 0

for filename in file_iterator():
    file_counter+=1

    audio_snippet = AudioSegment.from_wav(filename)[0:UNIT_LENGTH_AUDIO_MILLIS]
    
    # do something...
    audio_snippet.export("test{n}.wav".format(n=file_counter), format="wav")

    # quit after 3 files
    if file_counter >= 3:
        exit(0)