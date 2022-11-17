import librosa
import numpy as np
import matplotlib.pyplot as plt

log_melgram = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(raw_audio_data, sr=sr, n_fft=2048, hop_length=512,↪ power=2.0, n_mels=128)))
print(log_melgram.shape)

# plt.subplot(1, 2, 2)
# img = plt.imshow(np.flipud(log_melgram))
# plt.colorbar(img, format="%+2.f dB")
# plt.title('log(melspectrogram)'); plt.grid(False);plt.yticks([]);
# plt.yticks([0, 128], [str(sr // 2), '0']); plt.ylabel('[Hz]'); plt.xlabel('time␣
# ↪[index]');
