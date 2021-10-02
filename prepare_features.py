import librosa
from librosa.feature import melspectrogram
from librosa import power_to_db,amplitude_to_db,stft
from tqdm import tqdm
import soundfile as sf
import os
import numpy as np
from PIL import Image


### Run this script to generate mel spectrograms for the dataset
## Make sure the given data is located in 'path'
## Make sure to create a folder named "MEL" in the data directory,
## with subdirectories 'breathing','cough' and 'speech'


feat = 'MEL'
path = 'IEEE_HealthCareSummit_Dev_Data_Release'
for d in ['breathing','cough','speech']:
	print(d)
	for f in tqdm(os.listdir(path+'/AUDIO/'+d)):
		audio,sr = sf.read(os.path.join(path,'AUDIO',d,f))
		if feat=='MEL':
			S = melspectrogram(audio,sr)
			S_dB = power_to_db(S, ref=np.max)

		elif feat=='STFT':
			S = stft(audio,n_fft = 256,hop_length=256)
			S_dB = amplitude_to_db(S,ref=np.max)
			S_dB = S_dB[:128]

		S_dB = (S_dB+80)/80*255

		im = Image.fromarray(S_dB).convert('L')
		im.save(path+'/'+feat+'/'+d+'/'+f[:-5]+'.png')


