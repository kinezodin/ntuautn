from keras.layers import Dense, Activation, Add, Flatten, Input, Conv2D,\
    AveragePooling2D, Concatenate, MaxPool2D, BatchNormalization
from keras.models import Model,load_model
from keras.optimizers import Adam
import numpy as np
import random
from librosa.feature import melspectrogram
from librosa import power_to_db
import soundfile as sf
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


## Number of time-steps the NN sees
_SEQ_LEN = 128

## Which fold to evaluate
fold = '0'


## Where is the data located
path = 'IEEE_HealthCareSummit_Dev_Data_Release'

datas = ['breathing','speech','cough']

## Load all three models for the specific fold
models = dict()
for k in datas:
    models[k] = load_model(k+'_'+str(_SEQ_LEN)+'_'+fold+'.h5')

## Create a dictionary from sub_id to label
metadata = pd.read_csv(os.path.join(path, 'metadata.csv'),sep=' ')
labels = dict()
for i in range(metadata.shape[0]):
    lab = metadata['COVID_STATUS'][i]
    if lab=='p':
        y = 1.0
    elif lab=='n':
        y = 0.0
    labels[metadata['SUB_ID'][i]] = y



#Load validation IDs for the given fold
with open(os.path.join(path, 'LISTS', 'val_'+fold+'.csv'),'r') as f:
    d = f.read()
val_ids = d.split('\n')[:-1]



preds_segs = []
ground_truth = []
for sub_id in tqdm(val_ids):
    ## Load audio files, get mel spectrograms and prepare neural net input
    ground_truth.append(labels[sub_id])
    features = dict()
    for k in datas:
        audio,sr = sf.read(os.path.join(path,'AUDIO',k,sub_id+'.flac'))
        S = melspectrogram(audio,sr)
        S_dB = power_to_db(S, ref=np.max)
        S_dB = (S_dB+80)/80

        ## Check if spectrogram has length less than _SEQ_LEN
        if S_dB.shape[-1]<_SEQ_LEN:
            ## If yes, then pad it with zeros
            features[k] = np.zeros((1,128,_SEQ_LEN,1))
            features[k][0,:,:S_dB.shape[-1],0] = S_dB
        else:
            ## Split mel into overlapping windows of length _SEQ_LEN
            features[k] = list()
            for i in range(S_dB.shape[-1]-_SEQ_LEN):
                segment = S_dB[:,i:(i+_SEQ_LEN)]
                features[k].append(segment.reshape(segment.shape+(1,)))
            features[k] = np.array(features[k])


        
    ## Make the predictions with the neural nets
    preds = dict()
    for k in datas:
        preds[k] = models[k].predict(features[k])
        preds[k] = np.mean(preds[k])


    preds_segs.append(np.mean([preds[k] for k in preds]))


print("ROC_AUC : "+str(roc_auc_score(ground_truth,preds_segs)))
