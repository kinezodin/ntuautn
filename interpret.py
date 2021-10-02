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
import lime
from lime import lime_image

from skimage.segmentation import mark_boundaries


_SEQ_LEN = 128

## ID of subject to interpret predictions for
sub_id = 'VEZCLNIH'

## Where is the data located
path = 'IEEE_HealthCareSummit_Dev_Data_Release'

datas = ['breathing','speech','cough']


## Load the models
models = dict()
for k in datas:
    models[k] = load_model(k+'_'+str(_SEQ_LEN)+'.h5')

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




features = dict()
mels = dict()
for k in datas:
    audio,sr = sf.read(os.path.join(path,'AUDIO',k,sub_id+'.flac'))
    S = melspectrogram(audio,sr)
    S_dB = power_to_db(S, ref=np.max)
    S_dB = (S_dB+80)/80
    
    ## flip the spectrograms for visualization (bass frequencies on the bottom)
    mels[k] = np.flip(S_dB,axis=0)
    ## Check if spectrogram has length less than _SEQ_LEN
    if S_dB.shape[-1]<_SEQ_LEN:
        ## If yes, then pad it with zeros
        features[k] = np.zeros((1,128,_SEQ_LEN,1))
        features[k][0,:,:,0] = S_dB
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
    S_dB = mels[k]

    ## Plot the spectrogram
    plt.imshow(S_dB,zorder=1)
    plt.title(sub_id+' '+k)
    
    ## Overlay prediction probabilities
    x_es = [i for i in range(_SEQ_LEN//2,S_dB.shape[-1]-_SEQ_LEN//2)]
    y_es = (1-preds[k])*128
    plt.scatter(x=x_es, y=y_es,c='r',s=1,zorder=2)
    plt.plot(preds[k],zorder=2)
    plt.show()
    plt.clf()



## LIME expects 3 channel images, use this to convert to 1 channel
def lime_predict(img):
    new_im = np.mean(img,axis=-1)
    new_im = new_im.reshape(new_im.shape+(1,))
    return models['cough'].predict(new_im)


#### For lime: Get the largest prob segment from cough
## Otherwise, set max_seg to the segment you want to explain
max_seg = np.argmax(preds['cough'][:,0])
interesting_segment = mels['cough'][:,(max_seg):(max_seg+_SEQ_LEN)]

## Run LIME
explainer = lime_image.LimeImageExplainer()
print(interesting_segment.astype('double').shape)
explanation = explainer.explain_instance(interesting_segment.astype('double'), lime_predict, top_labels=1, hide_color=0, num_samples=1000)

## Show results
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()


