from keras.layers import Dense, Activation, Add, Flatten, Input, Conv2D,\
    AveragePooling2D, Concatenate, MaxPool2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import keras
import numpy as np
import random
from PIL import Image
from numpy import asarray
import pandas as pd
import os
from sklearn.metrics import roc_auc_score


## Where is the data located
path = 'IEEE_HealthCareSummit_Dev_Data_Release'
## Which fold to use for training
fold = '0'
## How many timesteps of Mel does the NN see
_SEQ_LEN = 128
## What data to train on
_DATA = 'speech'

#Load metadata
metadata = pd.read_csv(os.path.join(path, 'metadata.csv'),sep=' ')

#Load training IDs
with open(os.path.join(path, 'LISTS', 'train_'+fold+'.csv'),'r') as f:
    d = f.read()
train_ids = d.split('\n')[:-1]

#Load validation IDs
with open(os.path.join(path, 'LISTS', 'val_'+fold+'.csv'),'r') as f:
    d = f.read()
val_ids = d.split('\n')[:-1]


## Create a dictionary from sub_id to label
labels = dict()
for i in range(metadata.shape[0]):
    lab = metadata['COVID_STATUS'][i]
    if lab=='p':
        y = 1.0
    elif lab=='n':
        y = 0.0
    labels[metadata['SUB_ID'][i]] = y


positives_t = list()
negatives_t = list()


print(str(len(train_ids))+' total train ids')
for i in train_ids:
    #row = metadata[metadata['SUB_ID']==i].reset_index(drop=True)
    #print(row)
    if labels[i]>0.1:
        positives_t.append(i)
    else:
        #print('2')
        negatives_t.append(i)
    
print(str(len(positives_t))+' positives')
print(str(len(negatives_t))+' negatives')

def random_transform(x,height_range=10,width_range=100):
    for i in range(x.shape[0]):
        hshft = np.random.randint(-height_range,height_range)
        #print(shft)
        if hshft<0:
            x[i,:hshft] = x[i,-hshft:] 
        elif hshft>0:
            x[i,hshft:] = x[i,:-hshft]

        wshft = np.random.randint(-width_range,width_range)
        if wshft<0:
            x[i,:,:wshft]=x[i,:,-wshft:]
        elif wshft>0:
            x[i,:,wshft:] = x[i,:,:-wshft]
        

        coin = np.random.randint(2)
        if coin>0:
            x[i] = x[i]+x[i]*np.random.uniform(low=-0.2, high=0.2, size=x[i].shape)

    return x


def train_dg(bs=4,seq_len=256,data='breathing',balanced=False,augment=True):
    
    while True:
        ## If balanced, then half the batch is positive, half is negative
        if balanced==True:
            pos_ids = list(np.random.choice(positives_t,size=bs//2))
            neg_ids = list(np.random.choice(negatives_t,size=bs//2))
            ids = pos_ids+neg_ids
            random.shuffle(ids)
        else:
            ids = np.random.choice(train_ids,size=bs)

        x_b = np.zeros((bs,128,seq_len,1))
        y_b = np.zeros((bs,))

        ctr=0
        for i in ids:
            pth = os.path.join(path,'MEL',data,i+'.png')
            img = asarray(Image.open(pth))/255.0
            
            #print(img.shape)
            #print(seq_len)
            if img.shape[-1]<seq_len+1:
                x_b[ctr,:,:img.shape[-1],0] = img
            else:
                strt = np.random.randint(0,img.shape[-1]-seq_len)
                x_b[ctr,:,:,0] = img[:,strt:strt+seq_len]
            y_b[ctr] = labels[i]
            ctr+=1
        if augment:
            x_b = random_transform(x_b)

        yield x_b,y_b

def create_validation_data(seq_len=256,data='breathing'):
    images = []
    labs = []
    for i in val_ids:
        pth = os.path.join(path,'MEL',data,i+'.png')
        img = asarray(Image.open(pth))/255.0
        images.append(img.reshape(img.shape+(1,)))
        labs.append(labels[i])
    return images,labs





class ValidationCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        #seq_len = _SEQ_LEN
        self.data = _DATA
        self.best_acc=0
        self.best_roc=0.0
        print('preparing validation data')
        val_images, val_labels = create_validation_data(seq_len=_SEQ_LEN,data=self.data)
        self.arrs = list()
        self.labs = list()
        for i in range(len(val_images)):
            segs = val_images[i].shape[-2]//_SEQ_LEN
            if segs<1:
                x = np.zeros((1,128,_SEQ_LEN,1))
                x[0,:,:val_images[i].shape[-2]]=val_images[i]
            else:
                x = np.zeros((segs,128,_SEQ_LEN,1))
                for j in range(segs):
                    x[j] = val_images[i][:,j*_SEQ_LEN:(j+1)*_SEQ_LEN]
            self.labs.append(val_labels[i])
            self.arrs.append(x)


    def on_epoch_end(self, epoch, logs=None):
        true_positives=0
        true_negatives=0
        false_positives=0
        false_negatives=0
        preds = []
        for i in range(len(self.arrs)):
            a = self.arrs[i]
            pred = self.model.predict(a)
            mean_pred = np.mean(pred,axis=0)
            preds.append(mean_pred)
            if mean_pred>0.5 and self.labs[i]>0.5:
                true_positives+=1
            elif mean_pred>0.5 and self.labs[i]<0.5:
                false_positives+=1
            elif mean_pred<0.5 and self.labs[i]<0.5:
                true_negatives+=1
            elif mean_pred<0.5 and self.labs[i]>0.5:
                false_negatives+=1

        acc = (true_positives+true_negatives)/(true_negatives+true_positives+false_negatives+false_positives)
        auc = roc_auc_score(self.labs,preds)
        if auc>self.best_roc:
            self.best_roc=auc
            self.model.save(self.data+'_'+str(_SEQ_LEN)+'_'+fold+'.h5')
        #if acc>self.best_acc:
        #    self.best_acc=acc
        #    self.model.save(self.data+'_'+str(_SEQ_LEN)+'.h5')
        print('True positives : '+str(true_positives))
        print('True negatives : '+str(true_negatives))
        print('False positives : '+str(false_positives))
        print('False negatives : '+str(false_negatives))
        print("Acc : "+str(acc))
        print("ROC_AUC : "+str(auc))



def create_model(seq_len=256,data='breathing'):
    inp = Input((128,seq_len,1))
    c = Conv2D(32,16,activation='relu')(inp)
    c = MaxPool2D(2)(c)
    c = Conv2D(64,16,activation='relu')(c)
    c = MaxPool2D(2)(c)
    c = Conv2D(64,16,activation='relu')(c)
    c = Flatten()(c)
    c = Dense(1,activation='sigmoid')(c)
    m = Model(inp,c,name=data)
    m.compile(optimizer=Adam(lr=10e-5),loss='binary_crossentropy',metrics=['acc',])
    m.summary()
    return m


tdg = train_dg(seq_len = _SEQ_LEN, data=_DATA,balanced=True)

vcb = ValidationCallback()
m = create_model(seq_len = _SEQ_LEN,data = _DATA)

m.fit_generator(tdg,steps_per_epoch = 1024,epochs=100,callbacks=[vcb,])
