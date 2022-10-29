# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:15:24 2022

@author: nazmu
"""

#Here we implemented LSTM, Bi-LSTM, Conv2D, ConvLSTM2D models

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:50:27 2022

@author: nazmu
"""
"""
to do - first create reference dataset and do the filtering 
then train test split and to csv convert and run the algorithms 10 times each time
match with the reference labels and then finally make 10 csv file each 6-7-days
"""



import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#Importing required libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# sensitivity analysis of k in k-fold cross-validation
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from sklearn.metrics import confusion_matrix
import keras
from numpy import mean
import tensorflow as tf
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D, MaxPooling3D
#from keras.utils import to_categorical
from matplotlib import pyplot
from keras.layers import ConvLSTM2D,Conv2D,MaxPool2D
from tensorflow.keras import regularizers 
from keras.metrics import top_k_categorical_accuracy
def top5cat(y_test_ctg, y_predict):
  return top_k_categorical_accuracy(y_test_ctg, y_predict, k=5)

from numpy import array
import keras
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
#from keras.layers.merge import concatenate
from pickle import load

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

import csv, string
from nltk.collocations import *
import nltk.collocations
from nltk import ngrams
from collections import Counter

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPool1D
from keras.datasets import imdb
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
#from keras.layers.normalization import BatchNormalization
import re

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
#from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
#from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keras.utils.np_utils import to_categorical
import seaborn as sns
from keras.layers import Dense, GRU, Flatten, LSTM,GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from sklearn import preprocessing
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


#Input the original dataset
#ifile = pd.read_csv("D:/Research_work/exp4/0_min_6class_set4.csv")
ifile = pd.read_csv("D:/Plant Genomics Using ML/dataset/30_min_cry.csv") #Change it by using your own directory path

X_col = ifile['timestamp']
ifile = ifile.drop('timestamp',axis = 1)
#ifile = clean_dataset(ifile)
#sc = StandardScaler()
#sc = preprocessing.normalize()

#Dataset Preprocessing by MinMaxNormalization
ifile.fillna(199, inplace=True)
sc = MinMaxScaler(feature_range = (0,1))

i_tr = np.transpose(ifile)
i_norm = []
i = 0
# for i in range(len(ifile)):
#     i_norm.append(stats.zscore((i_tr[[i]]))) 
for i in range(len(ifile)):
    i_norm.append(sc.fit_transform(i_tr[[i]]))
    #i_norm.append(preprocessing.normalize(i_tr[[i]]))
X_norm_a = i_norm
X_norm_a = np.asarray(X_norm_a) 
X_norm_a= np.reshape(X_norm_a, (X_norm_a.shape[0],X_norm_a.shape[1]))
#X_norm_a= np.reshape(X_norm_a, (46,1432))
X_norm_a = pd.DataFrame(X_norm_a)   


#y = pd.read_csv("D:/Research_work/exp4/0_min_6_class_set4.csv")
y = pd.read_csv("D:/Plant Genomics Using ML/dataset/30_min_cry_class.csv")  #Change it by using your own directory path

y_r = y

le = LabelEncoder()
y_r = le.fit_transform(y_r)
y_p = pd.DataFrame(y_r)
y_p.to_csv("D:/Research_work/exp4/Dataset/cry_y.csv")

X_norm_a['timestamp'] = X_col

"""
for no shuffle train test split
"""
X_train, X_test, y_train, y_test = train_test_split(X_norm_a,y_r, test_size=0.20, stratify=y)
    #X_train, X_test, y_train, y_test = train_test_split(ifile,y_r, test_size=0.20, stratify=y)
    #print(X_train.shape, y_train.shape)
    #print (X_test.shape, y_test.shape)
X_train_col = X_train['timestamp']    
X_train = X_train.drop('timestamp', axis =1)
X_test_col = X_test['timestamp']
X_test = X_test.drop('timestamp', axis =1)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
#X_train = X_train.reset_index()
X_train.to_csv("D:/Research_work/exp4/Dataset/X_train_0_30_90_phot.csv")
X_test.to_csv("D:/Research_work/exp4/Dataset/X_test_0_30_90_phot.csv")
y_train.to_csv("D:/Research_work/exp4/Dataset/y_train_0_30_90_phot.csv")
y_test.to_csv("D:/Research_work/exp4/Dataset/y_test_0_30_90_phot.csv")
import csv
X_test_col.to_csv("D:/Research_work/Exp2/ResultZ1_3/X_test_z1_32_9_58_conv.csv")
X_train_col.to_csv("D:/Research_work/Exp2/ResultZ1_3/X_train_z1_9_58_conv.csv")

X_train = pd.read_csv("D:/Research_work/Exp2/Datasets/X_train_z1_4.csv")
X_test = pd.read_csv("D:/Research_work/Exp2/Datasets/X_test_z1_4.csv")
y_train = pd.read_csv("D:/Research_work/Exp2/Datasets/y_train_z1_4.csv")
y_test= pd.read_csv("D:/Research_work/Exp2/Datasets/y_test_z1_4.csv")


X_train_bck = X_train
X_test_bck = X_test
y_train_bck = y_train  #backup files
y_test_bck = y_test


X_train= np.asarray(X_train)
y_train=np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
    
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
    
    
y_train_ctg = to_categorical(y_train)
y_test_ctg = to_categorical(y_test)
y_test_ctg  = np.asarray(y_test_ctg).astype(np.int)
y_train_ctg  = np.asarray(y_train_ctg).astype(np.int)
    
X_train_2 = X_train
X_test_2 = X_test
y_train_2 = y_train_ctg
y_test_2 = y_test_ctg
    
X_train_3 = X_train
X_test_3 = X_test
y_train_3 = y_train_ctg
y_test_3 = y_test_ctg

#Reshaping Input array size    
X_train_31 = X_train_3.reshape(X_train_3.shape[0], 1, X_train_3.shape[1])
X_test_31 = X_test_3.reshape( X_test_3.shape[0],1, X_test_3.shape[1])
# X_train_3 = X_train_3.reshape(X_train_3.shape[0], 33,4)
# X_test_3 = X_test_3.reshape(X_test_3.shape[0], 33,4)

X_train_3 = X_train_3.reshape(X_train_3.shape[0], 47,3)  #z1
X_test_3 = X_test_3.reshape(X_test_3.shape[0], 47,3)

X_train_34 = X_train_3.reshape(X_train_3.shape[0], 47,3,1)  #z1
X_test_34 = X_test_3.reshape(X_test_3.shape[0], 47,3,1)

X_train_3 = X_train_3.reshape(X_train_3.shape[0], 48,3)
X_test_3 = X_test_3.reshape(X_test_3.shape[0], 48,3)
# X_train_3 = X_train_3.reshape(X_train_3.shape[0], 133,7)
# X_test_3 = X_test_3.reshape(X_test_3.shape[0], 133,7)


#Customized EarlyStopping method
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
my_callback = MyThresholdCallback(threshold=0.69)            



#Implementation of LSTM
model = Sequential()
model.add(LSTM(50, input_shape =(X_train.shape[1], X_train.shape[2])))
model.add(Dense(9))
model.compile(loss='mae', optimizer='adam')
model.add(LSTM(200, return_sequences = True, input_shape =(1, X_train.shape[2])))
model.add(LSTM(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5cat])
# kernel_regularizer =regularizers.l2(3e-2), recurrent_regularizer=regularizers.l2(3e-2)

#Implementation of Bi-Directional LSTM
model = Sequential()
model.add(Bidirectional(LSTM(250,return_sequences=True, return_state=False,input_shape=(1, X_train_3.shape[2]))))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(250, activation='relu')))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train_ctg.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_3, y_train_3, validation_data=(X_test_3, y_test_3), epochs=250, batch_size=32,callbacks=my_callback)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

X_train_2 = X_train_2.reshape(X_train_2.shape[0], 1,1,1, X_train_2.shape[1])
X_test_2 = X_test_2.reshape( X_test_2.shape[0],1,1,1, X_test_2.shape[1])
    
y_train_2 = y_train_2.reshape(y_train_2.shape[0],1,1, 1, y_train_2.shape[1])
y_test_2 = y_test_2.reshape( y_test_2.shape[0],1,1,1, y_test_2.shape[1])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
model = Sequential()
model.add(ConvLSTM2D(filters = 128, return_sequences=True, kernel_size = (1,1), activation="relu",input_shape =(1,1,1,X_train_2.shape[4])))
model.add(ConvLSTM2D(filters = 256, kernel_size = (1,1), activation="relu"))
model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
model.add(MaxPool2D(pool_size=(1,1)))
    #model.add(LSTM(256))
    #model.add(ConvLSTM2D(filters = 64, kernel_size = (1,1), activation="relu"))
    #model.add(MaxPool2D())
#model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
model.add(Dense(y_train_ctg.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
my_callback = MyThresholdCallback(threshold=0.66)  
history = model.fit(X_train_2,y_train_ctg, epochs=250, batch_size = 16, validation_data =(X_test_2,y_test_ctg), callbacks=my_callback)
    #ls_model.fit(X_train_3, y_train_ctg, epochs=150, batch_size =128, validation_data = (X_test_3, y_test_ctg))
model.evaluate(X_test_2,y_test_ctg, verbose=0, batch_size = 16)
y_predict = model.predict(X_test_2)
y_classes_test = [np.argmax(y1, axis=None, out=None) for y1 in y_test_ctg]
y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

    #y_classes_test

    #y_classes

y_predicts = le.inverse_transform(y_classes)

y_true = le.inverse_transform(y_classes_test)

y_predicts = pd.DataFrame(y_predicts)
y_true = pd.DataFrame(y_true)
y_predicts = pd.DataFrame(y_predicts)
y_true = pd.DataFrame(y_true)
#y_predicts = le.inverse_transform(y_classes)
y_classes2 = to_categorical(y_classes)

#y_true = le.inverse_transform(targets[test])
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_classes_test,y_classes)
acc = accuracy_score(y_classes_test, y_classes, normalize=False)
print(result)
print(acc)


#For Plotting the confusion matrix
cm_plot_labels = ['fkf1','lkp2','ztl']
cm_plot_labels = ['WT','phyA','phyB','phyC','phyD','phyE','phot1','phot2','phot1/2','cry1','cry2','cry3','cry1/2','ds-16','fkf1','lkp2','ztl']
cm_plot_labels = ['WT','phyC','phot1','phot2','phot1/2','ds-16']
cm_plot_labels = ['WT','phyE','phot1','phot2','phot1/2','ztl']
cm_plot_labels = ['fkf1_0min','ztl_0min','lkp2_0min','fkf1_30min','lkp2_30min','ztl_30min']
cm_plot_labels = ['phot2','phot1/2','phot1']
cm_plot_labels = ['phot1_0min','phot2_0min','phot1/2_0min','phot1_30min','phot2_30min','phot1/2_30min','phot1_90min','phot2_90min','phot1/2_90min']
cm_plot_labels = ['WT','phyC','cry1','cry2','ds-16','fkf1','ztl']
cm_plot_labels = ['cry1/2_0min','cry1/2_30min','cry1/2_90min','cry1_0min','cry1_30min','cry1_90min','cry2_0min','cry2_30min','cry2_90min']
cm_plot_labels = ['phyA','phyB','phyC','phyE','phyD']
cm_plot_labels = ['cry1','cry1/2','cry2']
my_callback = MyThresholdCallback(threshold=0.68) 

#Implementation of Basic Conv1D model

def basic_conv1D(n_filters=128, fsize=5, window_size=5, n_features=3):
 new_model = keras.Sequential()
 new_model.add(tf.keras.layers.Conv1D(128, fsize, padding="same", activation="relu", input_shape=(window_size, n_features)))
 #new_model.add(tf.keras.layers.GlobalAveragePooling1D())
 new_model.add(tf.keras.layers.Conv1D(256, fsize, padding="same", activation="relu"))
 # Flatten will take our convolution filters and lay them out end to end so our dense layer can predict based on the outcomes of each
 new_model.add(tf.keras.layers.Flatten())

 new_model.add(tf.keras.layers.Dense(180, activation="relu"))

 new_model.add(tf.keras.layers.Dense(100))
 new_model.add(tf.keras.layers.Dense(y_train_ctg.shape[1]))
 new_model.compile(optimizer="adam", loss="mean_squared_error", metrics = ['accuracy']) 
 return new_model

univar_model = basic_conv1D(n_filters=64, fsize=8, window_size=X_train_3.shape[1], n_features=X_train_3.shape[2])

univar_model.fit(X_train_3, y_train_ctg, epochs=300, batch_size = 16, validation_data = (X_test_3, y_test_ctg),callbacks=my_callback)
scores = univar_model.evaluate(X_test_3, y_test_ctg, batch_size=16,verbose=0)
y_predict = univar_model.predict(X_test_3)
y_classes_test = [np.argmax(y, axis=None, out=None) for y1 in y_test]
y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

#y_classes_test

#y_classes

y_predicts = le.inverse_transform(y_classes)

y_true = le.inverse_transform(y_test)
#y_true

#y_predicts

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_classes)
acc = accuracy_score(y_test, y_classes, normalize=False)
print(result)
print(acc)

cm = confusion_matrix(y_test,y_classes)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=cm_plot_labels, yticklabels=cm_plot_labels, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)    

#Implementation of Conv2D model
def basic_conv2D(n_filters=128, fsize=5, window_size=5, n_features=3):
 new_model = keras.Sequential()
 new_model.add(tf.keras.layers.Conv2D(64, fsize, padding="same", activation="relu", input_shape=(window_size, n_features,1)))
 #new_model.add(tf.keras.layers.GlobalAveragePooling1D())
 new_model.add(tf.keras.layers.Conv2D(72, fsize, padding="same", activation="relu"))
 # Flatten will take our convolution filters and lay them out end to end so our dense layer can predict based on the outcomes of each
 new_model.add(tf.keras.layers.Flatten())

 new_model.add(tf.keras.layers.Dense(180, activation="relu"))

 new_model.add(tf.keras.layers.Dense(100))
 new_model.add(tf.keras.layers.Dense(y_train_ctg.shape[1]))
 new_model.compile(optimizer="adam", loss="mean_squared_error", metrics = ['accuracy']) 
 return new_model

univar_model = basic_conv1D(n_filters=64, fsize=8, window_size=X_train_3.shape[1], n_features=X_train_3.shape[2])

univar_model.fit(X_train_34, y_train_ctg, epochs=300, batch_size = 16, validation_data = (X_test_3, y_test_ctg),callbacks=my_callback)
scores = univar_model.evaluate(X_test_34, y_test_ctg, batch_size=16,verbose=0)
y_predict = univar_model.predict(X_test_3)
y_classes_test = [np.argmax(y, axis=None, out=None) for y1 in y_test]
y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

#y_classes_test

#y_classes

y_predicts = le.inverse_transform(y_classes)

y_true = le.inverse_transform(y_test)
#y_true

#y_predicts

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_classes)
acc = accuracy_score(y_test, y_classes, normalize=False)
print(result)
print(acc)

cm = confusion_matrix(y_test,y_classes)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=cm_plot_labels, yticklabels=cm_plot_labels, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)    


#Basic LSTM Model without 10-fold
    
def basic_LSTM(window_size=5, n_features=5):
 new_model = keras.Sequential()
 new_model.add(LSTM(256, input_shape=(window_size, n_features),return_sequences=True,activation="relu"))
 new_model.add(Dropout(0.5))
 new_model.add(LSTM(256,activation="relu",return_sequences=True))
 new_model.add(Dropout(0.5))
 new_model.add(LSTM(256,activation="relu",return_sequences=True))
 new_model.add(Dropout(0.5))
 new_model.add(LSTM(256,activation="relu",return_sequences=False))
 new_model.add(Dropout(0.5))
 #new_model.add(Flatten())
 new_model.add(Dense(150, activation="relu"))
 #new_model.add(Dense(100, activation="linear"))
 new_model.add(Dense(y_test_ctg.shape[1],activation='softmax'))
 new_model.compile(optimizer="adam", loss="mean_squared_error", metrics = ['accuracy']) 
 return new_model
ls_model = basic_LSTM(window_size=X_train_3.shape[1], n_features=X_train_3.shape[2])
ls_model.summary()
ls_model.fit(X_train_3, y_train_ctg, epochs=150, batch_size =16, validation_data = (X_test_3, y_test_ctg),callbacks=my_callback)
ls_model.evaluate(X_test_3, y_test_ctg, verbose=0)

y_predict = ls_model.predict(X_test_3)
y_classes_test = [np.argmax(y, axis=None, out=None) for y1 in y_test]
y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

#y_classes_test

#y_classes

y_predicts = le.inverse_transform(y_classes)

y_true = le.inverse_transform(y_test)

y_predicts = pd.DataFrame(y_predicts)
y_true = pd.DataFrame(y_true) 


#Implementation of ConvLSTM2D model   

X_train_2 = X_train_2.reshape(X_train_2.shape[0], 1,1,1, X_train_2.shape[1])
X_test_2 = X_test_2.reshape( X_test_2.shape[0],1,1,1, X_test_2.shape[1])
    
y_train_2 = y_train_2.reshape(y_train_2.shape[0],1,1, 1, y_train_2.shape[1])
y_test_2 = y_test_2.reshape( y_test_2.shape[0],1,1,1, y_test_2.shape[1])
    
model = Sequential()
model.add(ConvLSTM2D(filters = 128, return_sequences=True, kernel_size = (1,1),padding = "same", activation="relu",input_shape =(1,1,1,X_train_2.shape[4])))
model.add(ConvLSTM2D(filters = 128, kernel_size = (1,1), activation="relu"))
model.add(BatchNormalization(epsilon=1e-06,momentum=0.9, weights=None))
model.add(MaxPool2D(pool_size=(1,1)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
    #model.add(ConvLSTM2D(filters = 64, kernel_size = (1,1), activation="relu"))
    #model.add(MaxPool2D(pool_size=(1,1)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))   
model.add(Dense(y_train_ctg.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
history = model.fit(X_train_2, y_train_ctg, epochs=300, batch_size=16, validation_data = (X_test_2, y_test_ctg))
scores = model.evaluate(X_test_2, y_test_ctg, verbose=0)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
    
y_predict = model.predict(X_test_2)
y_classes_test = [np.argmax(y, axis=None, out=None) for y1 in y_test]
y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]
    
    #y_classes_test
    
    #y_classes
    
y_predicts = le.inverse_transform(y_classes)
    
y_true = le.inverse_transform(y_test)

y_predicts = pd.DataFrame(y_predicts)
y_true = pd.DataFrame(y_true)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
result = classification_report(y_test,y_classes)
acc = accuracy_score(y_test, y_classes, normalize=False)
print(result)
print(acc)    


cm = confusion_matrix(y_test,y_classes)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=cm_plot_labels, yticklabels=cm_plot_labels, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)    

"""
#Implementation of 10 fold for each models

"""

X_train_21 = X_train_3.reshape(X_train_3.shape[0],1,1,3,47)
X_test_21 = X_test_3.reshape(X_test_3.shape[0],1,1,3,47)

inputs = np.concatenate((X_train_2, X_test_2), axis=0)
targets = np.concatenate((y_train_ctg, y_test_ctg), axis=0)

inputs = np.concatenate((X_train_3, X_test_3), axis=0)
targets = np.concatenate((y_train_ctg, y_test_ctg), axis=0)

# Define the K-fold Cross Validator
from keras.callbacks import EarlyStopping
earlystop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=.3,
    patience=50,
    #baseline = 0.38,
    verbose=1,
    mode="min",
    #restore_best_weights=True,
)
my_callback = MyThresholdCallback(threshold=0.62)  
k = 10
kfold = KFold(n_splits=k, shuffle=False)
acc_per_fold = []
loss_per_fold = []
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
   #  model = Sequential()
   #  model.add(ConvLSTM2D(filters = 128, return_sequences=True, kernel_size = (1,1), activation="relu",input_shape =(1,1,1,X_train_2.shape[4])))
   #  model.add(ConvLSTM2D(filters = 256, kernel_size = (1,1), activation="relu"))
   #  model.add(MaxPool2D(pool_size=(1,1)))
   #  #model.add(LSTM(256))
   #  #model.add(ConvLSTM2D(filters = 64, kernel_size = (1,1), activation="relu"))
   #  #model.add(MaxPool2D())
   #  model.add(Flatten())
   #  model.add(Dense(128, activation='relu'))
   #  model.add(Dropout(0.2))
   #  #model.add(Dense(64, activation='relu'))
   #  #model.add(Dropout(0.5))
   #  model.add(Dense(y_train_ctg.shape[1], activation='softmax'))
   #  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    


  # Define the model architecture
    def basic_conv1D(n_filters=10, fsize=5, window_size=5, n_features=4):
      new_model = keras.Sequential()
      new_model.add(tf.keras.layers.Conv1D(64, fsize, padding="same", activation="relu", input_shape=(window_size, n_features)))
      #new_model.add(tf.keras.layers.GlobalAveragePooling1D())
      new_model.add(tf.keras.layers.Conv1D(128, fsize, padding="same", activation="relu"))
      # Flatten will take our convolution filters and lay them out end to end so our dense layer can predict based on the outcomes of each
      new_model.add(tf.keras.layers.Flatten())
    
      new_model.add(tf.keras.layers.Dense(180, activation="relu"))
    
      new_model.add(tf.keras.layers.Dense(100))
      new_model.add(tf.keras.layers.Dense(y_train_ctg.shape[1],activation='softmax'))
      new_model.compile(optimizer="adam", loss="mean_squared_error", metrics = ['accuracy']) 
      return new_model
    
    univar_model = basic_conv1D(n_filters=24, fsize=8, window_size=X_train_3.shape[1], n_features=X_train_3.shape[2])
    
    history = univar_model.fit(inputs[train],targets[train], epochs=200, batch_size = 32, validation_data = (inputs[test],targets[test]),callbacks=my_callback)
    y_predict = univar_model.predict(inputs[test])
    y_classes_test = [np.argmax(y1, axis=None, out=None) for y1 in targets[test]]
    y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

    #y_classes_test

    #y_classes

    y_predicts = le.inverse_transform(y_classes)

    y_true = le.inverse_transform(y_classes_test)

    y_predicts = pd.DataFrame(y_predicts)
    y_true = pd.DataFrame(y_true)

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

  # Fit data to model
    # history = model.fit(inputs[train], targets[train], validation_data=(inputs[test], targets[test]),
    #           batch_size=32,
    #           epochs=300,
    #           verbose=1)
    print("Input Train", inputs[train].shape)
    print("Input Test", inputs[test].shape)
  # Generate generalization metrics
    #scores = univar_model.evaluate(inputs[test], targets[test], verbose=0)
    scores = model.evaluate(inputs[test], targets[test], verbose=0,batch_size=32)
    #print(f'Score for fold {fold_no}: {univar_model.metrics_names[0]} of {scores[0]}; {univar_model.metrics_names[1]} of {scores[1]*100}%')
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    y_predicts.to_csv('D:/Research_work/exp4/CNN_result/{}predicts_s_6_32_c1d_0.62_9-14.csv'.format(fold_no))
    y_true.to_csv('D:/Research_work/exp4/CNN_result/{}true_s_6_32_c1d_0.62_9-14.csv'.format(fold_no))  #s_17_12_32_c2d was actually 9 class
  # Increase fold number
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()
    fold_no = fold_no + 1

    # y_predict = univar_model.predict(inputs[test])
    # y_classes_test = [np.argmax(y, axis=None, out=None) for y1 in y_test]
    # y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

    # #y_classes_test

    # #y_classes

    # y_predicts = le.inverse_transform(y_classes)
    # y_classes2 = to_categorical(y_classes)

    # y_true = le.inverse_transform(targets[test])
    # from sklearn.metrics import classification_report
    # from sklearn.metrics import accuracy_score
    # result = classification_report(y_test,y_classes2)
    # acc = accuracy_score(y_test, y_classes, normalize=False)
    # print(result)
    # print(acc)
    
    
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


 
    

# def evaluate_model(trainX, trainy, testX, testy):
#       # define model
#     verbose, epochs, batch_size = 0, 200, 128
#     n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
#           # reshape data into time steps of sub-sequences
#     # n_steps, n_length = 3, 47
#     n_steps, n_length = 4,33
    
#     trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
#     testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
#           # define model
#     # model = Sequential()
#     # model.add(ConvLSTM2D(filters = 128, return_sequences=True, kernel_size = (1,1),padding = "same", activation="relu",input_shape =  (n_steps, 1, n_length, n_features)))
#     # model.add(ConvLSTM2D(filters = 128, kernel_size = (1,1), activation="relu"))
      # model.add(Dropout(0.2))
#     # model.add(MaxPool2D(pool_size=(1,1)))
#     # model.add(ConvLSTM2D(filters = 64, kernel_size = (1,1), activation="relu"))
#     # model.add(MaxPool2D())
#     # model.add(Flatten())
#     # model.add(Dense(64, activation='relu'))
#     # model.add(Dropout(0.2))
#     # model.add(Dense(64, activation='relu'))
#     # model.add(Dropout(0.5))
#     # model.add(Dense(n_outputs, activation='softmax'))
#     # model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    
#     model = Sequential()
#     model.add(ConvLSTM2D(filters=256, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
#     model.add(Dropout(0.3))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#            #fit network
#     model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
#           # evaluate model
#     _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
#     y_predict = model.predict(testX)
#     y_classes_test = [np.argmax(y1, axis=None, out=None) for y1 in testy]
#     y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]
#     y_predicts = le.inverse_transform(y_classes)
#     y_true = le.inverse_transform(y_classes_test)
#     y_predicts = pd.DataFrame(y_predicts)
#     y_true = pd.DataFrame(y_true)

#     return accuracy
# # summarize scores



inputs = np.concatenate((X_train_31, X_test_31), axis=0)
targets = np.concatenate((y_train_3, y_test_3), axis=0)

# inputs = np.concatenate((X_train, X_test), axis=0)
# targets = np.concatenate((y_train_3, y_test_3), axis=0)
# inputs = pd.DataFrame(inputs)

k = 10
kfold = KFold(n_splits=k, shuffle = False)
acc_per_fold = []
loss_per_fold = []
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    def bi_LSTM(window_size=5, n_features=4):
      new_model = keras.Sequential()
      new_model.add(tf.keras.layers.LSTM(256, 
                    input_shape=(window_size, n_features),
                    return_sequences=True,
                    activation="relu"))
      new_model.add(Dropout(0.5))
      new_model.add(Bidirectional(tf.keras.layers.LSTM(256,activation="relu",return_sequences=True)))
      new_model.add(Dropout(0.5))
      new_model.add(Bidirectional(tf.keras.layers.LSTM(256,activation="relu",return_sequences=True)))
      new_model.add(Dropout(0.5))
      new_model.add(tf.keras.layers.LSTM(256,activation="relu",return_sequences=False))
      new_model.add(Dropout(0.5))
      new_model.add(tf.keras.layers.Flatten())
      new_model.add(tf.keras.layers.Dense(150, activation="relu"))
      #new_model.add(tf.keras.layers.Dense(100, activation="linear"))
      new_model.add(tf.keras.layers.Dense(y_test_ctg.shape[1]))
      new_model.compile(optimizer="adam", loss="mean_squared_error", metrics = ['accuracy']) 
      return new_model

    #Uncomment this portion, if you want to run only Bi-LSTM
    def basic_LSTM(window_size=5, n_features=4):
      new_model = keras.Sequential()
      new_model.add(tf.keras.layers.LSTM(256, 
                    input_shape=(window_size, n_features),
                    return_sequences=True,
                    activation="relu"))
      new_model.add(Dropout(0.5))
      new_model.add((tf.keras.layers.LSTM(256,activation="relu",return_sequences=True)))
      new_model.add(Dropout(0.5))
      new_model.add((tf.keras.layers.LSTM(256,activation="relu",return_sequences=True)))
      new_model.add(Dropout(0.5))
      new_model.add(tf.keras.layers.LSTM(256,activation="relu",return_sequences=False))
      new_model.add(Dropout(0.5))
      new_model.add(tf.keras.layers.Flatten())
      new_model.add(tf.keras.layers.Dense(150, activation="relu"))
      #new_model.add(tf.keras.layers.Dense(100, activation="linear"))
      new_model.add(tf.keras.layers.Dense(y_test_ctg.shape[1],activation='softmax'))
      new_model.compile(optimizer="adam", loss="mean_squared_error", metrics = ['accuracy']) 
      return new_model
 
    
    #ls_model = basic_LSTM(window_size=X_train_31.shape[1], n_features=X_train_31.shape[2])
    ls_model = bi_LSTM(window_size=X_train_31.shape[1], n_features=X_train_31.shape[2])
    ls_model.summary()
    history = ls_model.fit(inputs[train],targets[train], epochs=250, batch_size = 32, validation_data = (inputs[test],targets[test]),callbacks=my_callback)
    #ls_model.fit(X_train_3, y_train_ctg, epochs=150, batch_size =128, validation_data = (X_test_3, y_test_ctg))
    ls_model.evaluate(inputs[test], targets[test], verbose=0)
    y_predict = ls_model.predict(inputs[test])
    y_classes_test = [np.argmax(y1, axis=None, out=None) for y1 in targets[test]]
    y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

    #y_classes_test

    #y_classes

    y_predicts = le.inverse_transform(y_classes)

    y_true = le.inverse_transform(y_classes_test)

    y_predicts = pd.DataFrame(y_predicts)
    y_true = pd.DataFrame(y_true)
    # #y_true
    import csv

#If you want to run CONV1D model, just uncomment this portion, and comment others
    # def basic_conv1D(n_filters=10, fsize=5, window_size=5, n_features=4):
    #   new_model = keras.Sequential()
    #   new_model.add(tf.keras.layers.Conv1D(n_filters, fsize, padding="same", activation="relu", input_shape=(window_size, n_features)))
    #   #new_model.add(tf.keras.layers.GlobalAveragePooling1D())
    #   new_model.add(tf.keras.layers.Conv1D(n_filters, fsize, padding="same", activation="relu"))
    #   # Flatten will take our convolution filters and lay them out end to end so our dense layer can predict based on the outcomes of each
    #   new_model.add(tf.keras.layers.Flatten())
    
    #   new_model.add(tf.keras.layers.Dense(180, activation="relu"))
    
    #   new_model.add(tf.keras.layers.Dense(100))
    #   new_model.add(tf.keras.layers.Dense(y_train_ctg.shape[1]))
    #   new_model.compile(optimizer="adam", loss="mean_squared_error", metrics = ['accuracy']) 
    #   return new_model
    
    # univar_model = basic_conv1D(n_filters=24, fsize=8, window_size=X_train_3.shape[1], n_features=X_train_3.shape[2])
    
    # history = univar_model.fit(inputs[train],targets[train], epochs=200, batch_size = 16, validation_data = (inputs[test],targets[test]))
    # y_predict = univar_model.predict(inputs[test])
    # y_classes_test = [np.argmax(y1, axis=None, out=None) for y1 in targets[test]]
    # y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]

    # #y_classes_test

    # #y_classes

    # y_predicts = le.inverse_transform(y_classes)

    # y_true = le.inverse_transform(y_classes_test)

    # y_predicts = pd.DataFrame(y_predicts)
    # y_true = pd.DataFrame(y_true)

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

#Change this portion depending on which model you are running.
    print("Input Train", inputs[train].shape)
    print("Input Test", inputs[test].shape)
  # Generate generalization metrics
    #scores = univar_model.evaluate(inputs[test], targets[test], verbose=0)
    scores = ls_model.evaluate(inputs[test], targets[test], verbose=0)
    #print(f'Score for fold {fold_no}: {univar_model.metrics_names[0]} of {scores[0]}; {univar_model.metrics_names[1]} of {scores[1]*100}%')
    print(f'Score for fold {fold_no}: {ls_model.metrics_names[0]} of {scores[0]}; {ls_model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    y_predicts.to_csv('D:/Research_work/Exp2/ResultZ1_diff_classes2/{}predicts_s_9_12_32_bilstm_10-9.csv'.format(fold_no))
    y_true.to_csv('D:/Research_work/Exp2/ResultZ1_diff_classes2/{}_true_s_9_12_32_bilstm_10-9.csv'.format(fold_no))
    #inputs_col.to_csv('D:/Research_work/Data/{}_inputs_col.csv'.format(fold_no))
    #inputs.to_csv('D:/Research_work/Data/{}_inputs.csv'.format(fold_no))
  # Increase fold number
    fold_no = fold_no + 1
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')







# k fold
trainX = X_train
trainy = y_train_3
testX = X_test
testy = y_test_ctg
n_steps, n_length = 3,48
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
sgd = SGD(lr=0.01, decay=0.005, momentum=0.9, nesterov=True)
testX = testX.reshape((testX.shape[0], testX.shape[1],1))
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
inputs = np.concatenate((trainX, testX), axis=0)
targets = np.concatenate((trainy, testy), axis=0)
k = 10
kfold = KFold(n_splits=k, shuffle=False)
acc_per_fold = []
loss_per_fold = []
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    #def evaluate_model(trainX, trainy, testX, testy):
          # define model
          verbose, epochs, batch_size = 0, 300,32
          #n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
          # reshape data into time steps of sub-sequences
          n_steps, n_length = 3,48
          #n_steps, n_length = 4,33
        #n_steps, n_length = 3, 39
          #trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
          #testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
          # define model
          model = Sequential()
          model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
          model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3,activation='relu')))
          model.add(TimeDistributed(Dropout(0.5)))
          model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
          model.add(TimeDistributed(Flatten()))
          model.add(LSTM(256, return_sequences=True, activation='relu'))
          #model.add(Bidirectional(LSTM(256, return_sequences=True,activation='relu')))
          model.add(Dropout(0.5))
          #model.add(Bidirectional(LSTM(256, return_sequences=False,activation = 'relu')))
          #model.add(Dropout(0.5))
          #model.add(Bidirectional(LSTM(100)))
          # model.add(Dropout(0.5))
          # model.add(LSTM(256, return_sequences=False, activation='relu'))
          # model.add(Dropout(0.5))
          model.add(Dense(200, activation='relu'))
          #model.add(Dense(100, activation='relu'))
          model.add(Dense(y_train_ctg.shape[1], activation='softmax'))
          model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
          #model = basic_LSTM(window_size=X_train_3.shape[1], n_features=X_train_3.shape[2])
          model.summary()
          history = model.fit(inputs[train],targets[train], epochs=200, batch_size = 32, validation_data = (inputs[test],targets[test]),callbacks=my_callback)
          #ls_model.fit(X_train_3, y_train_ctg, epochs=150, batch_size =128, validation_data = (X_test_3, y_test_ctg))
          model.evaluate(inputs[test], targets[test], verbose=0, batch_size = 32)
          y_predict = model.predict(inputs[test])
          y_classes_test = [np.argmax(y1, axis=None, out=None) for y1 in targets[test]]
          y_classes = [np.argmax(yy, axis=None, out=None) for yy in y_predict]
      
          #y_classes_test
      
          #y_classes
      
          y_predicts = le.inverse_transform(y_classes)
      
          y_true = le.inverse_transform(y_classes_test)
      
          y_predicts = pd.DataFrame(y_predicts)
          y_true = pd.DataFrame(y_true)
          # #y_true
          import csv
          print('------------------------------------------------------------------------')
          print(f'Training for fold {fold_no} ...')
      
        # Fit data to model
          # history = model.fit(inputs[train], targets[train], validation_data=(inputs[test], targets[test]),
          #           batch_size=32,
          #           epochs=300,
          #           verbose=1)
          print("Input Train", inputs[train].shape)
          print("Input Test", inputs[test].shape)
        # Generate generalization metrics
          #scores = univar_model.evaluate(inputs[test], targets[test], verbose=0)
          scores = model.evaluate(inputs[test], targets[test], verbose=0 ,batch_size = 32)
          #print(f'Score for fold {fold_no}: {univar_model.metrics_names[0]} of {scores[0]}; {univar_model.metrics_names[1]} of {scores[1]*100}%')
          print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
          acc_per_fold.append(scores[1] * 100)
          loss_per_fold.append(scores[0])
          y_predicts.to_csv('D:/Research_work/exp4/tcd_result/{}predicts_s_6_tcd_32_class_9-14.csv'.format(fold_no))
          y_true.to_csv('D:/Research_work/exp4/tcd_result/{}true_s_6_tcd_32_class_9-14.csv'.format(fold_no))
        # Increase fold number
          fold_no = fold_no + 1
          # pyplot.plot(history.history['loss'], label='train')
          # pyplot.plot(history.history['val_loss'], label='test')
          # pyplot.legend()
          # pyplot.show()
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')          


    
 
