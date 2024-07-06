# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:05:40 2024

@author: User
"""
from dateutil.relativedelta import relativedelta
from datetime import datetime

import numpy as np
import requests
import json
import time
import csv
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
from dateutil.relativedelta import relativedelta
from datetime import datetime

from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數

############################跑lstm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

###########神經元層
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention

# Adding the LSTM layer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm #使用進度條
import tensorflow as tf
from random import sample
import os


##############建資料##################

##載入NP檔
#x_bigdataT=np.load(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/x_bigdataT.npy')
#x_bigdata=x_bigdataT

#y_bigdata=np.load(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/y_bigdata.npy')




x_bigdataT=np.load(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/x_bigdataTst.npy')
x_bigdata=x_bigdataT
y_bigdata=np.load(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/y_bigdata.npy')


##跑模型##
indexs=np.random.permutation(len(x_bigdata)) #隨機排序 49005以下的數字
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
test_indexs=indexs[int(len(x_bigdata)*0.8):]

#x部分
x_bigdata_array=np.array(x_bigdata)
x_train=x_bigdata_array[train_indexs]
x_val=x_bigdata_array[val_indexs]
x_test=x_bigdata_array[test_indexs]
#y部分
y_bigdata_array=np.array(y_bigdata)
y_train=y_bigdata_array[train_indexs]
y_val=y_bigdata_array[val_indexs]
y_test=y_bigdata_array[test_indexs]

##開始訓練##
#########建立並訓練網路模型#############
# 建立一個Sequential型態的model
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention

# Adding the LSTM layer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm #使用進度條
import tensorflow as tf
import xgboost as xgb
from keras.layers import Concatenate
from keras.layers import Conv1D , MaxPool2D , Flatten , Dropout,Conv2D
from keras.layers import GRU  #載入GRU

###############套件
n = 10
n_steps = n 
n_features = 21
model = keras.Sequential(name='model-9')

model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape = (n_features,n_steps)))


model.add(LSTM(10,activation='tanh'))

#model.add(LSTM(10,activation='tanh', return_sequences=True, input_shape = (n_steps, n_features)))
####加入attention################
#model.add(SeqSelfAttention(attention_activation='tanh')) #ttention_activation='sigmoid')  relu
#model.add(LSTM(n,activation='relu'))   
model.add(Dense(1))
# 顯示網路模型架構
model.summary()


#自訂損失函數ustom_mean_squared_error
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

#設定訓練使用的優化器、損失函數和指標函數：
model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
              metrics=[keras.metrics.MeanAbsoluteError()])


#創建模型儲存目錄：
#在C:/Users/User/lab2-logs/models/建立模型目錄
model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/model9/'
#os.makedirs(model_dir)

#設定回調函數：
# TensorBoard回調函數會幫忙紀錄訓練資訊，並存成TensorBoard的紀錄檔
log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model9')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
#將模型儲存在C:/Users/User/lab2-logs/models/
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')


#訓練網路模型：
history = model.fit(x_train, y_train,  # 傳入訓練數據
               batch_size=64,  # 批次大小設為64
               epochs=100,  # 整個dataset訓練300遍
               validation_data=(x_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp])  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型


#訓練結果
history.history.keys()  # 查看history儲存的資訊有哪些

#在model.compile已經將損失函數設為均方誤差(Mean Square Error)
#所以history紀錄的loss和val_loss為Mean Squraed Error損失函數計算的損失值
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0.001, 0.006)
plt.title('Mean square error')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

# 載入模型
#model = keras.models.load_model(r'D:/2021 4月開始的找回程式之旅/lab2-logs/model9/Best-model-1.h5')

# 預測測試數據
y_pred = model.predict(x_test)

# 顯示誤差到小數點第二位 #0.05
meanmae_error=np.mean(abs(y_test- np.array(y_pred)))
print(" 平均mae誤差: {:.2f}".format(meanmae_error))
    
    
    
    