# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:53:25 2024

@author: User
"""
####套件安裝###########
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
######新增套件###########
from dateutil.relativedelta import relativedelta
from datetime import datetime
import requests
import json
import time
import csv
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
from dateutil.relativedelta import relativedelta
from datetime import datetime
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
import os



# Transforming the TSLA stock data into 3D arrays
###############################################開始整理資
#############################所有股票#################

AAPL= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/AAPL.csv', encoding='utf_8_sig')

AMZN= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/AMZN.csv', encoding='utf_8_sig')

GOOG= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/GOOG.csv', encoding='utf_8_sig')

MSFT= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/MSFT.csv', encoding='utf_8_sig')

TSLA= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/比較的論文程式碼/TFMS-Multifactor-Analysis/inputdata/TSLA.csv', encoding='utf_8_sig')
#####################合併大資料集 final_data_real##############################
final_data_real=[]
final_data_real.append(AAPL)
final_data_real.append(AMZN)
final_data_real.append(GOOG)
final_data_real.append(MSFT)
final_data_real.append(TSLA)

stock_id=['AAPL','AMZN','GOOGL','MSFT','TSLA']

stock_mae=[] #股票MSE
stock=[] #股票名稱

###複製表格
final_data_real_copy=final_data_real


####################loss function結案###################
###使用minmaxscalar
x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))



########################開始跑迴歸做比較################ 414筆資料
for k in range(0,5):  #len(stock_id)
    print("第"+str(k)+"支股票")
    tsla_data =final_data_real[k]
    ##將月份移到index#########
    tsla_data.columns
    tsla_data.set_index(['Date'], inplace=True)
    

    # Extracting the close price from the DataFrame
    tsla_close = tsla_data['Close'].values
    # Normalizing the TSLA stock data using MinMaxScaler
    tsla_data=tsla_data.drop('Adj Close', axis=1)
   
    ######## train 60% val 20% test 20%   ##############
    n = 10
    train =tsla_data[:int(len(tsla_data) *0.6)]
    val =tsla_data[:int(len(tsla_data) *0.8)]
    test =tsla_data[int(len(tsla_data) *0.8):]
    ##保留test10天候的數值
    y_testc=test['Close'][n:]
    feature_names = list(train.drop('Close', axis=1).columns)
    x_train = []
    y_train = []
    train_indexes = []
    #train 資料
    norm_data_xtrain = train[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(train)-n)):#range(0,len(train)-n) 
        ##加入minmax value        
        x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
        x_trainaddscalar=x_scaler.fit_transform(x_trainadd)

        x_train.append(x_trainaddscalar)  #x_train.append(norm_data_xtrain.iloc[i:i+n]. values) 
        #修改
        y_train.append(train['Close'].iloc[i+n]) #現有資料+10天的Yy_train.append(train['return_rate'].iloc[i+n-1])
        train_indexes.append(train.index[i+n]) #Y的日期  train_indexes.append(train.index[i+n-1]) #Y的日期
    ##轉成array
    print(x_train[0])
    
    x_train=np.array(x_train)
    y_train_dataframe=pd.DataFrame(y_train).iloc[:len(y_train)]
    y_train_tran=y_scaler.fit_transform(y_train_dataframe)
    y_train=np.array(y_train_tran).reshape(-1)
    print(x_train.shape)
    
    
    #val 資料
    x_val = []
    y_val = []
    val_indexes = []
    norm_data_xval = val[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(val)-n)):#range(0,len(train)-n) 
        ##加入minmax value        
        x_valadd=norm_data_xval.iloc[i:i+n]. values
        x_valaddscalar=x_scaler.fit_transform(x_valadd)

        x_val.append(x_valaddscalar)  #x_train.append(norm_data_xtrain.iloc[i:i+n]. values) 
        #修改
        y_val.append(val['Close'].iloc[i+n]) #現有資料+10天的Yy_train.append(train['return_rate'].iloc[i+n-1])
        val_indexes.append(val.index[i+n]) #Y的日期  train_indexes.append(train.index[i+n-1]) #Y的日期
    ##轉成array
    print(x_val[0])
    
    x_val=np.array(x_val)
    y_val_dataframe=pd.DataFrame(y_val).iloc[:len(y_val)]
    y_val_val=y_scaler.fit_transform(y_val_dataframe)
    y_val=np.array(y_val_val).reshape(-1)
    print(x_val.shape)    
      
    ##test部分##
    x_test = []
    y_test = []
    test_indexes = []
    
    norm_data_xtest = test[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(test)-n)): 
        x_testadd=norm_data_xtest.iloc[i:i+n]. values
        x_testaddscalar=x_scaler.fit_transform(x_testadd)
        x_test.append(x_testaddscalar) #x_test.append(norm_data_xtest.iloc[i:i+n]. values) 
        y_test.append(test['Close'].iloc[i+n]) #現有資料+30天的Y
        test_indexes.append(test.index[i+n]) #Y的日期

    #先備份
    x_test1=x_test
    y_test1=y_test

    x_test=np.array(x_test)

    y_test_dataframe=pd.DataFrame(y_test).iloc[:len(y_test)]
    y_test_tran=y_scaler.fit_transform(y_test_dataframe)
    y_test=np.array(y_test_tran).reshape(-1) 
    
    #開始跑模型
    n = 10
    n_steps = n 
    n_features = 4
    #model = Sequential(name='model-3')
    model = keras.models.Sequential()
    model.add(LSTM(30,activation='relu', return_sequences=True, input_shape = (n_steps, n_features)))  
    model.add(LSTM(60,activation='relu'))
    model.add(Dense(1))
    # 顯示網路模型架構
    model.summary()
    model.compile(keras.optimizers.Adam(0.001),
    loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()  loss=custom_mean_squared_error
    metrics=[keras.metrics.MeanAbsoluteError()])
    
    

    #設定回調函數
    model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock/model8/'
    #os.makedirs(model_dir)
    # TensorBoard回調函數會幫忙紀錄訓練資訊，並存成TensorBoard的紀錄檔
    log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock', 'model8')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    # ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
    #將模型儲存在C:/Users/User/lab2-logs/models/
    model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                                 monitor='val_mean_absolute_error', 
                                                 save_best_only=True, 
                                                 mode='min')
    
    
    
    
    
    
    #history = model.fit(x_train,y_train,batch_size=20,epochs=100, verbose=2)
    #原本條件
    #history = model.fit(x_train,y_train,batch_size=64,epochs=100)
    
    #訓練網路模型：
    history = model.fit(x_train, y_train,  # 傳入訓練數據
               batch_size=32,  # 批次大小設為64
               epochs=100,  # 整個dataset訓練300遍
               validation_data=(x_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp])  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型

    
    
    
    
    #############預測x_test
    predictions = model.predict(x_test)
    #np.array(predictions,y_test.shape)
    predictions1=predictions.reshape(-1)
    ##minmax還原成正常的prediction
    predictions_orign = y_scaler.inverse_transform(predictions)
    
    # 顯示誤差百分比 顯示到小數點第二位
    meanmae_error=np.mean(abs(predictions_orign- np.array(y_testc)))
    
    ##將誤差和資料儲存起來
    stock_mae.append(meanmae_error) #股票MSE
    stock.append(stock_id[k]) #股票名稱
    
    #####修改結束#################
    # 顯示誤差百分比 顯示到小數點第二位
    #print(" 平均mae誤差: {:.2f}".format(percentagemean_error))

##合併大資料 ############
big_lstm_data=pd.concat([pd.DataFrame(stock),pd.DataFrame(stock_mae)], axis=1)
big_lstm_data.mean()  #24.836001
big_lstm_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/五個股票/MINMAXSCALAR 3060five_outputdata_lstm.csv', encoding='utf_8_sig')

