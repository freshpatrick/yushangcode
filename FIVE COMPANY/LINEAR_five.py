# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:08:17 2024

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression



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





###使用minmaxscalar
x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))


for k in range(0,5):  #len(stock_id)
    print("第"+str(k)+"支股票")
    #先拿台泥做比較
    # Load data into a pandas DataFrame
    df = final_data_real[k]
    #df.head(10)
    
    
    

# Convert the date column to ordinal values
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].apply(lambda date: date.toordinal())

# Split the data into features (X) and target (y)
    X = df.drop(["Adj Close","Date"], axis=1)
    y = df["Adj Close"]
    
    
   ##########先儲存好mean和std變數###################
    train_y =y[:int(len(y) *0.8)]
    price_mean=y.mean()  #test['return_rate'].mean()
    price_std=y.std()  #test['return_rate'].std()
    
    #將原本y_test儲存起來
    y_testc=np.array(df["Adj Close"][int(len(df) *0.8):])
    
    #####標準化###
    #使用z-score標準化X
    #X=(X-X.mean())/X.std()
     #使用z-score標準化
    #y=(y-y.mean())/y.std()
    
    

# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #########使用minmaxscalar######
    X_train=x_scaler.fit_transform(X_train)
    #X_trainorign=x_scaler.inverse_transform(X_train1)
    y_train=y_scaler.fit_transform(pd.DataFrame(y_train)).reshape(-1)  
    X_test=x_scaler.fit_transform(X_test)
    y_train=y_scaler.fit_transform(pd.DataFrame(y_train)).reshape(-1)
    y_test=y_scaler.fit_transform(pd.DataFrame(y_test)).reshape(-1)
    #y_testorign=y_scaler.inverse_transform(pd.DataFrame(y_test1))
    
    #########minmaxscalar結束######
    
    
# Fit the Random Forest Regressor model to the training data
    rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)
    rfr.fit(X_train, y_train)


# Predict the stock prices on the testing set
    y_pred = rfr.predict(X_test)
    
    ##minmax還原成正常的prediction
    y_pred = y_scaler.inverse_transform(pd.DataFrame(y_pred))
    y_pred=np.reshape(y_pred,y_test.shape)



    
    # Calculate the mean squared error of the predictions
    mae = np.mean(np.abs(y_testc - y_pred))
    
    stock_mae.append(mae) #股票MSE
    stock.append(stock_id[k]) #股票名稱


big_rfr_data=pd.concat([pd.DataFrame(stock),pd.DataFrame(stock_mae)], axis=1)

big_rfr_data.mean() #42.494757


big_rfr_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/五個股票/MINMAXSCALARfive_outputdata_randomforest_regresion.csv', encoding='utf_8_sig')


    