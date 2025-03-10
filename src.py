import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler

#header and subheader
st.header('Stock Market Predictor')
stock =st.text_input('Enter Stock Symnbol', 'GOOG')

#getting the data
API_KEY = "4CT7Q5APB1Q99DFK"
ts = TimeSeries(key=API_KEY, output_format="pandas")

if stock: 
    data, meta_data = ts.get_daily(symbol=stock, outputsize="full")
    
    #showing stock data
    st.subheader('Stock Data')
    st.write('Data for {stock}')
    st.write(data)

#loading the model
model = load_model('Stock Predictions Model.keras')

#splitting the data
data.reset_index(inplace=True)
data.rename(columns={'1. open':'open','2. high': 'high','3. low':'low','4. close':'close','5. volume':'volume'}, inplace=True)
data_train = pd.DataFrame(data.close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.close[int(len(data)*0.80): len(data)])

#scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

#plotting Price vs moving average 50
st.subheader('Price vs MA50')
ma_50_days = data.close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r',label = 'moving average(50 days)')
plt.plot(data.close, 'g', label = 'closing price')
plt.legend()
plt.show()
st.pyplot(fig1)

#plotting Price vs moving average 50 vs moving average 100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r',label = 'moving average(50 days)')
plt.plot(ma_100_days, 'b',label = 'moving average(100 days)')
plt.plot(data.close, 'g',label = 'closing price')
plt.legend()
plt.show()
st.pyplot(fig2)

#preparing input for model
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

#predicting from model
predict = model.predict(x)

#reverse scaling the data
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

#plotting predicted vs original price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)