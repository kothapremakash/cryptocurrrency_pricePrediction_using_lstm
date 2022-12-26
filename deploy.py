import pickle
import os
import pandas as pd

import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


# For PLotting we will use these library

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

st.title('Cryptocurrency Price Prediction')
st.header('Deployment using streamlit')
st.markdown('To see the Bitcoin dataset')
df=pickle.load(open('df.pkl','rb'))

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')


y_2014 = df.loc[(df['Date'] >= '2021-11-12')
                     & (df['Date'] < '2022-11-12')]

y_2014.drop(y_2014[['Adj Close','Volume']],axis=1)
if st.button('ClickHere '):
    st.write(y_2014)
monthvise= y_2014.groupby(y_2014['Date'].dt.strftime('%B'))[['Open','Close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
             'September', 'October', 'November', 'December']
monthvise = monthvise.reindex(new_order, axis=0)
st.markdown('To see the Open and Close price monthly wise')
if st.button('ClickHere  '):
    st.bar_chart(monthvise)

y_2014.groupby(y_2014['Date'].dt.strftime('%B'))['Low','High'].min()
monthvise_high = y_2014.groupby(df['Date'].dt.strftime('%B'))['Low','High'].max()
monthvise_high = monthvise_high.reindex(new_order, axis=0)
st.markdown('To see the Bitcoin Analysis Chart')
if st.button('ClickHere   '):
    names = cycle(['Bitcoin Open Price','Bitcoin Close Price','Bitcoin High Price','Bitcoin Low Price'])

    fig = px.line(y_2014, x=y_2014.Date, y=[y_2014['Open'], y_2014['Close'],
                                          y_2014['High'], y_2014['Low']],
             labels={'Date': 'Date','value':'Stock value'})
    fig.update_layout(title_text='Bitcoin analysis chart', font_size=15, font_color='black',legend_title_text='Bitcoin Parameters')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.show()
st.markdown('To See the Time frame from 2015-2022 of Bitcoin')
if st.button('ClickHere    '):
    fig = px.line(df, x=df.Date, y=df.Close, labels={'date': 'Date', 'close': 'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
    fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2022', plot_bgcolor='white',
                      font_size=15, font_color='black')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

image = Image.open(r'C:\Users\prema\OneDrive\Desktop\loss.png')
st.markdown('To See the Training and validation loss')
if st.button("ClickHere     "):
    st.image(image, caption='Training and validation loss')
image = Image.open(r'C:\Users\prema\OneDrive\Desktop\predicted_closeprice.jpg')
st.markdown('To see Compare close price between predicted close price')
if st.button("ClickHere      "):
    st.image(image, caption='Orginal close price Vs Predicted close price')
image = Image.open(r'C:\Users\prema\OneDrive\Desktop\Predicted_nt30.jpg')
st.markdown('To see the predicted next 30 days close price')
if st.button("ClickHere        "):
    st.image(image, caption='compare the last 15 days closed price vs next 30 days closed price')
image = Image.open(r'C:\Users\prema\OneDrive\Desktop\Wholepredicted_closeprice.jpg')
st.markdown('To see the Whole predicted  close price')
if st.button("ClickHere         "):
    st.image(image, caption='Whole closing Bitcoin Price with Prediction')





