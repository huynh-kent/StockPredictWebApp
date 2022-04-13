import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# start date-today
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
DAYS = 365
MONTHS = 12

# streamlit
st.title("Stock Prediction App")
# stock list
stocks = ("SPY", "AAPL", "GOOG", "MSFT", "GME", "BTC", "TSLA", "NVDA", "FTNT", "AMZN", "AMD", "QQQ")
# slider
selected_stock = st.selectbox("Select Desired Stock", stocks)
num_years = st.slider("Years of prediction:", 1, 4)
period = num_years * DAYS

# @st.cache # caches data everytime we load a stock, note could cause errors
def loadStockData(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True) # puts date in first column
    return data

# load text & data
data_loadstate = st.text("Load data...") 
data = loadStockData(selected_stock)
data_loadstate.text("Loading data... DONE!")

# data section
st.subheader('Raw data')
st.write(data.tail()) # tail has most recent data

# graph data with plotly
def plot_rawData():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_rawData()

# Prediction
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"}) # rename for fbprophet, takes ds,y

# train model
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period) # future df
predict = model.predict(future) 

# prediction data
st.subheader('Prediction Data')
st.write(predict.tail())

# plot/graph
st.write('predict data')
fig1 = plot_plotly(model, predict)
st.plotly_chart(fig1)
st.write('predict components')
fig2 = model.plot_components(predict)
st.write(fig2)