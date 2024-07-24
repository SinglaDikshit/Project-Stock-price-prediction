import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'company' not in st.session_state:
    st.session_state['company'] = ''


# Navigation functions
def navigate_to(page):
    st.session_state['page'] = page


def define_company(company_name, data_file):
    st.session_state['company'] = company_name
    try:
        # Load data from CSV
        data = pd.read_csv(data_file)
        data['ds'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data['y'] = data[st.session_state['value']]
        data = data[['ds', 'y']]
        return data
    except FileNotFoundError:
        st.error(f"Data file '{data_file}' not found!")
        return None


def zom():
    data = define_company('zomato', 'ZOMATO.NS.csv')
    if data is not None:
        predict_stock_price(data)


def ford():
    data = define_company('ford', 'F (1).csv')
    if data is not None:
        predict_stock_price(data)


def nvda():
    data = define_company('nvidia', 'NVDA (2).csv')
    if data is not None:
        predict_stock_price(data)


def predict_stock_price(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    def predict_value(date_str):
        date = pd.to_datetime(date_str)
        prediction = forecast[forecast['ds'] == date]
        if not prediction.empty:
            predicted_value = prediction['yhat'].values[0]
            print(f"The predicted stock value for {date_str} is: {predicted_value:.2f}")
            return predicted_value
        else:
            return None

    # User Interface based on selected company
    if st.session_state['company'] == 'zomato':
        st.title('ZOMATO')
        tummy = st.date_input('Enter the date')
        st.session_state['value'] = st.selectbox('Which value do you want to predict?', ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume'])
        st.image("zomato.jpeg", width=300)

        if tummy and st.session_state['value']:
            predicted_value = predict_value(tummy)
            if predicted_value is not None:
                st.heading(f'The predicted stock value for {tummy} is: {predicted_value:.2f} rs')
            else:
                st.write(f'No prediction available for {tummy}.')

        def showgraph():
            fig, ax = plt.subplots()
            ax.plot(data['ds'], data['y'])
            ax.set_title('Graph of Dataset')
            ax.set_xlabel('Date')
