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

def zom():
    st.session_state['company'] = 'zomato'

def ford():
    st.session_state['company'] = 'ford'

def nvda():
    st.session_state['company']='Nvidia'


# Page: Home
if st.session_state['page'] == 'home':
    st.title('SPW PDW Project')
    st.write('This is a model to predict future stock prices of companies. It is trained on dataset of three companies i.e. Zomato,Nvidia and Ford motor. It will ask for what value you want to predict on which date.')
    
    if st.button('Go to main page', on_click=navigate_to, args=['mum']):
        pass
    
    st.divider()
    st.header('Dikshit Singla')
    st.write('I am a student of IIT Bombay currently pursuing my Dual degree in electrical engineering department. ')
    
# Page: Main
elif st.session_state['page'] == 'mum':
    st.balloons()
    st.title('Get an idea of the future stock price of the following company')
    st.write('Click on the company name')
    
    if st.button("Ford Motor", on_click=ford):
        pass
    elif st.button("Zomato", on_click=zom):
        pass
    elif st.button('Nvdia',on_click=nvda):
        pass

    if st.session_state['company'] == 'zomato':
        st.title('ZOMATO')
        tummy = st.date_input('Enter the date')
        n= st.selectbox('Which value do you want to predict among',['High','Low','Open','Close','Adj Close','Volume'])
        st.image("zomato.jpeg",width=300)

            # Load your stock price data
            # The data should be a CSV file with columns 'Date' and 'Close' (or adjust the column names accordingly)
        data = pd.read_csv('ZOMATO.NS.csv')
        if tummy and n:
                # try:

            data['ds'] = pd.to_datetime(data['Date'],format= '%Y-%m-%d')
            data['y'] = data[n]
            data = data[['ds','y']]
        

                #train_data = data.iloc[:-365]  # all data except the last 365 days
                #test_data = data.iloc[-365:] 
                # Initialize the Prophet model
            model = Prophet()
            
            model.fit(data)
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
                #forecast_test = forecast[['ds', 'yhat']].set_index('ds').join(test_data.set_index('ds'))
                #forecast_test.dropna(inplace=True)
            
            def predict_stock_value(date_str):
                date = pd.to_datetime(date_str)
                prediction = forecast[forecast['ds'] == date]
                    #if date not in future['ds'].values:
                    #   print(f"Date {date_str} is out of the prediction range.")
                    #  return None
                    #prediction = forecast[forecast['ds'] == date]
                if not prediction.empty:
                    predicted_value = prediction['yhat'].values[0]
                    print(f"The predicted stock value for {date_str} is: {predicted_value:.2f}")
                    return predicted_value
                else:
                    #print(f"No prediction available for {date_str}.")
                    return None
            
            predicted_value=predict_stock_value(tummy)
            if predicted_value is not None:
                st.heading(f'The predicted stock value for {tummy} is: {predicted_value:.2f} rs')
            else:
                st.write(f'No prediction available for {tummy}.')


        def showgraph():
            fig, ax = plt.subplots()
            ax.plot(data['ds'], data['y'])
            ax.set_title('Graph of Dataset')
            ax.set_xlabel('Date')
            ax.set_ylabel(n)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        if st.button('Show graph of dataset')== True:
            showgraph()
        

    elif st.session_state['company'] == 'ford':
        st.title('FORD MOTORS')
        yummy = st.date_input('Enter the date')
        m = st.selectbox('Which value do you want to predict among the following',['High','Low','Open','Close','Adj Close','Volume'])
        st.image("frod.jpeg",width=300)

        data = pd.read_csv('F (1).csv')
        if yummy and m:
            #data['ds'] = pd.to_datetime(data['Date'])
            data['ds'] = pd.to_datetime(data['Date'],format= '%Y-%m-%d')
        
            #newinput = yummy.strftime("%y%m%d")
                #st.write(newinput in newlist.tolist())
            data['y'] = data[m]
            data = data[['ds', 'y']]
                
            #train_data = data.iloc[:-365]
            #test_data = data.iloc[-365:]
            model = Prophet()
            model.fit(data)
                
                # Make future predictions
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
                
                # Function to predict stock value
            def predict_stock_value(date_str):
                #date = date_str # pd.(date_str,format="%Y-%m-%d")
                #newlist = future['ds'].apply(lambda x: x.strftime("%y%m%d")).tolist()
                date = pd.to_datetime(date_str)
                prediction = forecast[forecast['ds'] == date]
                #if date in newlist:
                    #   st.write(f"Date {date_str} is out of the prediction range.")
                    #  return None
                #prediction = forecast[forecast['ds'] == date]
                if not prediction.empty:
                    predicted_value = prediction['yhat'].values[0]
                    print(f"The predicted stock value for {date_str} is: {predicted_value:.2f}")
                    return predicted_value
                else:
                    #st.write(f"No prediction available for {date_str}.")
                    return None
            predicted_value=predict_stock_value(yummy)
            if predicted_value is not None:
               st.write(f'The predicted stock value for {yummy} is: {predicted_value:.2f} dollar')
            else:
                st.write(f'No prediction available for {yummy}.')
        
    
        def showgraph():
            fig, ax = plt.subplots()
            ax.plot(data['ds'], data['y'])
            ax.set_title('Graph of Dataset')
            ax.set_xlabel('Date')
            ax.set_ylabel(m)
            plt.xticks(rotation=45)

            st.pyplot(fig)

        if st.button('Show graph of dataset')== True:
            showgraph()

    else:
        st.title('NVIDIA')
        gummy = st.date_input('Enter the date')
        o= st.selectbox('Which value do you want to predict among',['High','Low','Open','Close','Adj Close','Volume'])
        st.image("nvidia.webp",width=300)
        # Load your stock price data
        # The data should be a CSV file with columns 'Date' and 'Close' (or adjust the column names accordingly)
        data = pd.read_csv('NVDA (2).csv')
        if gummy and o:
            # try:

            data['ds'] = pd.to_datetime(data['Date'],format= '%Y-%m-%d')
            data['y'] = data[o]
            data = data[['ds','y']]
            # Initialize the Prophet model
            model = Prophet()
            
            model.fit(data)
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
        
            def predict_stock_value(date_str):
                date = pd.to_datetime(date_str)
                prediction = forecast[forecast['ds'] == date]
                #if date not in future['ds'].values:
                    #   print(f"Date {date_str} is out of the prediction range.")
                    #  return None
                #prediction = forecast[forecast['ds'] == date]
                if not prediction.empty:
                    predicted_value = prediction['yhat'].values[0]
                    print(f"The predicted stock value for {date_str} is: {predicted_value:.2f}")
                    return predicted_value
                else:
                    #print(f"No prediction available for {date_str}.")
                    return None
        
            predicted_value=predict_stock_value(gummy)
            if predicted_value is not None:
                st.write(f'The predicted stock value for {gummy} is: {predicted_value:.2f} rs')
            else:
                st.write(f'No prediction available for {gummy}.')


        def showgraph():
            fig, ax = plt.subplots()
            ax.plot(data['ds'], data['y'])
            ax.set_title('Graph of Dataset')
            ax.set_xlabel('Date')
            ax.set_ylabel(o)
            st.pyplot(fig)

        if st.button('Show graph of dataset')==True:
            showgraph()

if st.session_state['page']=='mum':         
    st.button('Go to home page', on_click=navigate_to, args=['home'])    