
import streamlit as st
#from streamlit_option_menu import option_menu
import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

#page congiguration
st.set_page_config(page_title= "Book Recommendation",
                   page_icon= 'random',
                   layout= "wide",)



st.markdown("<h1 style='text-align: center; color: black;'>Book Recommendation</h1>",
                unsafe_allow_html=True)
"""
selected = option_menu(None, ["PREDICT RE SALE PRICE"],
                           icons=['cash-coin'],orientation='horizontal',default_index=0,
styles={
        "container": {"background-color":'white',"height":"60px","border": "3px solid #000000","border-radius": "0px"},
        "icon": {"color": "black", "font-size": "16px"}, 
        "nav-link": {"color":"black","font-size": "15px", "text-align": "centre", "margin":"4px", "--hover-color": "white","border": "1px solid #000000", },
        "nav-link-selected": {"background-color": "#5F259F"},})"""

#if selected == 'PREDICT RE SALE PRICE':
    #flat_type = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM','MULTI-GENERATION']
c1,c2=st.columns([2,2])
with c1:
    cid=st.text_input('Enter your Customer id')
    book_price=st.text_input('Enter the book price')
    book_Language=st.text_input('Enter the preferred book language')
    publisher_name=st.text_input('Enter the preferred Publisher name')
with c2:
    Author_name=st.text_input('Enter the preferred Author name')
    published_decade=st.text_input('Enter the preferred Prefered decade')
    page_category=st.text_input('Enter the preferred page size')
    days_since_last_order=st.text_input('Enter the no of days since your last order')

with c1:
    st.write('')
    st.write('')
    st.write('')
   
    if st.button('PREDICT SELLING PRICE'):
        with open('customer_id_encoder.pkl', 'rb') as file:
            fte = pickle.load(file)
        with open('book_language_name_encoder.pkl', 'rb') as file:
            fbe = pickle.load(file)
        with open('publisher_name_encoder.pkl', 'rb') as file:
            fse = pickle.load(file)
        with open('author_name_encoder.pkl', 'rb') as file:
            fme = pickle.load(file)
        with open('published_decade_encoder.pkl', 'rb') as file:
            fae = pickle.load(file)
        with open('page_category_encoder.pkl', 'rb') as file:
            foe = pickle.load(file)
        with open('page_category_encoder.pkl', 'rb') as file:
            scaled_datax = pickle.load(file)                
        with open('scalery1.pkl', 'rb') as file:
            scaled_datay = pickle.load(file)
        with open('dtreg_model1.pkl','rb') as file:
            dtreg_loaded_model = pickle.load(file)
        days_since_last_order1 = pd.to_numeric(days_since_last_order, errors='coerce')
        book_price1 = pd.to_numeric(book_price, errors='coerce')
        
        cid_e=np.array([cid])
        e_cid = fae.transform(cid_e)
        e_cid =e_cid[0].astype(int)
        Author_name_e=np.array([Author_name])
        st.write(Author_name_e)
        e_Author_name = fme.transform(Author_name_e)
        e_Author_name =e_Author_name[0].astype(int)
        book_Language_e=np.array([book_Language])
        e_book_Language = fbe.transform(book_Language_e)
        e_book_Language =e_book_Language[0].astype(int)
        publisher_name_e=np.array([publisher_name])
        e_publisher_name = fse.transform(publisher_name_e)
        e_publisher_name =e_publisher_name[0].astype(int)
        published_decade_e=np.array([published_decade])
        e_published_decade = fae.transform(published_decade_e)
        e_published_decade =e_published_decade[0].astype(int)
        page_category_e=np.array([page_category])
        e_page_category = foe.transform(page_category_e)
        e_page_category =e_page_category[0].astype(int)
       
        
        data =[]
        data.append(e_cid)
        data.append(e_book_Language)
        data.append(e_Author_name)
        data.append(e_publisher_name)
        data.append(e_published_decade)
        data.append(e_page_category)
        data.append(days_since_last_order1)
        data.append(book_price1)
        x = np.array(data).reshape(1, -1)
        st.write(x)
        pred_model = scaled_datax.transform(x)
        predicted_probs = model.predict(pred_model)
# Gt the index of the predicted class (most probable book title)
        predicted_index = np.argmax(predicted_probs)
# Dcode the index to get the book title
        predicted_book_title = label_encoder.inverse_transform([predicted_index])[0]
        #price_predict= dtreg_loaded_model.predict(pred_model)
        #y_pred_inverse_scaled = scaled_datay.inverse_transform(price_predict.reshape(-1, 1)).flatten()
        #y_pred_original = np.exp(y_pred_inverse_scaled)
        #predicted_price = str(y_pred_original)[1:-1]
        st.write(f'Predicted re sale value : :green[₹] :green[{predicted_book_title}]')