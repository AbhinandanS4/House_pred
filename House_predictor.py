import streamlit as st
import pandas as pd
import pickle

url = 'https://raw.githubusercontent.com/AbhinandanS4/House_pred/refs/heads/main/Housing.csv'
df = pd.read_csv(url, index_col=0)
df.drop(columns=['guestroom','basement','hotwaterheating','prefarea'],inplace=True)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

le=LabelEncoder()
Scr=StandardScaler()

def load_model():

    with open('lr_model.pkl','rb') as file:

        data=pickle.load(file)

    return data

lr= pickle.load(open('lr_model.pkl','rb'))


st.title("House Price Predictor")
st.write("""Please Select The Features of Your House""")
area = st.text_input("Area (in sq ft.):")
bedrooms=st.text_input('Bedrooms:')
bathrooms=st.text_input('Bathrooms:')
stories=st.text_input('Stories:')
parking=st.text_input('Parkings Available:')
mainroad = st.selectbox("Main Road", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning?", ['yes', 'no'])
furnishingstatus=st.selectbox("Furnishing Status",['furnished','semi-furnished','unfurnished'])
ok=st.button("Predict")
if ok:
    # Prepare the input data based on user input
    x = pd.DataFrame({
        'area': [float(area)],
        'bedrooms': [int(bedrooms)],
        'bathrooms': [int(bathrooms)],
        'stories': [int(stories)],
        'mainroad': [mainroad],
        'airconditioning': [airconditioning],
        'parking': [int(parking)],
        'furnishingstatus': [furnishingstatus]
    })  
    x['mainroad']=le.fit_transform(x['mainroad'])
    x['airconditioning']=le.fit_transform(x['airconditioning'])
    x['furnishingstatus']=le.fit_transform(x['furnishingstatus'])
    pred=int(lr.predict(x))

        
    st.write(f"Estimated Cost of the House will be: {u'\u20B9'}{pred}")
