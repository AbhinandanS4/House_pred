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
area = st.slider("Area (in sq ft.):",100,17000,5000)
bedrooms=st.slider('Bedrooms:',0,15,10)
bathrooms=st.slider('Bathrooms:',0,15,10)
stories=st.slider('Stories:',0,10,5)
parking=st.slider('Parkings Available:',0,5,2)
prefarea=st.selectbox("Preferred Area?" ,['yes','no'])
mainroad = st.selectbox("Main Road", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning?", ['yes', 'no'])
furnishingstatus=st.selectbox("Furnishing Status",['furnished','semi-furnished','unfurnished'])
ok=st.button("Predict")
x = pd.DataFrame({
        'area': [float(area)],
        'bedrooms': [int(bedrooms)],
        'bathrooms': [int(bathrooms)],
        'stories': [int(stories)],
        'mainroad': [mainroad],
        'airconditioning': [airconditioning],
        'parking': [int(parking)],
        'prefarea':[prefarea],
        'furnishingstatus': [furnishingstatus]
})
st.subheader("Your Selections:")
st.dataframe(x)
if ok:
    # Prepare the input data based on user input 
    x['mainroad']=le.fit_transform(x['mainroad'])
    x['airconditioning']=le.fit_transform(x['airconditioning'])
    x['furnishingstatus']=le.fit_transform(x['furnishingstatus'])
    x['prefarea']=le.fit_transform(x['prefarea'])
    pred=int(lr.predict(x))

        
    st.write(f"Estimated Cost of the House will be: {u'\u20B9'}{pred}")
    st.header("Feedback")
feedback = st.text_area("Please share your feedback about the prediction or the app's usability:")

if st.button("Submit Feedback"):
    with open("feedback.txt", "a") as f:
        f.write(feedback + "\n")
  # Here you can process and store the feedback (e.g., save to a file or database).
    st.success("Thank you for your feedback!")
