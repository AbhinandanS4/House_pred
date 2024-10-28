import streamlit as st
import pandas as pd
import pickle
import numpy as np
url = 'https://raw.githubusercontent.com/AbhinandanS4/House_pred/refs/heads/main/Housing.csv'
df = pd.read_csv(url, index_col=0)
df.drop(columns=['guestroom','basement','hotwaterheating','prefarea'],inplace=True)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=12)
X_train['mainroad']=le.fit_transform(X_train['mainroad'])
X_train['airconditioning']=le.fit_transform(X_train['airconditioning'])
X_train['furnishingstatus']=le.fit_transform(X_train['furnishingstatus'])

le=LabelEncoder()
Scr=StandardScaler()

def load_model():

    with open('lr_model.pkl','rb') as file:

        data=pickle.load(file)

    return data

lr= pickle.load(open('lr_model.pkl','rb'))


st.title("House Price Predictor")
selection=st.selectbox("Select Your Prediction Type",['Ranged','Discrete'])
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
ok=st.button("Predict")
if ok:
    # Prepare the input data based on user input 
    x['mainroad']=le.fit_transform(x['mainroad'])
    x['airconditioning']=le.fit_transform(x['airconditioning'])
    x['furnishingstatus']=le.fit_transform(x['furnishingstatus'])
    x['prefarea']=le.fit_transform(x['prefarea'])
    def lr_prediction_range(model, X_new, confidence_level=0.95):
        predictions = model.predict(x)
        prediction_std = np.std(model.predict(X_train) - y_train)
        margin_of_error = prediction_std * 1.2  # 1.2 corresponds to a 20% confidence interval
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        return lower_bound, upper_bound
    pred=int(lr.predict(x))
    lower_bound, upper_bound = lr_prediction_range(lr, x)
    if selection=='Ranged':
        st.write(f"Your House will cost will be in range: {u'\u20B9'}{round(lower_bound[0],2)} - {u'\u20B9'}{round(upper_bound[0],2)}" )
    elif selection=='Discrete':
        st.write(f"Estimated Cost of the House will be: {u'\u20B9'}{pred}")
if st.button("Submit Feedback"):
    with open("feedback.txt", "a") as f:
        f.write(feedback + "\n")
  # Here you can process and store the feedback (e.g., save to a file or database).
    st.success("Thank you for your feedback!")
