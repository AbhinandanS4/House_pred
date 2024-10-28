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
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error
    def lr_prediction_range(model, X_new, num_simulations=100, confidence_multiplier=1.2):
    # Initialize a list to store predictions for each simulation
        simulation_predictions = []
    
    # Perturb `X_new` and make predictions
        for _ in range(num_simulations):
        # Add small noise to `X_new` for each feature
            perturbed_X = X_new.copy()
            for col in perturbed_X.columns:
                if perturbed_X[col].dtype in [np.float64, np.int64]:  # Only apply to numeric columns
                    perturbed_X[col] += np.random.normal(0, 0.01 * perturbed_X[col].std(), perturbed_X[col].shape)
        
        # Make a prediction with the perturbed input and store it
            simulation_predictions.append(model.predict(perturbed_X)[0])
    
    # Calculate the standard deviation of the simulated predictions
        prediction_std = np.std(simulation_predictions)
    
    # Make the initial prediction on `X_new`
        predictions = model.predict(X_new)
    
    # Define margin of error based on the standard deviation of simulated predictions
        margin_of_error = prediction_std * confidence_multiplier
    
    # Compute lower and upper bounds
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        return lower_bound, upper_bound

    pred=int(lr.predict(x))
    lower_bound, upper_bound = lr_prediction_range(lr, X, y, x)
    if selection=='Ranged':
        st.write(f"Your House will cost will be in range: {u'\u20B9'}{round(lower_bound[0],2)} - {u'\u20B9'}{round(upper_bound[0],2)}" )
    elif selection=='Discrete':
        st.write(f"Estimated Cost of the House will be: {u'\u20B9'}{pred}")
if st.button("Submit Feedback"):
    with open("feedback.txt", "a") as f:
        f.write(feedback + "\n")
  # Here you can process and store the feedback (e.g., save to a file or database).
    st.success("Thank you for your feedback!")
