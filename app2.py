import pandas as pd
import joblib
#import pickle as pk
import streamlit as st
import ssl
from sklearn.preprocessing import OrdinalEncoder

model = joblib.load('model_rf.pkl')

ordinal_encoder = joblib.load('ordinal_encoder.pkl')

#model = pk.load(open('model.pkl', 'rb'))

st.header('Car Prediction ML Model')

ssl._create_default_https_context = ssl._create_unverified_context

data = pd.read_csv('https://github.com/sergiomora03/AdvancedTopicsAnalytics/raw/main/datasets/dataTrain_carListings.zip')

año= str(st.slider('Select a car Year', 1990, 2024))
Kilometraje = st.slider('Select a car Mileage', 0, 300000)
marca = st.selectbox('Select a car Make', data['Make'].unique())
estado = st.selectbox('Select a car State', data['State'].unique())
#modelo = st.selectbox('Select a car Model', data['Model'].unique())

modelos_filtrados = data[data['Make'] == marca]['Model'].unique()
modelo = st.selectbox('Select a car Model', modelos_filtrados)

if st.button("Predict"):
    # Crear el DataFrame de predicción
    Y_pred = pd.DataFrame([[año, Kilometraje, marca, estado, modelo]], columns=['Year', 'Mileage', 'Make', 'State', 'Model'])

    ordinal_encoder = OrdinalEncoder()
    data[['State', 'Make', 'Model']] = ordinal_encoder.fit_transform(data[['State', 'Make', 'Model']])

  
    # Verificar si la marca y el modelo existen en el codificador
    try:
        Y_encoded = Y_pred.copy()
        Y_encoded[['State', 'Make', 'Model']] = ordinal_encoder.transform(Y_pred[['State', 'Make', 'Model']]).astype(int)

        # Mostrar los datos antes y después de la codificación
        #st.write("Datos de predicción antes de codificar:")
        #st.write(Y_pred)
        #st.write("Datos de predicción después de codificar:")
        #st.write(Y_encoded)

        # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
        feature_order = ['Year', 'Mileage', 'State', 'Make', 'Model']
        Y_encoded = Y_encoded[feature_order]
        
        # Realizar la predicción con los datos codificados
        prediccion = model.predict(Y_encoded)

        # Mostrar el resultado de la predicción
        st.write(f'El precio estimado del coche es: ${prediccion[0]:,.2f}')

    except ValueError as e:
        st.error(f"Error: {e}")
