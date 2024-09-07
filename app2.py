import pandas as pd
import joblib
import streamlit as st
import ssl
from sklearn.preprocessing import OrdinalEncoder

model = joblib.load('model_xgb.pkl')

ordinal_encoder = joblib.load('ordinal_encoder.pkl')



#model = pk.load(open('model.pkl', 'rb'))

st.header('Modelo de prediccion de Carros')

ssl._create_default_https_context = ssl._create_unverified_context

data = pd.read_csv('https://github.com/sergiomora03/AdvancedTopicsAnalytics/raw/main/datasets/dataTrain_carListings.zip')

año= str(st.slider('Seleccione Año de Fabricación', 1990, 2024))
Kilometraje = st.slider('Seleccione Kilometraje', 0, 300000, format="%d Km")
marca = st.selectbox('Seleccionar Marca del carro', data['Make'].unique())
estado = st.selectbox('Seleccionar Estado', data['State'].unique())
#modelo = st.selectbox('Select a car Model', data['Model'].unique())

modelos_filtrados = data[data['Make'] == marca]['Model'].unique()
modelo = st.selectbox('Seleccionar Modelo de carro', modelos_filtrados)

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
