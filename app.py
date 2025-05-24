import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Predicción de Riesgo Vehicular", layout="centered")

# Cargar el modelo y herramientas
with open("modelo-clas-tree-knn-nn.pkl", "rb") as file:
    modelTree, modelKnn, modelNN, labelencoder, variables, scaler = pickle.load(file)

# Encabezado con imagen
st.title("🚗 Predicción de Riesgo Vehicular")
st.markdown("Esta aplicación predice el **riesgo de aseguramiento** de un conductor según su edad y tipo de vehículo usando modelos de machine learning.")

# Imagen representativa
image = Image.open("auto_riesgo.jpg")  # Cambia el nombre de archivo según tu imagen
st.image(image, caption="Análisis de riesgo en seguros vehiculares", use_column_width=True)

# Sidebar para inputs
st.sidebar.header("🔧 Parámetros de entrada")

age = st.sidebar.slider("Edad del conductor", 18, 100, 30)
cartype = st.sidebar.selectbox("Tipo de vehículo", ["combi", "family", "minivan", "sport"])
modelo_seleccionado = st.sidebar.selectbox("Modelo de clasificación", ["Árbol de Decisión", "KNN", "Red Neuronal"])

if st.sidebar.button("Predecir riesgo"):
    # Preparar datos de entrada
    cartype_options = ["combi", "family", "minivan", "sport"]
    cartype_dummies = [1 if cartype == option else 0 for option in cartype_options]
    X_input = np.array([age] + cartype_dummies).reshape(1, -1)
    X_input[:, 0] = scaler.transform(X_input[:, [0]]).ravel()  # Normaliza edad

    # Hacer predicción
    if modelo_seleccionado == "Árbol de Decisión":
        pred = modelTree.predict(X_input)
    elif modelo_seleccionado == "KNN":
        pred = modelKnn.predict(X_input)
    else:
        pred = modelNN.predict(X_input)

    riesgo = labelencoder.inverse_transform(pred)[0]

    # Mostrar resultado
    st.success(f"🔎 Riesgo predicho: **{riesgo}**")
    st.balloons()

# Footer
st.markdown("---")
st.markdown("Desarrollado por David, Misael y Eduar - Proyecto de clasificación de riesgos de seguros. ©2025")
