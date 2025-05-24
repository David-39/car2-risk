import streamlit as st
import numpy as np
import pickle

# Cargar el modelo
with open("modelo-clas-tree-knn-nn.pkl", "rb") as file:
    modelTree, modelKnn, modelNN, labelencoder, variables, scaler = pickle.load(file)

# Título de la app
st.title("Predicción de Riesgo de Vehículos")
st.write("Ingrese los datos del cliente para predecir el riesgo.")

# Entradas del usuario
age = st.slider("Edad del conductor", 18, 100, 30)
cartype = st.selectbox("Tipo de vehículo", ["combi", "family", "minivan", "sport"])

# Preprocesamiento de entrada
cartype_options = ["combi", "family", "minivan", "sport"]
cartype_dummies = [1 if cartype == option else 0 for option in cartype_options]

# Ordenar las entradas igual que en el entrenamiento
X_input = np.array([age] + cartype_dummies).reshape(1, -1)
X_input[:, 0] = scaler.transform(X_input[:, [0]]).ravel()  # Normalizamos la edad

# Selección del modelo
modelo_seleccionado = st.selectbox("Seleccione el modelo para la predicción", ["Árbol de Decisión", "KNN", "Red Neuronal"])

if st.button("Predecir"):
    if modelo_seleccionado == "Árbol de Decisión":
        pred = modelTree.predict(X_input)
    elif modelo_seleccionado == "KNN":
        pred = modelKnn.predict(X_input)
    elif modelo_seleccionado == "Red Neuronal":
        pred = modelNN.predict(X_input)

    pred_label = labelencoder.inverse_transform(pred)[0]
    st.success(f"El nivel de riesgo predicho es: **{pred_label}**")