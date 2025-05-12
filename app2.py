import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ---------------------- USUARIOS ----------------------
if "users" not in st.session_state:
    st.session_state.users = {
        "admin": "1234",
        "odontologo": "periodontal"
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = ""

# ---------------------- PACIENTES E HISTORIAL ----------------------
if "pacientes" not in st.session_state:
    st.session_state.pacientes = []

if "historial" not in st.session_state:
    st.session_state.historial = {}

# ---------------------- CARGA Y GESTIÓN DEL MODELO ----------------------
def load_or_create_model():
    """Carga el modelo o crea uno nuevo si no existe."""
    model_path = "unet_periodontal.h5"
    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
    else:
        model = create_new_model()
    return model

def create_new_model():
    """Crea un nuevo modelo de red neuronal."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def retrain_model(training_images, training_labels):
    """Reentrena el modelo con los nuevos datos."""
    model = load_or_create_model()
    model.fit(training_images, training_labels, epochs=10, batch_size=32, validation_split=0.2)
    model.save("unet_periodontal.h5")
    st.success("Modelo reentrenado y guardado exitosamente.")
    return model

def evaluate_model(validation_images, validation_labels):
    """Evalúa el desempeño del modelo."""
    model = load_or_create_model()
    loss, accuracy = model.evaluate(validation_images, validation_labels)
    st.write(f"Precisión del modelo en conjunto de validación: {accuracy * 100:.2f}%")
    return accuracy

# ---------------------- INICIO DE SESIÓN ----------------------
def login():
    st.title("🔐 Iniciar sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.success(f"Bienvenido {username}")
        else:
            st.error("Usuario o contraseña incorrectos")

# ---------------------- PANEL DE ADMINISTRACIÓN DE USUARIOS ----------------------
def admin_user_panel():
    st.title("⚙️ Panel de Administración de Usuarios")
    st.subheader("Crear nuevo usuario")
    new_user = st.text_input("Nuevo usuario")
    new_pass = st.text_input("Contraseña", type="password")
    if st.button("Crear"):
        if new_user in st.session_state.users:
            st.warning("Ya existe.")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("Usuario creado.")
    st.subheader("Usuarios existentes")
    for user in list(st.session_state.users):
        if user != "admin":
            col1, col2 = st.columns([3, 1])
            col1.write(f"👤 {user}")
            if col2.button(f"Eliminar {user}"):
                del st.session_state.users[user]
                st.success(f"'{user}' eliminado.")
                st.rerun()

# ---------------------- GESTIÓN DEL MODELO SEGMENTADOR ----------------------
def admin_model_management():
    st.title("🔄 Gestión del Modelo Segmentador")
    st.subheader("Reentrenar modelo")
    
    uploaded_training_images = st.file_uploader("Sube las imágenes de entrenamiento", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    uploaded_training_labels = st.file_uploader("Sube las etiquetas de entrenamiento", type=["csv"], accept_multiple_files=False)

    if uploaded_training_images and uploaded_training_labels:
        training_images = [img_to_array(load_img(img, target_size=(IMG_HEIGHT, IMG_WIDTH))) / 255.0 for img in uploaded_training_images]
        training_labels = pd.read_csv(uploaded_training_labels).values
        if st.button("Reentrenar modelo"):
            model = retrain_model(np.array(training_images), np.array(training_labels))
            st.success("Modelo reentrenado correctamente.")

    # Evaluación del modelo
    st.subheader("Evaluar modelo")
    uploaded_validation_images = st.file_uploader("Sube las imágenes de validación", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    uploaded_validation_labels = st.file_uploader("Sube las etiquetas de validación", type=["csv"], accept_multiple_files=False)

    if uploaded_validation_images and uploaded_validation_labels:
        validation_images = [img_to_array(load_img(img, target_size=(IMG_HEIGHT, IMG_WIDTH))) / 255.0 for img in uploaded_validation_images]
        validation_labels = pd.read_csv(uploaded_validation_labels).values
        if st.button("Evaluar desempeño"):
            accuracy = evaluate_model(np.array(validation_images), np.array(validation_labels))
            st.success(f"Precisión del modelo: {accuracy * 100:.2f}%")

# ---------------------- PACIENTES E HISTORIAL ----------------------
def ver_historial():
    st.title("📋 Historial Clínico")
    if not st.session_state.pacientes:
        st.info("No hay pacientes registrados.")
        return
    paciente = st.selectbox("Selecciona un paciente", st.session_state.pacientes)
    historial = st.session_state.historial.get(paciente, [])
    if historial:
        for i, entry in enumerate(historial):
            st.markdown(f"**{i+1}.** {entry}")
    else:
        st.warning("Este paciente no tiene historial clínico aún.")

def registrar_paciente():
    st.title("🧾 Registrar nuevo paciente")
    nombre = st.text_input("Nombre del paciente")
    edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
    dni = st.text_input("DNI")
    if st.button("Registrar"):
        if nombre.strip() == "" or dni.strip() == "":
            st.error("Nombre y DNI son obligatorios.")
        elif nombre in st.session_state.pacientes:
            st.warning("Este paciente ya está registrado.")
        else:
            st.session_state.pacientes.append(nombre)
            st.session_state.historial[nombre] = []
            st.success("Paciente registrado correctamente.")

# ---------------------- FLUJO ----------------------
if not st.session_state.logged_in:
    login()
    st.stop()

# Menú lateral
st.sidebar.title("Menú")

# Botón de cerrar sesión
st.sidebar.markdown("---")
if st.sidebar.button("🔓 Cerrar sesión"):
    st.session_state.logout = True
    st.rerun()

# Manejo de cierre de sesión
if st.session_state.get("logout", False):
    st.session_state.logged_in = False
    st.session_state.current_user = ""
    st.session_state.logout = False
    st.rerun()

# Menú según el rol
if st.session_state.current_user == "admin":
    opcion = st.sidebar.radio("Seleccione una opción", [
        "⚙️ Panel de Administración de Usuarios", 
        "🔄 Gestión del Modelo Segmentador"
    ])

    if opcion == "⚙️ Panel de Administración de Usuarios":
        admin_user_panel()

    elif opcion == "🔄 Gestión del Modelo Segmentador":
        admin_model_management()

elif st.session_state.current_user == "odontologo":
    # Opciones del odontólogo
    opcion = st.sidebar.radio("Seleccione una opción", [
        "📷 Diagnóstico con IA", "📋 Ver Historial Clínico", "🧾 Registrar Paciente"
    ])

    if opcion == "📋 Ver Historial Clínico":
        ver_historial()
    elif opcion == "🧾 Registrar Paciente":
        registrar_paciente()
    else:
        st.title("🦷 Detección de Enfermedad Periodontal")
        if not st.session_state.pacientes:
            st.warning("Primero debes registrar un paciente.")
        else:
            paciente_sel = st.selectbox("Selecciona un paciente", st.session_state.pacientes)
            archivo = st.file_uploader("Sube una imagen (.jpg)", type=["jpg", "jpeg"])
            if archivo is not None:
                temp_path = "uploaded_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(archivo.getbuffer())
                show_image_and_prediction(temp_path, paciente_sel)
