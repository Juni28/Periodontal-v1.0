import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

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

def admin_panel():
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
            col1, col2 = st.columns([3,1])
            col1.write(f"👤 {user}")
            if col2.button(f"Eliminar {user}"):
                del st.session_state.users[user]
                st.success(f"'{user}' eliminado.")
                st.experimental_rerun()

# ---------------------- IA y diagnóstico ----------------------
model = load_model("unet_periodontal.h5", compile=False)

diagnosis_labels = {
    0: "Sano", 1: "Gingivitis Leve", 2: "Gingivitis Moderada",
    3: "Gingivitis Severa", 4: "Placa Bacteriana", 5: "Sarro", 6: "Otros"
}

def get_predicted_diagnosis(pred_mask):
    classes, counts = np.unique(pred_mask, return_counts=True)
    class_id = classes[np.argmax(counts)]
    return diagnosis_labels.get(class_id, "Desconocido"), class_id

def show_image_and_prediction(image_path, patient_name):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    pred_mask = np.argmax(pred, axis=-1)

    pred_diag, pred_class = get_predicted_diagnosis(pred_mask)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img)
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")
    ax[1].imshow(pred_mask, cmap="jet", alpha=0.6)
    ax[1].set_title("Predicción del Modelo")
    ax[1].axis("off")

    st.pyplot(fig)

    st.subheader("🧠 Diagnóstico IA")
    st.markdown(f"""
    **Segmentación**: lesiones detectadas en múltiples zonas  
    **Nivel de enfermedad**: Clase {pred_class} ({pred_diag})
    """)

    st.subheader("📝 Observaciones del odontólogo")
    obs = st.text_area("Escribe tus observaciones aquí...")
    if st.button("Guardar observaciones"):
        entry = f"Diagnóstico IA: Clase {pred_class} ({pred_diag}) | Observaciones: {obs}"
        st.session_state.historial.setdefault(patient_name, []).append(entry)
        st.success("Observaciones guardadas en el historial del paciente.")

# ---------------------- GESTIÓN CLÍNICA ----------------------
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
    st.session_state.logged_in = False
    st.session_state.current_user = ""
    st.experimental_rerun()

# Admin Panel solo para el admin
if st.session_state.current_user == "admin":
    admin_panel()
    st.markdown("---")

# Opciones del menú
opcion = st.sidebar.radio("Seleccione una opción", [
    "📷 Diagnóstico con IA", "📋 Ver Historial Clínico", "🧾 Registrar Paciente"
])

# Funcionalidad según la opción seleccionada
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
