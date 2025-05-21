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
        "admin": {"password": "1234", "role": "admin"},
        "odontologo": {"password": "periodontal", "role": "odontologo"}
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if "registering" not in st.session_state:
    st.session_state.registering = False

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
        if username in st.session_state.users and st.session_state.users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.success(f"Bienvenido {username}")
            
        else:
            st.error("Usuario o contraseña incorrectos")
    if st.button("Registrarse"):
        st.session_state.registering = True
        

def register():
    st.title("📝 Registrarse")
    username = st.text_input("Nuevo usuario")
    password = st.text_input("Contraseña", type="password")
    nombre = st.text_input("Nombre")
    apellido = st.text_input("Apellido")
    dni = st.text_input("DNI")
    edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
    sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
    if st.button("Registrarse"):
        if username in st.session_state.users:
            st.warning("El usuario ya existe.")
        elif len(dni) != 8 or not dni.isdigit():
            st.error("DNI debe tener 8 cifras numéricas.")
        else:
            st.session_state.users[username] = {
                "password": password,
                "nombre": nombre,
                "apellido": apellido,
                "dni": dni,
                "edad": edad,
                "sexo": sexo,
                "role": "paciente"
            }
            st.session_state.pacientes.append(username)
            st.session_state.historial[username] = []
            st.success("Usuario registrado correctamente. Por favor, inicia sesión.")
            st.session_state.registering = False
            
    if st.button("Volver al inicio de sesión"):
        st.session_state.registering = False
        

def admin_panel():
    st.title("⚙️ Panel de Administración de Usuarios")
    st.subheader("Crear nuevo usuario")
    new_user = st.text_input("Nuevo usuario")
    new_pass = st.text_input("Contraseña", type="password")
    nombre = st.text_input("Nombre")
    apellido = st.text_input("Apellido")
    dni = st.text_input("DNI")
    edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
    sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
    role = st.selectbox("Rol", ["paciente", "odontologo", "admin"])
    if st.button("Crear"):
        if new_user in st.session_state.users:
            st.warning("Ya existe.")
        elif len(dni) != 8 or not dni.isdigit():
            st.error("DNI debe tener 8 cifras numéricas.")
        else:
            st.session_state.users[new_user] = {
                "password": new_pass,
                "nombre": nombre,
                "apellido": apellido,
                "dni": dni,
                "edad": edad,
                "sexo": sexo,
                "role": role
            }
            if role == "paciente":
                st.session_state.pacientes.append(new_user)
                st.session_state.historial[new_user] = []
            st.success("Usuario creado.")
    st.subheader("Usuarios existentes")
    for user in list(st.session_state.users):
        if user != "admin":
            col1, col2 = st.columns([3,1])
            col1.write(f"👤 {user}")
            if col2.button(f"Eliminar {user}"):
                del st.session_state.users[user]
                st.success(f"'{user}' eliminado.")

def show_patient_data():
    user_data = st.session_state.users[st.session_state.current_user]
    st.title(f"Bienvenido {user_data['nombre']} {user_data['apellido']}")
    st.write(f"DNI: {user_data['dni']}")
    st.write(f"Edad: {user_data['edad']}")
    st.write(f"Sexo: {user_data['sexo']}")
    if st.button("Ver Informes"):
        ver_historial_paciente()

def ver_historial_paciente():
    st.title("📋 Historial Clínico")
    historial = st.session_state.historial.get(st.session_state.current_user, [])
    if historial:
        for i, entry in enumerate(historial):
            st.markdown(f"**{i+1}.** {entry}")
    else:
        st.warning("No tienes historial clínico aún.")

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
    img_array_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array_exp)[0]
    pred_mask = np.argmax(pred, axis=-1)

    pred_diag, pred_class = get_predicted_diagnosis(pred_mask)

    # Crear superposición con colores
    overlay = img_array.copy()
    mask_colors = {
        1: [255, 255, 0],   # Gingivitis leve: Amarillo
        2: [255, 165, 0],   # Gingivitis moderada: Naranja
        3: [255, 0, 0],     # Gingivitis severa: Rojo
        4: [0, 255, 0],     # Placa bacteriana: Verde
        5: [0, 0, 255],     # Sarro: Azul
        6: [128, 0, 128]    # Otros: Morado
    }

    alpha = 0.5
    color_mask = np.zeros_like(img_array)

    for class_id, color in mask_colors.items():
        mask = pred_mask == class_id
        for i in range(3):  # Aplicar color a cada canal RGB
            color_mask[..., i] += mask * color[i]

    # Aplicar transparencia (alpha blending)
    blended = (1 - alpha) * img_array + alpha * (color_mask / 255.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img)
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")
    ax[1].imshow(blended)
    ax[1].set_title("Predicción del Modelo (superpuesta)")
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

# ---------------------- FLUJO ----------------------
if not st.session_state.logged_in:
    if st.session_state.registering:
        register()
    else:
        login()
    st.stop()

# Menú lateral
st.sidebar.title("Menú")

# Botón de cerrar sesión
st.sidebar.markdown("---")
if st.sidebar.button("🔓 Cerrar sesión"):
    st.session_state.logged_in = False
    st.session_state.current_user = ""
    

# Admin Panel solo para el admin
if st.session_state.current_user == "admin":
    admin_panel()
    st.markdown("---")

# Mostrar datos del paciente
if st.session_state.current_user in st.session_state.pacientes:
    show_patient_data()
elif st.session_state.current_user and st.session_state.users[st.session_state.current_user]["role"] == "odontologo":
    opcion = st.sidebar.radio("Seleccione una opción", [
        "📷 Diagnóstico con IA", "📋 Ver Historial Clínico"
    ])

    # Funcionalidad según la opción seleccionada
    if opcion == "📋 Ver Historial Clínico":
        ver_historial()
    else:
        st.title("🦷 Detección de Enfermedad Periodontal")
        paciente_sel = st.selectbox("Selecciona un paciente", st.session_state.pacientes)
        archivo = st.file_uploader("Sube una imagen (.jpg)", type=["jpg", "jpeg"])
        if archivo is not None:
            temp_path = "uploaded_image.jpg"
            with open(temp_path, "wb") as f:
                f.write(archivo.getbuffer())
            show_image_and_prediction(temp_path, paciente_sel)
else:
    opcion = st.sidebar.radio("Seleccione una opción", [
        "📷 Diagnóstico con IA", "📋 Ver Historial Clínico"
    ])

    # Funcionalidad según la opción seleccionada
    if opcion == "📋 Ver Historial Clínico":
        ver_historial()
    else:
        st.title("🦷 Detección de Enfermedad Periodontal")
        paciente_sel = st.selectbox("Selecciona un paciente", st.session_state.pacientes)
        archivo = st.file_uploader("Sube una imagen (.jpg)", type=["jpg", "jpeg"])
        if archivo is not None:
            temp_path = "uploaded_image.jpg"
            with open(temp_path, "wb") as f:
                f.write(archivo.getbuffer())
            show_image_and_prediction(temp_path, paciente_sel)
