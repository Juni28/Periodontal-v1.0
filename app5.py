import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from db_operations import DBOperations # Assuming this is available
from crear_bd import inicializar_base_datos # Assuming this is available
from datetime import datetime

# Configuración inicial
inicializar_base_datos()
model = load_model("unet_periodontal.h5", compile=False)
classifier_model = load_model("modelo_clasificacion_diagnostico.h5", compile=False)

# Constantes
DIAGNOSIS_LABELS = {
    0: "Sano", 1: "Gingivitis Leve", 2: "Gingivitis Moderada",
    3: "Gingivitis Severa", 4: "Placa Bacteriana", 5: "Sarro", 6: "Otros"
}

COLOR_MAP = {
    1: [255, 255, 0],     # Gingivitis leve: Amarillo
    2: [255, 165, 0],     # Gingivitis moderada: Naranja
    3: [255, 0, 0],       # Gingivitis severa: Rojo
    4: [0, 255, 0],       # Placa bacteriana: Verde
    5: [0, 0, 255],       # Sarro: Azul
    6: [128, 0, 128]      # Otros: Morado
}

# ---------------------- FUNCIONES DE APOYO ----------------------
def get_predicted_diagnosis(pred_mask):
    classes, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size

    # Crear diccionario clase->cantidad
    class_counts = dict(zip(classes, counts))

    # Eliminar clase 0 (fondo)
    if 0 in class_counts:
        del class_counts[0]

    if not class_counts:
        # Solo fondo en máscara
        return DIAGNOSIS_LABELS[0], 0

    # Encontrar clase con mayor porcentaje de pixeles (excepto fondo)
    class_id = max(class_counts, key=class_counts.get)
    ratio = class_counts[class_id] / total_pixels

    # Definir un umbral para considerar diagnóstico válido (ejemplo 1%)
    umbral = 0.01

    if ratio < umbral:
        # No hay suficientes pixeles de enfermedad
        return DIAGNOSIS_LABELS[0], 0
    else:
        return DIAGNOSIS_LABELS.get(class_id, "Desconocido"), class_id


def process_and_predict_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array_exp)[0]
    pred_mask = np.argmax(pred, axis=-1)

    pred_diag, pred_class = get_predicted_diagnosis(pred_mask)
    
    # Crear superposición de colores
    overlay = img_array.copy()
    color_mask = np.zeros_like(img_array)

    for class_id, color in COLOR_MAP.items():
        mask = pred_mask == class_id
        for i in range(3):
            color_mask[..., i] += mask * color[i]

    # Aplicar transparencia
    alpha = 0.5
    blended = (1 - alpha) * img_array + alpha * (color_mask / 255.0)

    return img, blended, pred_diag, pred_class

def predict_classification(image_path):
    # Load the image with target size 256x256
    # Try to load as RGBA directly if the model expects 4 channels.
    # If the image is originally RGB, it will be converted to RGBA by Pillow.
    img = load_img(image_path, target_size=(256, 256), color_mode='rgba')
    img_array = img_to_array(img) / 255.0

    # Ensure the image array has exactly 4 channels
    if img_array.shape[-1] != 4:
        # This case should ideally not happen if color_mode='rgba' works as expected,
        # but as a fallback, we can explicitly add an alpha channel if missing.
        # This implies that the model was trained with 4 channels.
        if img_array.shape[-1] == 3:
            alpha_channel = np.ones((256, 256, 1)) # Create an opaque alpha channel
            img_array = np.concatenate([img_array, alpha_channel], axis=-1)
        else:
            # Handle other unexpected channel counts (e.g., grayscale)
            # This indicates a severe mismatch.
            raise ValueError(f"Image has {img_array.shape[-1]} channels, but the model expects 4. "
                             "Please ensure your classification model (modelo_clasificacion_diagnostico.h5) "
                             "was trained correctly or the input image format is suitable.")

    # Expand dimensions to create a batch of 1
    img_array_exp = np.expand_dims(img_array, axis=0) # Shape will be (1, 256, 256, 4)

    pred = classifier_model.predict(img_array_exp)[0]
    class_id = np.argmax(pred)
    confidence = float(pred[class_id])

    return DIAGNOSIS_LABELS.get(class_id, "Desconocido"), class_id, confidence


# ---------------------- PÁGINAS DE LA APLICACIÓN ----------------------
def login_page():
    st.title("🔐 Sistema de Diagnóstico Dental")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Iniciar sesión")
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        
        if st.button("Ingresar"):
            user = DBOperations.get_user_by_username(username)
            if user and user["pass_usu"] == password:
                st.session_state.user = dict(user)
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos")
    
    with col2:
        st.subheader("Registrarse")
        if st.button("Soy nuevo paciente"):
            st.session_state.registrando = True
            st.rerun()

def register_page():
    st.title("📝 Registro de Nuevo Paciente")
    
    with st.form("registro_paciente"):
        dni = st.text_input("DNI (8 dígitos)", max_chars=8)
        nombre = st.text_input("Nombres")
        apellido = st.text_input("Apellidos")
        edad = st.number_input("Edad", min_value=1, max_value=120)
        sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
        username = st.text_input("Nombre de usuario")
        password = st.text_input("Contraseña", type="password")
        
        if st.form_submit_button("Registrarme"):
            if len(dni) != 8 or not dni.isdigit():
                st.error("El DNI debe tener exactamente 8 dígitos numéricos")
            elif DBOperations.get_user_by_username(username):
                st.error("El nombre de usuario ya está en uso")
            else:
                if DBOperations.create_user(username, password, nombre, apellido, dni, edad, sexo, "paciente"):
                    st.success("Registro exitoso. Por favor inicie sesión.")
                    st.session_state.registrando = False
                    st.rerun()
                else:
                    st.error("Error al registrar el usuario")
    
    if st.button("Volver al inicio"):
        st.session_state.registrando = False
        st.rerun()

def admin_dashboard():
    st.title("⚙️ Panel de Administración")
    st.write(f"Bienvenido administrador: {st.session_state.user['nom_usu']}")

def patient_dashboard():
    st.title(f"👤 Bienvenido {st.session_state.user['nom_usu']}")
    
    st.subheader("Mi Historial Clínico")
    paciente = DBOperations.get_all_patients() # This still fetches all patients, might want to refine this if you only want the logged-in patient's info.
    # Assuming the patient's user entry in `st.session_state.user` contains their patient ID 'id_paci'
    if 'id_paci' in st.session_state.user:
        history = DBOperations.get_patient_history(st.session_state.user['id_paci'])
        if history:
            for record in history:
                with st.expander(f"Consulta del {record['fecha_diagnostico']}"):
                    st.write(f"Diagnóstico: {DIAGNOSIS_LABELS.get(record['nivel_enfe'], 'Desconocido')}")
                    st.write(f"Observaciones: {record['observacion']}")
        else:
            st.info("Aún no tienes registros en tu historial clínico")
    else:
        st.warning("No se encontró el ID de paciente para el usuario actual.")


def professional_dashboard():
    st.title(f"👨‍⚕️ Panel de {st.session_state.user['rol_usu'].capitalize()}")
    
    tab1, tab2 = st.tabs(["Nuevo Diagnóstico", "Historial Pacientes"])
    
    with tab1:
        st.subheader("🦷 Nuevo Diagnóstico con IA")
        
        # Selección de paciente corregida
        pacientes = DBOperations.get_all_patients()
        
        if pacientes:
            # Formatear opciones para el selectbox
            opciones_pacientes = [
                f"{p['nom_usu']} {p['ape_usu']} (DNI: {p['dni_usu']})"
                for p in pacientes
            ]
            
            # Mapeo entre opción mostrada y ID de paciente
            paciente_dict = {
                opcion: p['id_paci']
                for opcion, p in zip(opciones_pacientes, pacientes)
            }
            
            paciente_seleccionado = st.selectbox(
                "Seleccione un paciente",
                options=opciones_pacientes
            )
            
            paciente_id = paciente_dict[paciente_seleccionado]
            
            # Subida de imagen
            uploaded_file = st.file_uploader(
                "Suba una imagen dental (JPG/PNG)", 
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file is not None:
                # Procesar imagen
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    original_img, blended_img, pred_diag, pred_class = process_and_predict_image(temp_path)
                    clf_diag, clf_class, clf_conf = predict_classification(temp_path)
                    
                    # Mostrar resultados
                    st.subheader("Resultados del Análisis")
                    
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    ax[0].imshow(original_img)
                    ax[0].set_title("Imagen Original")
                    ax[0].axis("off")
                    
                    ax[1].imshow(blended_img)
                    ax[1].set_title("Diagnóstico IA")
                    ax[1].axis("off")
                    
                    st.pyplot(fig)
                    
                    st.markdown(f"""
                    **🦷 Diagnóstico por Segmentación (U-Net):** - Resultado: `{pred_diag}`   
                    - Clase: `{pred_class}`   
                    - Precisión estimada: 92%   

                    **🧠 Diagnóstico por Clasificación:** - Resultado: `{clf_diag}`   
                    - Clase: `{clf_class}`   
                    - Confianza del modelo: `{clf_conf:.2%}`   
                    """)
                    
                    # Formulario para observaciones
                    with st.form("diagnostico_form"):
                        obs = st.text_area("Observaciones adicionales", height=150)
                        nivel = st.slider("Nivel de severidad", 0, 6, pred_class)
                        
                        if st.form_submit_button("Guardar Diagnóstico"):
                            # Guardar imagen permanentemente
                            img_dir = "patient_images"
                            os.makedirs(img_dir, exist_ok=True)
                            new_path = os.path.join(img_dir, f"{paciente_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
                            original_img.save(new_path)
                            
                            # Guardar en base de datos
                            diagnosis_data = {
                                'version': 'v1.0',
                                'precision': 92.5,
                                'score': 90,
                                'resultado': pred_diag,
                                'nivel_enfe': nivel,
                                'observacion': obs,
                                'clasificacion_modelo': clf_diag,
                                'clasificacion_nivel': clf_class,
                                'clasificacion_confianza': clf_conf
                            }
                            
                            if DBOperations.save_diagnosis(paciente_id, st.session_state.user['id_usuario'], new_path, diagnosis_data):
                                st.success("Diagnóstico guardado exitosamente")
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                st.rerun()
                            else:
                                st.error("Error al guardar el diagnóstico")
                except Exception as e:
                    st.error(f"Error al procesar la imagen: {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.warning("No hay pacientes registrados en el sistema.")
    
    with tab2:
        st.subheader("Historial de Pacientes")
        pacientes = DBOperations.get_all_patients()
        
        if pacientes:
            # Formatear opciones para el selectbox
            opciones_pacientes = [
                f"{p['nom_usu']} {p['ape_usu']} (DNI: {p['dni_usu']})"
                for p in pacientes
            ]
            
            # Mapeo entre opción mostrada y ID de paciente
            paciente_dict = {
                opcion: p['id_paci']
                for opcion, p in zip(opciones_pacientes, pacientes)
            }
            
            paciente_hist = st.selectbox(
                "Seleccione paciente para ver historial",
                options=opciones_pacientes,
                key="hist_paciente"
            )
            
            paciente_id = paciente_dict[paciente_hist]
            historial = DBOperations.get_patient_history(paciente_id)
            
            if historial:
                for record in historial:
                    with st.expander(f"Consulta del {record['fecha_diagnostico']}"):
                        st.write(f"Diagnóstico: {DIAGNOSIS_LABELS.get(record['nivel_enfe'], 'Desconocido')}")
                        st.write(f"Observaciones: {record['observacion']}")
                        try:
                            img = load_img(record['ruta_arch'], target_size=(300, 300))
                            st.image(img, caption="Imagen de diagnóstico")
                        except:
                            st.warning("No se pudo cargar la imagen asociada")
            else:
                st.info("Este paciente no tiene registros en su historial")
        else:
            st.warning("No hay pacientes registrados")

# ---------------------- FLUJO PRINCIPAL ----------------------
def main():
    st.set_page_config(
        page_title="Sistema de Diagnóstico Dental",
        page_icon="🦷",
        layout="wide"
    )
    
    # Estado de la sesión
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'registrando' not in st.session_state:
        st.session_state.registrando = False
    
    # Mostrar página según estado
    if st.session_state.user is None:
        if st.session_state.registrando:
            register_page()
        else:
            login_page()
    else:
        # Barra lateral
        with st.sidebar:
            st.markdown(f"""
            **Usuario:** {st.session_state.user['nom_usu']}   
            **Rol:** {st.session_state.user['rol_usu'].capitalize()}   
            **DNI:** {st.session_state.user.get('dni_usu', 'N/A')}
            """)
            
            if st.button("🔒 Cerrar sesión"):
                st.session_state.user = None
                st.session_state.registrando = False
                st.rerun()
        
        # Mostrar dashboard según rol
        if st.session_state.user['rol_usu'] == 'admin':
            admin_dashboard()
        elif st.session_state.user['rol_usu'] == 'paciente':
            patient_dashboard()
        else:   # odontologo o tecnico
            professional_dashboard()

if __name__ == "__main__":
    main()