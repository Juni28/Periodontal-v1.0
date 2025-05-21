import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = sqlite3.connect("diagnostico_dental.db")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class DBOperations:
    @staticmethod
    def get_user_by_username(username):
        with get_db_connection() as conn:
            return conn.execute(
                "SELECT * FROM Usuario WHERE usu_usu = ?", 
                (username,)
            ).fetchone()

    @staticmethod
    def get_all_patients():
        with get_db_connection() as conn:
            return conn.execute("""
                SELECT p.id_paci, u.nom_usu, u.ape_usu, u.dni_usu 
                FROM Paciente p
                JOIN Usuario u ON p.id_usuario = u.id_usuario
            """).fetchall()

    @staticmethod
    def save_diagnosis(patient_id, user_id, image_path, diagnosis_data):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Guardar imagen
                cursor.execute("""
                    INSERT INTO Imagen (id_paci, id_usu, ruta_arch, tipo)
                    VALUES (?, ?, ?, ?)
                """, (patient_id, user_id, image_path, 'diagnostico'))
                image_id = cursor.lastrowid
                
                # Guardar resultados del modelo IA
                cursor.execute("""
                    INSERT INTO ModeloIA (id_ima, version, precision, score)
                    VALUES (?, ?, ?, ?)
                """, (image_id, diagnosis_data['version'], diagnosis_data['precision'], diagnosis_data['score']))
                
                # Guardar segmentación
                cursor.execute("""
                    INSERT INTO Segmentacion (id_ima, resultado)
                    VALUES (?, ?)
                """, (image_id, diagnosis_data['resultado']))
                seg_id = cursor.lastrowid
                
                # Guardar diagnóstico
                cursor.execute("""
                    INSERT INTO Diagnostico (id_seg, nivel_enfe, observacion, id_paciente)
                    VALUES (?, ?, ?, ?)
                """, (seg_id, diagnosis_data['nivel_enfe'], diagnosis_data['observacion'], patient_id))
                
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error al guardar diagnóstico: {e}")
            return False

    @staticmethod
    def get_patient_history(patient_id):
        with get_db_connection() as conn:
            return conn.execute("""
                SELECT d.fecha_diagnostico, d.nivel_enfe, d.observacion, i.ruta_arch,
                       m.version as modelo_version, m.precision as modelo_precision
                FROM Diagnostico d
                JOIN Segmentacion s ON d.id_seg = s.id_seg
                JOIN Imagen i ON s.id_ima = i.id_ima
                JOIN ModeloIA m ON i.id_ima = m.id_ima
                WHERE d.id_paciente = ?
                ORDER BY d.fecha_diagnostico DESC
            """, (patient_id,)).fetchall()

    @staticmethod
    def create_user(username, password, nombre, apellido, dni, edad, sexo, rol):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO Usuario (usu_usu, nom_usu, ape_usu, pass_usu, rol_usu, dni_usu, edad_usu, sexo_usu)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (username, nombre, apellido, password, rol, dni, edad, sexo))
                
                if rol == 'paciente':
                    user_id = cursor.lastrowid
                    cursor.execute("INSERT INTO Paciente (id_usuario) VALUES (?)", (user_id,))
                    paci_id = cursor.lastrowid
                    cursor.execute("INSERT INTO Historial (id_paci) VALUES (?)", (paci_id,))
                
                conn.commit()
                return True
        except sqlite3.IntegrityError as e:
            print(f"Error al crear usuario: {e}")
            return False