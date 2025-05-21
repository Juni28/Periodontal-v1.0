import sqlite3
import os

def inicializar_base_datos():
    # Conexión a la base de datos
    conn = sqlite3.connect("diagnostico_dental.db")
    cursor = conn.cursor()

    # Activar claves foráneas
    cursor.execute("PRAGMA foreign_keys = ON")

    # Crear tablas
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS Usuario (
        id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
        usu_usu TEXT UNIQUE NOT NULL,
        nom_usu TEXT NOT NULL,
        ape_usu TEXT NOT NULL,
        pass_usu TEXT NOT NULL,
        rol_usu TEXT NOT NULL CHECK(rol_usu IN ('admin', 'odontologo', 'paciente', 'tecnico')),
        dni_usu TEXT UNIQUE,
        edad_usu INTEGER,
        sexo_usu TEXT,
        fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS Historial (
        id_hist INTEGER PRIMARY KEY AUTOINCREMENT,
        id_paci INTEGER UNIQUE,
        FOREIGN KEY (id_paci) REFERENCES Paciente(id_paci) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Paciente (
        id_paci INTEGER PRIMARY KEY AUTOINCREMENT,
        id_usuario INTEGER UNIQUE,
        FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Imagen (
        id_ima INTEGER PRIMARY KEY AUTOINCREMENT,
        id_paci INTEGER NOT NULL,
        id_usu INTEGER NOT NULL,
        ruta_arch TEXT UNIQUE NOT NULL,
        tipo TEXT NOT NULL,
        fecha_subida DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (id_paci) REFERENCES Paciente(id_paci) ON DELETE CASCADE,
        FOREIGN KEY (id_usu) REFERENCES Usuario(id_usuario) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS ModeloIA (
        id_mod INTEGER PRIMARY KEY AUTOINCREMENT,
        id_ima INTEGER UNIQUE,
        version TEXT NOT NULL,
        precision REAL NOT NULL,
        score INTEGER NOT NULL,
        fecha_procesado DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (id_ima) REFERENCES Imagen(id_ima) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Segmentacion (
        id_seg INTEGER PRIMARY KEY AUTOINCREMENT,
        id_ima INTEGER UNIQUE,
        resultado TEXT NOT NULL,
        fecha_procesado DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (id_ima) REFERENCES Imagen(id_ima) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Diagnostico (
        id_diag INTEGER PRIMARY KEY AUTOINCREMENT,
        id_seg INTEGER UNIQUE,
        nivel_enfe INTEGER NOT NULL,
        observacion TEXT NOT NULL,
        id_paciente INTEGER NOT NULL,
        fecha_diagnostico DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (id_seg) REFERENCES Segmentacion(id_seg) ON DELETE CASCADE,
        FOREIGN KEY (id_paciente) REFERENCES Paciente(id_paci) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Informe (
        id_info INTEGER PRIMARY KEY AUTOINCREMENT,
        id_diag INTEGER UNIQUE,
        fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (id_diag) REFERENCES Diagnostico(id_diag) ON DELETE CASCADE
    );
    """)

    # Insertar datos iniciales
    try:
        cursor.execute("SELECT COUNT(*) FROM Usuario WHERE usu_usu = 'admin'")
        if cursor.fetchone()[0] == 0:
            # Insertar usuarios de ejemplo
            cursor.executemany("""
            INSERT INTO Usuario (usu_usu, nom_usu, ape_usu, pass_usu, rol_usu, dni_usu, edad_usu, sexo_usu)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", [
                ('admin', 'Admin', 'Sistema', 'admin123', 'admin', '00000000', 30, 'Masculino'),
                ('odontologo1', 'Juan', 'Pérez', 'odon123', 'odontologo', '11111111', 35, 'Masculino'),
                ('tecnico1', 'María', 'López', 'tec123', 'tecnico', '22222222', 28, 'Femenino'),
                ('paciente1', 'Carlos', 'Gómez', 'pac123', 'paciente', '33333333', 45, 'Masculino'),
                ('paciente2', 'Ana', 'Martínez', 'pac456', 'paciente', '44444444', 32, 'Femenino')
            ])

            # Insertar pacientes
            cursor.executemany("INSERT INTO Paciente (id_usuario) VALUES (?)", [
                (4,), (5,)
            ])

            # Insertar historiales
            cursor.executemany("INSERT INTO Historial (id_paci) VALUES (?)", [
                (1,), (2,)
            ])

        conn.commit()
        print("✅ Base de datos inicializada correctamente")
    except sqlite3.Error as e:
        print(f"❌ Error al inicializar la base de datos: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inicializar_base_datos()