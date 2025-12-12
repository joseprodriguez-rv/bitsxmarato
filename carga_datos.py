import pandas as pd
import numpy as np

# --- 1. CARGA DE DATOS ---
print("⏳ Cargando base de datos completa...")
# Usamos openpyxl para leer el Excel
df = pd.read_excel('datos.xlsx', engine='openpyxl')

# --- 2. SELECCIÓN EXTENDIDA DE VARIABLES (Basada en Leyenda y Proyecto NEST) ---
# Organizamos las variables por bloques para que sepas qué estás analizando.

cols_clinicas = [
    'edad', 
    'imc', 
    'asa',                  # Riesgo anestésico (refleja estado físico general)
    'valor_de_ca125'        # Biomarcador en sangre
]

cols_patologia_tumor = [
    'tipo_histologico',     # Tipo pre-qx
    'histo_defin',          # Tipo definitivo (El "Gold Standard")
    'grado_histologi',      # Grado de diferenciación (1, 2, 3)
    'tamano_tumoral',       # Tamaño en mm/cm
    'infiltracion_mi',      # ¿Cuánto invade el útero? (<50% o >50%)
    'afectacion_linf',      # Invasión linfovascular (LVSI) - CRÍTICO para riesgo
    'infilt_estr_cervix',   # ¿Invade el cuello uterino?
    'inf_param_vag',        # ¿Invade parametrios o vagina?
    'estadiaje_pre_i',      # Estadio antes de operar
    'FIGO2023'              # Estadio FIGO oficial (2018/2023)
]

cols_ganglios = [
    'AP_centinela_pelvico', # ¿Ganglio centinela positivo?
    'AP_ganPelv',           # ¿Ganglios pélvicos afectados?
    'AP_glanPaor',          # ¿Ganglios paraaórticos afectados?
    'n_GC_Afect',           # Número de ganglios centinelas afectados
]

cols_molecular = [
    'recep_est_porcent',    # Receptores Estrógenos (%) - CLAVE para NSMP
    'rece_de_Ppor',         # Receptores Progesterona (%)
    'p53_ihq',              # Inmunohistoquímica p53 (Wild type vs Mutated)
    'mut_pole',             # Mutación POLE (para descartar/confirmar grupo)
    'msh6', 'msh2', 'pms2', 'mlh1' # Proteínas MMR (para descartar inestabilidad)
]

cols_tratamiento = [
    'tto_1_quirugico',      # Tipo de cirugía
    'Tributaria_a_Radioterapia', # ¿Se indicó radio?
    'rdt',                  # ¿Recibió Radioterapia externa?
    'bqt',                  # ¿Recibió Braquiterapia?
    'qt',                   # ¿Recibió Quimioterapia?
    'Tratamiento_sistemico_realizad' # Detalle sistémico
]

cols_outcome = [
    'recidiva',             # TARGET 1: ¿Recayó? (Sí/No)
    'estado',               # TARGET 2: Estado vital (Vivo/Exitus)
    'causa_muerte',         # ¿Murió por cáncer o por otra cosa?
    'fecha_qx',             # Fecha base (cirugía)
    'fecha_de_recidi',      # Fecha del evento (recaída)
    'f_muerte',             # Fecha del evento (muerte)
    'Ultima_fecha',         # Fecha de último contacto (censura)
    'loc_recidiva_r01'      # Dónde recayó (local, pélvica, a distancia)
]

# Unimos todas las listas
todas_las_variables = cols_clinicas + cols_patologia_tumor + cols_ganglios + cols_molecular + cols_tratamiento + cols_outcome

# Filtramos para coger solo las que realmente existen en el Excel (por si algún nombre varía ligeramente)
cols_existentes = [c for c in todas_las_variables if c in df.columns]
missing_cols = [c for c in todas_las_variables if c not in df.columns]

print(f"✅ Variables encontradas: {len(cols_existentes)} de {len(todas_las_variables)}")
if missing_cols:
    print(f"⚠️ Advertencia: No encontré estas columnas (revisa nombres): {missing_cols}")

# Creamos el dataset limpio
df_nest = df[cols_existentes].copy()

# --- 3. INGENIERÍA DE VARIABLES (CÁLCULO DE TIEMPOS) ---
print("⚙️ Calculando tiempos de supervivencia...")

# Convertir a formato fecha
fechas_clave = ['fecha_qx', 'fecha_de_recidi', 'f_muerte', 'Ultima_fecha']
for col in fechas_clave:
    if col in df_nest.columns:
        df_nest[col] = pd.to_datetime(df_nest[col], errors='coerce')

# A) TIEMPO LIBRE DE ENFERMEDAD (Disease-Free Survival - DFS)
# Si recayó, fecha fin = recidiva. Si no, fecha fin = ultima visita.
if 'fecha_qx' in df_nest.columns:
    df_nest['DFS_fecha_fin'] = df_nest['fecha_de_recidi'].fillna(df_nest['Ultima_fecha'])
    df_nest['DFS_MESES'] = (df_nest['DFS_fecha_fin'] - df_nest['fecha_qx']).dt.days / 30.44

# B) SUPERVIVENCIA GLOBAL (Overall Survival - OS)
# Si murió, fecha fin = muerte. Si vive, fecha fin = ultima visita.
    df_nest['OS_fecha_fin'] = df_nest['f_muerte'].fillna(df_nest['Ultima_fecha'])
    df_nest['OS_MESES'] = (df_nest['OS_fecha_fin'] - df_nest['fecha_qx']).dt.days / 30.44

# Limpieza final: Eliminar filas sin datos de tiempo (errores de fecha)
df_nest = df_nest[df_nest['DFS_MESES'] > 0]

# --- 4. GUARDADO ---
nombre_salida = 'Dataset_NEST_Completo.csv'
df_nest.to_csv(nombre_salida, index=False)

print("\n------------------------------------------------")
print(f"🚀 ¡LISTO! Dataset completo guardado como: {nombre_salida}")
print(f"Dimensiones finales: {df_nest.shape[0]} pacientes x {df_nest.shape[1]} variables")
print("------------------------------------------------")
print("Primeras 5 filas del dataset listo para IA:")
print(df_nest[['recidiva', 'DFS_MESES', 'edad', 'FIGO2023', 'recep_est_porcent']].head())