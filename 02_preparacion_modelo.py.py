import pandas as pd
import numpy as np

# 1. CARGAR DATASET
df = pd.read_csv('Dataset_NEST_Completo.csv')
print(f"📊 Pacientes iniciales: {len(df)}")

# --- PASO 1: EL FILTRO NSMP (CRÍTICO PARA EL RETO) ---
# El reto NEST es solo para NSMP. Debemos excluir POLE, MMRd y p53abn.
# Basado en la lógica de exclusión [cite: 22, 28]

# NOTA: Revisa en tu excel cómo están escritos estos valores (ej. 1/0, Positivo/Negativo).
# Aquí asumo que 1 = Alterado/Mutado y 0 = Normal/WildType, o códigos numéricos del Excel.
# Ajusta los valores de comparación según lo que veas al imprimir 'unique()'.

print("\n--- Verificando valores moleculares ---")
print("Valores en p53:", df['p53_ihq'].unique())
print("Valores en POLE:", df['mut_pole'].unique())

# Lógica de filtrado (Adaptar según tus datos reales):
# NSMP se define por: POLE no mutado, MMR estable (no deficiente), p53 wild-type.
# Supongamos que en tu excel '1' es alterado y '0' (o 2) es normal.
# EJEMPLO GENÉRICO (TÚ DEBES AJUSTAR LA CONDICIÓN if):
# df_nsmp = df[ (df['mut_pole'] != 1) & (df['p53_ihq'] != 1) ]

# Si no tienes claro los códigos, por ahora usaremos TODO el dataset pero crearemos
# una columna 'es_nsmp' para que la IA aprenda a distinguirlos.
df['posible_NSMP'] = np.where((df['p53_ihq'].astype(str).str.contains('0|Normal', case=False)) & 
                              (df['mut_pole'].astype(str).str.contains('0|No', case=False)), 1, 0)

# Para la hackathon, si tienes pocas pacientes, a veces es mejor NO filtrar filas 
# para no quedarte con muy pocos datos, pero usar la variable molecular como predictor.
print("⚠️ Vamos a usar todas las pacientes pero controlando las variables moleculares.")
df_clean = df.copy()


# --- PASO 2: RELLENAR HUECOS (IMPUTACIÓN) ---
print("\n🧹 Rellenando valores vacíos...")

# A) Variables Numéricas -> Usamos la MEDIANA (menos sensible a valores extremos)
cols_numericas = ['edad', 'imc', 'tamano_tumoral', 'recep_est_porcent']
for col in cols_numericas:
    if col in df_clean.columns:
        mediana = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(mediana)

# B) Variables Categóricas -> Usamos un valor fijo (-1) o la Moda
# Es vital convertir las categorías a números.
cols_categoricas = ['grado_histologi', 'infiltracion_mi', 'afectacion_linf', 'FIGO2023', 'p53_ihq']

for col in cols_categoricas:
    if col in df_clean.columns:
        # 1. Rellenar NAs con "Desconocido" o -1
        df_clean[col] = df_clean[col].fillna(-1)
        
        # 2. Convertir a numérico forzado (si hay texto mezclado)
        # Esto convierte "Grado 1" -> 1, "Grado 3" -> 3
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(-1)

# --- PASO 3: PREPARAR EL TARGET (Y) Y LAS FEATURES (X) ---

# Definimos si la paciente es de "ALTO RIESGO" o "BAJO RIESGO"
# Opción A: Predecir Recurrencia (1 = Recae, 0 = No recae) -> Clasificación Binaria
target = 'recidiva'

# Matriz de características (X)
# Quitamos las columnas de respuesta (fechas, muerte, recidiva) para no hacer trampa
cols_drop = ['recidiva', 'estado', 'causa_muerte', 'fecha_qx', 'fecha_de_recidi', 
             'f_muerte', 'Ultima_fecha', 'DFS_fecha_fin', 'OS_fecha_fin', 
             'loc_recidiva_r01', 'DFS_MESES', 'OS_MESES']

# Nos aseguramos de borrar solo lo que existe
cols_drop_existentes = [c for c in cols_drop if c in df_clean.columns]
X = df_clean.drop(columns=cols_drop_existentes)
y = df_clean[target] # ¿Recayó?

# Guardamos también el tiempo para modelos avanzados (Cox)
y_time = df_clean['DFS_MESES'] if 'DFS_MESES' in df_clean.columns else None

print("\n--------------------------------")
print("✅ DATOS LISTOS PARA ENTRENAR")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print("--------------------------------")

# Guardar para el siguiente paso (Entrenamiento)
X.to_csv('X_train.csv', index=False)
y.to_csv('y_train.csv', index=False)
if y_time is not None: y_time.to_csv('y_time.csv', index=False)