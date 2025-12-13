import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="NEST Calculator | Hack the Uterus",
    page_icon="🧬",
    layout="wide"
)

# --- TÍTULO Y ESTILO ---
st.title("🧬 NEST: NSMP Endometrial Stratification Tool")
st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para estratificar el riesgo de recidiva en pacientes con 
Cáncer de Endometrio de perfil molecular **NSMP** (No Specific Molecular Profile).
""")
st.markdown("---")

# --- 1. CARGA DE DATOS Y ENTRENAMIENTO DEL MODELO ---
@st.cache_data
def cargar_y_entrenar():
    # Cargar el dataset
    try:
        # Asegúrate de que el CSV tiene los nombres de columna que has especificado
        df = pd.read_csv('Dataset_NEST_Final_Reclassificat.csv')
    except FileNotFoundError:
        st.error("⚠️ No se encuentra el archivo 'Dataset_NEST_Final_Reclassificat.csv'.")
        return None, None, None

    # Definir variables predictoras (FEATURES) según tu nueva lista
    # Seleccionamos las variables clínicas más relevantes para el cálculo de riesgo
    features = [
        'edad', 
        'imc', 
        'grado_histologi', 
        'infiltracion_mi', 
        'afectacion_linf', 
        'FIGO2023' 
    ]
    
    # Variable objetivo (TARGET)
    target = 'recidiva'
    
    # Verificación de seguridad: ¿Existen estas columnas?
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Faltan las siguientes columnas en tu CSV: {missing_cols}")
        st.write("Columnas detectadas en el archivo:", df.columns.tolist())
        return None, None, None

    # Preprocesamiento rápido: Eliminar filas con nulos en estas columnas específicas
    # (Aunque tu script anterior ya imputó datos, es una seguridad extra)
    df_model = df[features + [target]].dropna()

    X = df_model[features]
    y = df_model[target]

    # Entrenar Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, class_weight='balanced')
    clf.fit(X, y)

    return clf, features, df

model, feature_names, df_raw = cargar_y_entrenar()

if model is not None:
    # --- 2. BARRA LATERAL (INPUTS DEL MÉDICO) ---
    st.sidebar.header("📝 Datos de la Paciente")
    
    # 1. EDAD
    input_edad = st.sidebar.slider("Edad", min_value=30, max_value=95, value=65)
    
    # 2. IMC
    input_imc = st.sidebar.slider("IMC (Índice de Masa Corporal)", min_value=15.0, max_value=50.0, value=28.0)
    
    # 3. GRADO HISTOLÓGICO (Variable: grado_histologi)
    # 1: Bajo grado (G1-G2), 2: Alto grado (G3)
    opciones_grado = {"Bajo Grado (G1-G2)": 1, "Alto Grado (G3)": 2}
    label_grado = st.sidebar.selectbox("Grado Histológico", list(opciones_grado.keys()))
    input_grado = opciones_grado[label_grado]

    # 4. INFILTRACIÓN MIOMETRIAL (Variable: infiltracion_mi)
    # 0: No, 1: <50%, 2: >50%, 3: Serosa
    opciones_infilt = {
        "Sin infiltración": 0, 
        "Infiltración < 50%": 1, 
        "Infiltración > 50%": 2, 
        "Infiltración Serosa": 3
    }
    label_infilt = st.sidebar.selectbox("Infiltración Miometrial", list(opciones_infilt.keys()))
    input_infilt = opciones_infilt[label_infilt]

    # 5. AFECTACIÓN LINFOVASCULAR (Variable: afectacion_linf)
    # 0: No, 1: Sí
    opciones_linfo = {"No": 0, "Sí": 1}
    label_linfo = st.sidebar.selectbox("Afectación Linfovascular", list(opciones_linfo.keys()))
    input_linfo = opciones_linfo[label_linfo]

    # 6. ESTADIO FIGO 2023 (Variable: FIGO2023)
    # Mapeo aproximado basado en diccionarios estándar FIGO
    opciones_figo = {
        "IA1": 1, "IA2": 2, "IA3": 3,
        "IB": 4, "IC": 5,
        "IIA": 6, "IIB": 7, "IIC": 8,
        "IIIA": 9, "IIIB": 10, "IIIC": 11,
        "IVA": 12, "IVB": 13, "IVC": 14
    }
    label_figo = st.sidebar.selectbox("Estadio FIGO 2023", list(opciones_figo.keys()))
    input_figo = opciones_figo[label_figo]

    # Crear DataFrame con los inputs (¡Mismo orden y nombres que en el entrenamiento!)
    input_data = pd.DataFrame([[
        input_edad, 
        input_imc, 
        input_grado, 
        input_infilt, 
        input_linfo, 
        input_figo
    ]], columns=feature_names)

    # --- 3. PREDICCIÓN Y RESULTADOS ---
    st.subheader("Resultados del Modelo NEST")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        # Predecir probabilidad
        probabilidad = model.predict_proba(input_data)[0][1] # Probabilidad de clase 1 (Recidiva)
        percent_risk = round(probabilidad * 100, 2)

        # Semáforo de Riesgo
        if percent_risk < 15:
            color = "#2ecc71" # Verde
            riesgo_txt = "BAJO RIESGO"
            recomendacion = "Seguimiento estándar."
        elif percent_risk < 50:
            color = "#f1c40f" # Amarillo/Naranja
            riesgo_txt = "RIESGO INTERMEDIO"
            recomendacion = "Valorar adyuvancia (Braquiterapia/RT Externa)."
        else:
            color = "#e74c3c" # Rojo
            riesgo_txt = "ALTO RIESGO"
            recomendacion = "Considerar tratamiento sistémico (QT +/- RT)."

        # Tarjeta de Resultado
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border: 2px solid {color}; text-align: center;">
            <h4 style="margin:0; color: #555;">Probabilidad de Recidiva</h4>
            <h1 style="color:{color}; font-size: 60px; margin: 10px 0;">{percent_risk}%</h1>
            <h3 style="color:{color}; margin:0;">{riesgo_txt}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Sugerencia Clínica:** {recomendacion}")

    with col2:
        st.write("### Importancia de Variables")
        # Gráfico de Importancia
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.barh(range(len(indices)), importances[indices], color="#3498db")
        ax.set_yticks(range(len(indices)))
        # Mapeamos los nombres técnicos a nombres legibles para el gráfico
        nombres_legibles = {
            'edad': 'Edad', 'imc': 'IMC', 'grado_histologi': 'Grado',
            'infiltracion_mi': 'Infilt. Miometrial', 
            'afectacion_linf': 'Inv. Linfovascular', 'FIGO2023': 'Estadio FIGO'
        }
        labels = [nombres_legibles.get(feature_names[i], feature_names[i]) for i in indices]
        ax.set_yticklabels(labels)
        ax.set_xlabel("Peso en el modelo")
        st.pyplot(fig)

    # Debug: Mostrar datos raw (opcional)
    with st.expander("Ver Datos de Entrenamiento"):
        st.dataframe(df_raw.head())

else:
    st.warning("Esperando archivo de datos...")

# --- FOOTER ---
st.markdown("---")
st.caption("Hack the Uterus 2024 | Modelo Predictivo NSMP")