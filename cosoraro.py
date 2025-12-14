import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
import numpy as np
import xgboost as xgb

# Configuración de la página
st.set_page_config(
    page_title="NEST - Endometrial Stratification Tool",
    layout="wide"
)

# Añadir tipografía formal
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'EB Garamond', serif !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'EB Garamond', serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'EB Garamond', serif !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        font-family: 'EB Garamond', serif !important;
    }
    
    .stTextInput label, .stSelectbox label, .stNumberInput label, .stDateInput label {
        font-family: 'EB Garamond', serif !important;
    }
    
    .stButton button {
        font-family: 'EB Garamond', serif !important;
        font-size: 1.1rem !important;
    }
    
    .stMetric label, .stMetric div {
        font-family: 'EB Garamond', serif !important;
    }
    
    div[data-testid="stMarkdownContainer"] {
        font-family: 'EB Garamond', serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CARGA DEL MODELO ====================

@st.cache_resource
def cargar_modelo_y_config():
    try:
        modelo = xgb.XGBClassifier()
        modelo.load_model('modelo_definitivo.json')
        
        try:
            with open('modelo_config.json', 'r') as f:
                config = json.load(f)
            mejor_umbral = config['mejor_umbral']
            metricas = config['metricas_test']
            st.success(f"Modelo XGBoost cargado correctamente")
            st.info(f"Umbral óptimo: {mejor_umbral:.2f} | Sensibilidad: {metricas['sensibilidad']:.2%} | Accuracy: {metricas['accuracy']:.2%}")
            return modelo, config
        except FileNotFoundError:
            st.warning("No se encontró 'modelo_config.json'. Usando umbral por defecto (0.5)")
            return modelo, {'mejor_umbral': 0.5, 'metricas_test': None}
            
    except FileNotFoundError:
        st.error("No se encontró 'modelo_definitivo.json'")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None

modelo, config = cargar_modelo_y_config()
UMBRAL_OPTIMO = config['mejor_umbral'] if config else 0.5

# ==================== FUNCIÓN PARA PREPARAR FEATURES ====================

def preparar_features_modelo(datos):
    """Prepara las 35 features exactas que espera el modelo"""
    
    # Lista exacta de las 35 features en orden
    clinical_features = [
        "edad", "imc", "tipo_histologico", "grado_histologi", "infiltracion_mi",
        "ecotv_infiltobj", "ecotv_infiltsub", "metasta_distan", "grupo_riesgo",
        "estadiaje_pre_i", "tto_NA", "tto_1_quirugico", "asa", "histo_defin",
        "tamano_tumoral", "afectacion_linf", "AP_centinela_pelvico", "beta_cateninap",
        "mut_pole", "p53_ihq", "FIGO2023", "grupo_de_riesgo_definitivo",
        "Tributaria_a_Radioterapia", "bqt", "qt", "Tratamiento_sistemico_realizad",
        "FN_year", "FN_month", "FN_day", 
        "f_diag_year", "f_diag_month", "f_diag_day", 
        "Ultima_fecha_year", "Ultima_fecha_month", "Ultima_fecha_day"
    ]
    
    # Crear DataFrame con las features
    X = pd.DataFrame([datos])[clinical_features]
    return X

# ==================== FUNCIÓN DE PREDICCIÓN ====================

def predecir_con_modelo(datos, modelo, umbral=UMBRAL_OPTIMO):
    if modelo is None:
        return None, None, None
    
    try:
        X = preparar_features_modelo(datos)
        probabilidades = modelo.predict_proba(X)[0]
        
        prob_no_recidiva = probabilidades[0]
        prob_recidiva = probabilidades[1]
        prediccion_clase = 1 if prob_recidiva >= umbral else 0
        
        return prob_recidiva, prob_no_recidiva, prediccion_clase
        
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None, None, None

# ==================== INTERFAZ STREAMLIT ====================

st.title("🔬 NEST - NSMP Endometrial Stratification Tool")
st.markdown("### Calculadora de Riesgo para Cáncer Endometrial NSMP")

if modelo is not None:
    st.success("Modelo XGBoost activo (35 features clínicas)")
else:
    st.error("No se pudo cargar el modelo")

st.markdown("---")

# ==================== FORMULARIO DE ENTRADA ====================

st.header("Datos de la Paciente")

with st.form("datos_paciente"):
    
    # ========== SECCIÓN 1: DATOS DEMOGRÁFICOS ==========
    st.subheader("Datos Demográficos")
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=65)
        imc = st.number_input("IMC (kg/m²)", min_value=15.0, max_value=60.0, value=28.0, step=0.1)
    
    with col2:
        FN = st.date_input("Fecha de Nacimiento (FN)", 
                          min_value = datetime.now().replace(year=datetime.now().year - 100), max_value= datetime.now(), value=datetime.now().replace(year=datetime.now().year - 65))
        f_diag = st.date_input("Fecha de Diagnóstico", value=datetime.now())
    
    st.markdown("---")
    
    # ========== SECCIÓN 2: HISTOLOGÍA INICIAL ==========
    st.subheader("Histología Inicial (Biopsia)")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        tipo_histologico = st.selectbox("Tipo Histológico", [
            "Hiperplasia con atípias",
            "Carcinoma endometroide",
            "Carcinoma seroso",
            "Carcinoma Celulas claras",
            "Carcinoma Indiferenciado",
            "Carcinoma Mixto",
            "Carcinoma Escamoso",
            "Carcinosarcoma",
            "Leiomiosarcoma",
            "Sarcoma de estroma endometrial",
            "Sarcoma indiferenciado",
            "Adenosarcoma",
            "Otros"
        ])
    
    with col4:
        Grado = st.selectbox("Grado", [
            "Bajo grado (G1-G2)",
            "Alto grado (G3)"
        ])
    
    with col5:
        grupo_riesgo = st.selectbox("Grupo de Riesgo Preoperatorio", [
            "Riesgo bajo",
            "Riego intermedio",
            "Riesgo alto"
        ])
    
    st.markdown("---")
    
    # ========== SECCIÓN 3: ECOGRAFÍA ==========
    st.subheader("Datos de Ecografía Transvaginal")
    col6, col7 = st.columns(2)
    
    with col6:
        ecotv_infiltobj = st.selectbox("Infiltración Miometrial - Método Objetivo (Karlsson)", [
            "No aplicado",
            "<50%",
            ">50%",
            "No valorable"
        ])
    
    with col7:
        ecotv_infiltsub = st.selectbox("Infiltración Miometrial - Método Subjetivo", [
            "No aplicado",
            "<50%",
            ">50%",
            "No valorable"
        ])
    
    st.markdown("---")
    
    # ========== SECCIÓN 4: ESTADIAJE PREOPERATORIO ==========
    st.subheader("Estadiaje Preoperatorio")
    col8, col9 = st.columns(2)
    
    with col8:
        estadiaje_pre_i = st.selectbox("Estadiaje Pre-Quirúrgico", [
            "Estadio I",
            "Estadio II",
            "Estadio III y IV"
        ])
    
    with col9:
        metasta_distan = st.selectbox("Metástasis a Distancia", ["No", "Si"])
    
    st.markdown("---")
    
    # ========== SECCIÓN 5: TRATAMIENTO ==========
    st.subheader("Tratamiento")
    col10, col11, col12 = st.columns(3)
    
    with col10:
        tto_NA = st.selectbox("Tratamiento Neoadyuvante", ["No", "Si"])
        tto_1_quirugico = st.selectbox("Tratamiento 1º Quirúrgico", ["Si", "No"])
    
    with col11:
        asa = st.selectbox("ASA Score", [
            "ASA 1",
            "ASA 2",
            "ASA 3",
            "ASA 4",
            "ASA 5",
            "ASA 6",
            "Desconocido"
        ])
    
    with col12:
        st.info("Clasificación ASA del estado físico")
    
    st.markdown("---")
    
    # ========== SECCIÓN 6: HISTOLOGÍA DEFINITIVA (POST-CIRUGÍA) ==========
    st.subheader("Histología Definitiva (Pieza Quirúrgica)")
    col13, col14, col15 = st.columns(3)
    
    with col13:
        histo_defin = st.selectbox("Tipo Histológico Definitivo", [
            "Hiperplasia con atipias",
            "Carcinoma endometrioide",
            "Carcinoma seroso",
            "Carcinoma de celulas claras",
            "Carcinoma Indiferenciado",
            "Carcinoma mixto",
            "Carcinoma escamoso",
            "Carcinosarcoma",
            "Otros"
        ])
    
    with col14:
        grado_histologi = st.selectbox("Grado Histológico Definitivo", [
            "Bajo grado (G1-G2)",
            "Alto grado (G3)"
        ])
    
    with col15:
        tamano_tumoral = st.number_input("Tamaño Tumoral (cm)", 
                                        min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    
    col16, col17 = st.columns(2)
    
    with col16:
        infiltracion_mi = st.selectbox("Infiltración Miometrial-Serosa", [
            "No infiltracion",
            "Infiltracion miometrial <50%",
            "Infiltracion miometrial >50%",
            "Infiltracion serosa"
        ])
    
    with col17:
        afectacion_linf = st.selectbox("Afectación Linfovascular", ["No", "Si"])
    
    st.markdown("---")
    
    # ========== SECCIÓN 7: GANGLIOS ==========
    st.subheader("Evaluación Ganglionar")
    
    AP_centinela_pelvico = st.selectbox("AP Centinela Pélvico", [
        "Negativo (pN0)",
        "Cels. tumorales aisladas (pN0(i+))",
        "Micrometastasis (pN1(mi))",
        "Macrometastasis (pN1)",
        "pNx"
    ])
    
    st.markdown("---")
    
    # ========== SECCIÓN 8: MARCADORES MOLECULARES ==========
    st.subheader("Marcadores Moleculares")
    col18, col19, col20 = st.columns(3)
    
    with col18:
        beta_cateninap = st.selectbox("Beta Catenina (Positividad nuclear)", [
            "No",
            "Si",
            "No realizado"
        ])
    
    with col19:
        mut_pole = st.selectbox("Mutación de POLE", [
            "Mutado",
            "No Mutado",
            "No realizado"
        ])
    
    with col20:
        p53_ihq = st.selectbox("p53 IHQ", [
            "Normal",
            "Anormal",
            "No realizada"
        ])
    
    st.markdown("---")
    
    # ========== SECCIÓN 9: ESTADIFICACIÓN DEFINITIVA ==========
    st.subheader("Estadificación y Grupo de Riesgo Definitivos")
    col21, col22 = st.columns(2)
    
    with col21:
        FIGO2023 = st.selectbox("Estadiaje Quirúrgico FIGO 2023", [
            "IA1", "IA2", "IA3", "IB", "IC",
            "IIA", "IIB", "IIC",
            "IIIA", "IIIB", "IIIC",
            "IVA", "IVB", "IVC"
        ])
    
    with col22:
        grupo_de_riesgo_definitivo = st.selectbox("Grupo de Riesgo Definitivo", [
            "Riesgo bajo",
            "Riesgo intermedio",
            "Riesgo intermedio-alto",
            "Riesgo alto",
            "Avanzados"
        ])
    
    st.markdown("---")
    
    # ========== SECCIÓN 10: TRATAMIENTO ADYUVANTE ==========
    st.subheader("Tratamiento Adyuvante")
    col23, col24, col25, col26 = st.columns(4)
    
    with col23:
        Tributaria_a_Radioterapia = st.selectbox("¿Tributaria a Radioterapia?", ["No", "Si"])
    
    with col24:
        bqt = st.selectbox("¿Tributaria a Braquiterapia?", ["No", "Si"])
    
    with col25:
        qt = st.selectbox("¿Tributaria a Tratamiento Sistémico?", ["No", "Si"])
    
    with col26:
        Tratamiento_sistemico_realizad = st.selectbox("Tratamiento Sistémico Realizado", [
            "No realizada",
            "Dosis parcial",
            "Dosis completa"
        ])
    
    st.markdown("---")
    
    # Botón de envío
    submitted = st.form_submit_button("Calcular Riesgo de Recidiva", 
                                     type="primary", 
                                     use_container_width=True)

# ==================== PROCESAMIENTO Y RESULTADOS ====================

if submitted:
    
    # Mapear valores a códigos numéricos según la tabla
    
    # Tipo histológico
    tipo_hist_map = {
        "Hiperplasia con atípias": 1, "Carcinoma endometroide": 2,
        "Carcinoma seroso": 3, "Carcinoma Celulas claras": 4,
        "Carcinoma Indiferenciado": 5, "Carcinoma Mixto": 6,
        "Carcinoma Escamoso": 7, "Carcinosarcoma": 8,
        "Leiomiosarcoma": 9, "Sarcoma de estroma endometrial": 10,
        "Sarcoma indiferenciado": 11, "Adenosarcoma": 12, "Otros": 88
    }
    
    # Grado
    grado_map = {"Bajo grado (G1-G2)": 1, "Alto grado (G3)": 2}
    
    # Grupo de riesgo preoperatorio
    grupo_riesgo_map = {"Riesgo bajo": 1, "Riego intermedio": 2, "Riesgo alto": 3}
    
    # Ecografía
    eco_map = {"No aplicado": 1, "<50%": 2, ">50%": 3, "No valorable": 4}
    
    # Estadiaje pre-IQ
    estadiaje_pre_map = {"Estadio I": 0, "Estadio II": 1, "Estadio III y IV": 2}
    
    # ASA
    asa_map = {"ASA 1": 0, "ASA 2": 1, "ASA 3": 2, "ASA 4": 3, 
               "ASA 5": 4, "ASA 6": 5, "Desconocido": 6}
    
    # Histología definitiva
    histo_def_map = {
        "Hiperplasia con atipias": 1, "Carcinoma endometrioide": 2,
        "Carcinoma seroso": 3, "Carcinoma de celulas claras": 4,
        "Carcinoma Indiferenciado": 5, "Carcinoma mixto": 6,
        "Carcinoma escamoso": 7, "Carcinosarcoma": 8, "Otros": 9
    }
    
    # Infiltración miometrial
    infiltracion_map = {
        "No infiltracion": 0,
        "Infiltracion miometrial <50%": 1,
        "Infiltracion miometrial >50%": 2,
        "Infiltracion serosa": 3
    }
    
    # AP centinela
    ap_cent_map = {
        "Negativo (pN0)": 0,
        "Cels. tumorales aisladas (pN0(i+))": 1,
        "Micrometastasis (pN1(mi))": 2,
        "Macrometastasis (pN1)": 3,
        "pNx": 4
    }
    
    # Marcadores moleculares
    beta_map = {"No": 0, "Si": 1, "No realizado": 2}
    pole_map = {"Mutado": 1, "No Mutado": 2, "No realizado": 3}
    p53_map = {"Normal": 1, "Anormal": 2, "No realizada": 3}
    
    # FIGO 2023
    figo_map = {
        "IA1": 1, "IA2": 2, "IA3": 3, "IB": 4, "IC": 5,
        "IIA": 6, "IIB": 7, "IIC": 8,
        "IIIA": 9, "IIIB": 10, "IIIC": 11,
        "IVA": 12, "IVB": 13, "IVC": 14
    }
    
    # Grupo riesgo definitivo
    grupo_def_map = {
        "Riesgo bajo": 1, "Riesgo intermedio": 2,
        "Riesgo intermedio-alto": 3, "Riesgo alto": 4, "Avanzados": 5
    }
    
    # Tratamiento sistémico realizado
    tto_sist_map = {"No realizada": 0, "Dosis parcial": 1, "Dosis completa": 2}
    
    # Binarios
    binary_map = {"No": 0, "Si": 1}
    
    # Fecha actual para última visita
    Ultima_fecha = datetime.now()
    
    # Crear diccionario con todas las 35 features
    datos_modelo = {
        "edad": edad,
        "imc": imc,
        "tipo_histologico": tipo_hist_map[tipo_histologico],
        "grado_histologi": grado_map[grado_histologi],
        "infiltracion_mi": infiltracion_map[infiltracion_mi],
        "ecotv_infiltobj": eco_map[ecotv_infiltobj],
        "ecotv_infiltsub": eco_map[ecotv_infiltsub],
        "metasta_distan": binary_map[metasta_distan],
        "grupo_riesgo": grupo_riesgo_map[grupo_riesgo],
        "estadiaje_pre_i": estadiaje_pre_map[estadiaje_pre_i],
        "tto_NA": binary_map[tto_NA],
        "tto_1_quirugico": binary_map[tto_1_quirugico],
        "asa": asa_map[asa],
        "histo_defin": histo_def_map[histo_defin],
        "tamano_tumoral": tamano_tumoral,
        "afectacion_linf": binary_map[afectacion_linf],
        "AP_centinela_pelvico": ap_cent_map[AP_centinela_pelvico],
        "beta_cateninap": beta_map[beta_cateninap],
        "mut_pole": pole_map[mut_pole],
        "p53_ihq": p53_map[p53_ihq],
        "FIGO2023": figo_map[FIGO2023],
        "grupo_de_riesgo_definitivo": grupo_def_map[grupo_de_riesgo_definitivo],
        "Tributaria_a_Radioterapia": binary_map[Tributaria_a_Radioterapia],
        "bqt": binary_map[bqt],
        "qt": binary_map[qt],
        "Tratamiento_sistemico_realizad": tto_sist_map[Tratamiento_sistemico_realizad],
        "FN_year": FN.year,
        "FN_month": FN.month,
        "FN_day": FN.day,
        "f_diag_year": f_diag.year,
        "f_diag_month": f_diag.month,
        "f_diag_day": f_diag.day,
        "Ultima_fecha_year": Ultima_fecha.year,
        "Ultima_fecha_month": Ultima_fecha.month,
        "Ultima_fecha_day": Ultima_fecha.day
    }
    
    # Realizar predicción
    prob_recidiva, prob_no_recidiva, clase_predicha = predecir_con_modelo(
        datos_modelo, modelo, UMBRAL_OPTIMO
    )
    
    if prob_recidiva is not None:
        st.markdown("---")
        st.header("Resultados del Análisis")
        
        st.info(f"🤖 **Modelo:** XGBoost (35 features) | 🎯 **Umbral:** {UMBRAL_OPTIMO:.2f}")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Probabilidad de Recidiva",
                     f"{prob_recidiva*100:.1f}%")
        
        with col_res2:
            st.metric("Probabilidad de No Recidiva",
                     f"{prob_no_recidiva*100:.1f}%")
        
        with col_res3:
            clasificacion = "ALTO RIESGO" if clase_predicha == 1 else "BAJO RIESGO"
            delta_color = "inverse" if clase_predicha == 1 else "normal"
            st.metric("Clasificación", clasificacion, 
                     delta="⚠️" if clase_predicha == 1 else "✅",
                     delta_color=delta_color)
        
        # Interpretación
        if clase_predicha == 0:
            st.success(f"🟢 **Bajo riesgo de recidiva** detectado (probabilidad: {prob_recidiva*100:.1f}%)")
            st.info("**Recomendación:** Vigilancia activa con seguimiento regular")
        else:
            st.error(f"🔴 **Alto riesgo de recidiva** detectado (probabilidad: {prob_recidiva*100:.1f}%)")
            st.warning("**Recomendación:** Considerar terapia adyuvante intensiva")
        
        # Gráfico gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob_recidiva * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Riesgo de Recidiva (%)", 'font': {'size': 24, 'family': 'EB Garamond, serif'}},
            delta = {'reference': UMBRAL_OPTIMO * 100},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, UMBRAL_OPTIMO * 100], 'color': '#90EE90'},
                    {'range': [UMBRAL_OPTIMO * 100, 100], 'color': '#FF6B6B'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': UMBRAL_OPTIMO * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=350,
            font=dict(family='EB Garamond, serif')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.warning("""
        ⚠️ **Nota Importante**: 
        - Esta herramienta está en fase de validación clínica
        - Los resultados deben interpretarse por un profesional médico
        - El umbral ha sido optimizado para maximizar la sensibilidad
        """)
