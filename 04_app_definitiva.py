import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import numpy as np
import xgboost as xgb
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="NEST - Endometrial Stratification Tool",
    page_icon="🔬",
    layout="wide"
)

# ==================== CARGA DEL MODELO Y CONFIGURACIÓN ====================

@st.cache_resource
def cargar_modelo_y_config():
    """
    Carga el modelo XGBoost pre-entrenado y su configuración con el mejor umbral
    """
    try:
        # Cargar modelo
        modelo = xgb.XGBClassifier()
        modelo.load_model('modelo_definitivo.json')
        
        # Cargar configuración con el mejor umbral
        try:
            with open('modelo_config.json', 'r') as f:
                config = json.load(f)
            mejor_umbral = config['mejor_umbral']
            metricas = config['metricas_test']
            st.success(f"✅ Modelo XGBoost cargado correctamente")
            st.info(f"🎯 Umbral óptimo: {mejor_umbral:.2f} | Sensibilidad: {metricas['sensibilidad']:.2%} | Accuracy: {metricas['accuracy']:.2%}")
            return modelo, config
        except FileNotFoundError:
            st.warning("⚠️ No se encontró 'modelo_config.json'. Usando umbral por defecto (0.5)")
            return modelo, {'mejor_umbral': 0.5, 'metricas_test': None}
            
    except FileNotFoundError:
        st.error("❌ No se encontró 'modelo_definitivo.json' en la carpeta de la aplicación.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None, None

# Cargar modelo y configuración al inicio
modelo, config = cargar_modelo_y_config()
UMBRAL_OPTIMO = config['mejor_umbral'] if config else 0.5

# ==================== MAPEOS DE VARIABLES ====================

# Mapeos para convertir selecciones del usuario a valores numéricos
TIPO_HISTOLOGICO_MAP = {
    "Endometrioide": 0,
    "Seroso": 1,
    "Células claras": 2,
    "Carcinosarcoma": 3,
    "Mixto": 4
}

GRADO_MAP = {
    "G1 (Bien diferenciado)": 1,
    "G2 (Moderadamente diferenciado)": 2,
    "G3 (Pobremente diferenciado)": 3
}

INFILTRACION_MAP = {
    "Sin infiltración": 0,
    "< 50%": 1,
    "≥ 50%": 2
}

AFECTACION_LINFO_MAP = {
    "Negativa": 0,
    "Indeterminada": 1,
    "Positiva": 2
}

RECEPTORES_MAP = {
    "Negativos": 0,
    "No disponible": 1,
    "Positivos": 2
}

ESTADIO_FIGO_MAP = {
    "IA": 1, "IB": 2, "II": 3, "IIIA": 4, "IIIB": 5,
    "IIIC1": 6, "IIIC2": 7, "IVA": 8, "IVB": 9
}

BINARY_MAP = {
    "No": 0,
    "Sí": 1,
    "Desconocido": -1
}

# ==================== FUNCIÓN PARA PREPARAR FEATURES ====================

def preparar_features_modelo(datos_clinicos):
    """
    Prepara las 39 features que el modelo XGBoost espera recibir
    
    IMPORTANTE: El orden y nombres deben coincidir exactamente con el entrenamiento
    """
    
    # Obtener fecha actual para features derivadas
    fecha_actual = datetime.now()
    fecha_diagnostico = datos_clinicos.get('fecha_diagnostico', fecha_actual)
    fecha_nacimiento = datos_clinicos.get('fecha_nacimiento', 
                                          fecha_actual.replace(year=fecha_actual.year - datos_clinicos['edad']))
    
    # Crear diccionario con todas las features
    features_dict = {
        # Features clínicas básicas
        "edad": datos_clinicos['edad'],
        "imc": datos_clinicos['imc'],
        "tipo_histologico": TIPO_HISTOLOGICO_MAP.get(datos_clinicos['tipo_histologico'], 0),
        "grado_histologi": GRADO_MAP.get(datos_clinicos['grado'], 1),
        "infiltracion_mi": INFILTRACION_MAP.get(datos_clinicos['infiltracion_miometrial'], 0),
        
        # Features de ecografía (valores por defecto si no están disponibles)
        "ecotv_infiltobj": datos_clinicos.get('ecotv_infiltobj', 0),
        "ecotv_infiltsub": datos_clinicos.get('ecotv_infiltsub', 0),
        
        # Metástasis
        "metasta_distan": BINARY_MAP.get(datos_clinicos.get('metastasis_distante', 'No'), 0),
        
        # Grupo de riesgo y estadiaje
        "grupo_riesgo": datos_clinicos.get('grupo_riesgo_inicial', 1),  # 1=bajo, 2=intermedio, 3=alto
        "estadiaje_pre_i": ESTADIO_FIGO_MAP.get(datos_clinicos['estadio_figo'], 1),
        
        # Tratamientos
        "tto_NA": BINARY_MAP.get(datos_clinicos.get('tto_neoadyuvante', 'No'), 0),
        "tto_1_quirugico": BINARY_MAP.get(datos_clinicos.get('tto_quirurgico', 'Sí'), 1),
        "asa": datos_clinicos.get('asa_score', 2),  # Score ASA (1-5)
        
        # Histología definitiva
        "histo_defin": TIPO_HISTOLOGICO_MAP.get(datos_clinicos['tipo_histologico'], 0),
        "tamano_tumoral": datos_clinicos['tamano_tumoral'],
        "afectacion_linf": AFECTACION_LINFO_MAP.get(datos_clinicos['afectacion_linfovascular'], 0),
        
        # Ganglio centinela
        "AP_centinela_pelvico": BINARY_MAP.get(datos_clinicos.get('centinela_positivo', 'No'), 0),
        
        # Marcadores moleculares
        "beta_cateninap": BINARY_MAP.get(datos_clinicos.get('beta_catenina', 'Desconocido'), -1),
        "mut_pole": BINARY_MAP.get(datos_clinicos.get('mutacion_pole', 'Desconocido'), -1),
        "p53_ihq": RECEPTORES_MAP.get(datos_clinicos.get('p53_ihq', 'No disponible'), 1),
        
        # Estadiaje y riesgo definitivos
        "FIGO2023": ESTADIO_FIGO_MAP.get(datos_clinicos['estadio_figo'], 1),
        "grupo_de_riesgo_definitivo": datos_clinicos.get('grupo_riesgo_definitivo', 1),
        
        # Tratamientos adyuvantes
        "Tributaria_a_Radioterapia": BINARY_MAP.get(datos_clinicos.get('tributaria_rt', 'No'), 0),
        "bqt": BINARY_MAP.get(datos_clinicos.get('braquiterapia', 'No'), 0),
        "qt": BINARY_MAP.get(datos_clinicos.get('quimioterapia', 'No'), 0),
        "Tratamiento_sistemico_realizad": BINARY_MAP.get(datos_clinicos.get('tto_sistemico', 'No'), 0),
    
        
        # Features derivadas de fechas
        "FN_year": fecha_nacimiento.year,
        "FN_month": fecha_nacimiento.month,
        "FN_day": fecha_nacimiento.day,
        
        "f_diag_year": fecha_diagnostico.year,
        "f_diag_month": fecha_diagnostico.month,
        "f_diag_day": fecha_diagnostico.day,
        
        "Ultima_fecha_year": fecha_actual.year,
        "Ultima_fecha_month": fecha_actual.month,
        "Ultima_fecha_day": fecha_actual.day
    }
    
    # Crear DataFrame con el orden exacto de las features
    clinical_features = [
        "edad", "imc", "tipo_histologico", "grado_histologi", "infiltracion_mi",
        "ecotv_infiltobj", "ecotv_infiltsub", "metasta_distan", "grupo_riesgo",
        "estadiaje_pre_i", "tto_NA", "tto_1_quirugico", "asa", "histo_defin",
        "tamano_tumoral", "afectacion_linf", "AP_centinela_pelvico", "beta_cateninap",
        "mut_pole", "p53_ihq", "FIGO2023", "grupo_de_riesgo_definitivo",
        "Tributaria_a_Radioterapia", "bqt", "qt", "Tratamiento_sistemico_realizad",
        "FN_year", "FN_month", "FN_day", "f_diag_year", "f_diag_month", "f_diag_day", 
        "Ultima_fecha_year", "Ultima_fecha_month", "Ultima_fecha_day"
    ]
    
    # Crear DataFrame con las features en el orden correcto
    X = pd.DataFrame([features_dict])[clinical_features]
    
    return X

# ==================== FUNCIÓN DE PREDICCIÓN CON UMBRAL ÓPTIMO ====================

def predecir_con_modelo(datos_clinicos, modelo, umbral=UMBRAL_OPTIMO):
    """
    Realiza la predicción usando el modelo XGBoost con el umbral óptimo
    """
    if modelo is None:
        return None, None, None
    
    try:
        X = preparar_features_modelo(datos_clinicos)
        
        # Predecir probabilidades
        probabilidades = modelo.predict_proba(X)[0]
        
        prob_no_recidiva = probabilidades[0]
        prob_recidiva = probabilidades[1]
        
        # Aplicar el umbral óptimo para la clasificación
        prediccion_clase = 1 if prob_recidiva >= umbral else 0
        
        return prob_recidiva, prob_no_recidiva, prediccion_clase
        
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")
        st.error("Verifica que todas las features estén correctamente definidas")
        return None, None, None

# ==================== MODELO BACKUP ====================

def logistic(x):
    return 1 / (1 + np.exp(-x))

def modelo_predictivo_backup(score):
    """Modelo estadístico de backup"""
    recidiva_alpha = -3.0
    recidiva_beta = 0.35
    exitus_alpha = -4.0
    exitus_beta = 0.30
    
    prob_recidiva = logistic(recidiva_alpha + recidiva_beta * score)
    prob_exitus = logistic(exitus_alpha + exitus_beta * score)
    
    return prob_recidiva, prob_exitus

# ==================== INTERFAZ STREAMLIT ====================

st.title("🔬 NEST - NSMP Endometrial Stratification Tool")
st.markdown("### Calculadora de Riesgo para Cáncer Endometrial NSMP")

# Indicador de estado del modelo
if modelo is not None:
    col_status1, col_status2 = st.columns([2, 1])
    with col_status1:
        st.success("🤖 Modelo XGBoost activo (39 features clínicas)")
    with col_status2:
        if config and config.get('metricas_test'):
            with st.expander("📊 Ver métricas del modelo"):
                metricas = config['metricas_test']
                st.metric("Sensibilidad", f"{metricas['sensibilidad']:.2%}")
                st.metric("Accuracy", f"{metricas['accuracy']:.2%}")
                st.metric("Precisión", f"{metricas['precision']:.2%}")
                st.metric("F1-Score", f"{metricas['f1_score']:.2%}")
                st.metric("Umbral Óptimo", f"{UMBRAL_OPTIMO:.2f}")
else:
    st.warning("⚠️ Usando modelo estadístico de backup")

st.markdown("---")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📊 Diagnóstico y Análisis", "🤖 Interpretación IA", "🔢 Cálculos Detallados"])

# ==================== PESTAÑA 1 ====================
with tab1:
    st.header("📋 Evaluación Clínica")
    
    with st.expander("ℹ️ Información del Proyecto"):
        st.markdown("""
        **NEST** es una herramienta de estratificación pronóstica para pacientes con cáncer endometrial 
        de perfil molecular **NSMP (No Specific Molecular Profile)**.
        
        El modelo de Machine Learning integra 39 variables clínicas, histopatológicas y moleculares 
        para predecir el riesgo de recidiva con un umbral optimizado para maximizar la sensibilidad.
        """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos Demográficos")
        edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=65)
        imc = st.number_input("IMC (kg/m²)", min_value=15.0, max_value=60.0, value=28.0)
        fecha_nacimiento = st.date_input("Fecha de Nacimiento", 
                                         value=datetime.now().replace(year=datetime.now().year - edad))
        fecha_diagnostico = st.date_input("Fecha de Diagnóstico", value=datetime.now())
        
        st.subheader("Características Histológicas")
        tipo_histologico = st.selectbox("Tipo Histológico",
            ["Endometrioide", "Seroso", "Células claras", "Carcinosarcoma", "Mixto"])
        
        grado = st.selectbox("Grado Histológico",
            ["G1 (Bien diferenciado)", "G2 (Moderadamente diferenciado)", "G3 (Pobremente diferenciado)"])
        
        tamano_tumoral = st.number_input("Tamaño Tumoral (cm)", min_value=0.1, max_value=20.0, value=3.0)
    
    with col2:
        st.subheader("Factores Pronósticos")
        
        infiltracion_miometrial = st.selectbox("Infiltración Miometrial",
            ["< 50%", "≥ 50%", "Sin infiltración"])
        
        afectacion_linfovascular = st.selectbox("Afectación Linfovascular",
            ["Negativa", "Positiva", "Indeterminada"])
        
        receptores_estrogenos = st.selectbox("Receptores de Estrógenos",
            ["Positivos", "Negativos", "No disponible"])
        
        receptores_progesterona = st.selectbox("Receptores de Progesterona",
            ["Positivos", "Negativos", "No disponible"])
        
        estadio_figo = st.selectbox("Estadio FIGO",
            ["IA", "IB", "II", "IIIA", "IIIB", "IIIC1", "IIIC2", "IVA", "IVB"])
    
    # Sección expandible para features adicionales
    with st.expander("🔬 Marcadores Moleculares y Datos Adicionales (Opcional)"):
        col3, col4 = st.columns(2)
        
        with col3:
            mutacion_pole = st.selectbox("Mutación POLE", ["Desconocido", "No", "Sí"])
            beta_catenina = st.selectbox("Beta-catenina", ["Desconocido", "No", "Sí"])
            p53_ihq = st.selectbox("p53 IHQ", ["No disponible", "Positivos", "Negativos"])
            centinela_positivo = st.selectbox("Ganglio Centinela Positivo", ["No", "Sí"])
        
        with col4:
            metastasis_distante = st.selectbox("Metástasis a Distancia", ["No", "Sí"])
            tto_neoadyuvante = st.selectbox("Tratamiento Neoadyuvante", ["No", "Sí"])
            braquiterapia = st.selectbox("Braquiterapia Realizada", ["No", "Sí"])
            quimioterapia = st.selectbox("Quimioterapia Realizada", ["No", "Sí"])
    
    st.markdown("---")
    
    if 'calcular_presionado' not in st.session_state:
        st.session_state.calcular_presionado = False
    
    if st.button("🔍 Calcular Score de Riesgo con IA", type="primary", use_container_width=True):
        st.session_state.calcular_presionado = True
        
        # Calcular score tradicional
        score = 0
        factores_riesgo = []
        detalles_calculo = []
        
        # Sistema de puntuación tradicional (simplificado)
        if edad > 70:
            score += 2
            factores_riesgo.append("Edad > 70 años (+2)")
            detalles_calculo.append({"Factor": "Edad", "Valor": edad, "Puntos": 2})
        elif edad > 60:
            score += 1
            factores_riesgo.append("Edad > 60 años (+1)")
            detalles_calculo.append({"Factor": "Edad", "Valor": edad, "Puntos": 1})
        
        if imc >= 35:
            score += 2
            factores_riesgo.append("IMC ≥ 35 (+2)")
            detalles_calculo.append({"Factor": "IMC", "Valor": imc, "Puntos": 2})
        elif imc >= 30:
            score += 1
            factores_riesgo.append("IMC ≥ 30 (+1)")
            detalles_calculo.append({"Factor": "IMC", "Valor": imc, "Puntos": 1})
        
        if "G3" in grado:
            score += 3
            factores_riesgo.append("Grado G3 (+3)")
            detalles_calculo.append({"Factor": "Grado", "Valor": "G3", "Puntos": 3})
        elif "G2" in grado:
            score += 1
            factores_riesgo.append("Grado G2 (+1)")
            detalles_calculo.append({"Factor": "Grado", "Valor": "G2", "Puntos": 1})
        
        if "≥ 50%" in infiltracion_miometrial:
            score += 3
            factores_riesgo.append("Infiltración ≥ 50% (+3)")
            detalles_calculo.append({"Factor": "Infiltración", "Valor": "≥50%", "Puntos": 3})
        
        if afectacion_linfovascular == "Positiva":
            score += 3
            factores_riesgo.append("Afectación linfovascular (+3)")
            detalles_calculo.append({"Factor": "Afectación Linfo", "Valor": "Positiva", "Puntos": 3})
        
        # Clasificación tradicional
        if score <= 5:
            categoria_riesgo = "BAJO"
            riesgo_recidiva = "< 15%"
            recomendacion = "Vigilancia activa"
        elif score <= 10:
            categoria_riesgo = "INTERMEDIO"
            riesgo_recidiva = "15-30%"
            recomendacion = "Considerar braquiterapia vaginal"
        else:
            categoria_riesgo = "ALTO"
            riesgo_recidiva = "> 30%"
            recomendacion = "Terapia adyuvante recomendada"
        
        # Preparar datos para el modelo
        datos_clinicos = {
            "edad": edad,
            "imc": imc,
            "tipo_histologico": tipo_histologico,
            "grado": grado,
            "tamano_tumoral": tamano_tumoral,
            "infiltracion_miometrial": infiltracion_miometrial,
            "afectacion_linfovascular": afectacion_linfovascular,
            "receptores_estrogenos": receptores_estrogenos,
            "receptores_progesterona": receptores_progesterona,
            "estadio_figo": estadio_figo,
            "fecha_nacimiento": fecha_nacimiento,
            "fecha_diagnostico": fecha_diagnostico,
            # Features opcionales
            "mutacion_pole": mutacion_pole,
            "beta_catenina": beta_catenina,
            "p53_ihq": p53_ihq,
            "centinela_positivo": centinela_positivo,
            "metastasis_distante": metastasis_distante,
            "tto_neoadyuvante": tto_neoadyuvante,
            "braquiterapia": braquiterapia,
            "quimioterapia": quimioterapia
        }
        
        
        # Predicción con modelo XGBoost o backup
        if modelo is not None:
            prob_recidiva, prob_no_recidiva, clase_predicha = predecir_con_modelo(datos_clinicos, modelo, UMBRAL_OPTIMO)
            
            if prob_recidiva is not None:
                st.session_state.predicciones = {
                    "prob_recidiva": prob_recidiva,
                    "prob_no_recidiva": prob_no_recidiva,
                    "clase_predicha": clase_predicha,
                    "umbral_usado": UMBRAL_OPTIMO,
                    "modelo_usado": "XGBoost (39 features + umbral óptimo)"
                }
            else:
                # Fallback al modelo estadístico
                prob_recidiva, prob_exitus = modelo_predictivo_backup(score)
                st.session_state.predicciones = {
                    "prob_recidiva": prob_recidiva,
                    "prob_exitus": prob_exitus,
                    "clase_predicha": 1 if prob_recidiva >= 0.5 else 0,
                    "umbral_usado": 0.5,
                    "modelo_usado": "Estadístico (backup)"
                }
        else:
            prob_recidiva, prob_exitus = modelo_predictivo_backup(score)
            st.session_state.predicciones = {
                "prob_recidiva": prob_recidiva,
                "prob_exitus": prob_exitus,
                "clase_predicha": 1 if prob_recidiva >= 0.5 else 0,
                "umbral_usado": 0.5,
                "modelo_usado": "Estadístico (backup)"
            }
        
        # Guardar en session_state
        st.session_state.score = score
        st.session_state.categoria_riesgo = categoria_riesgo
        st.session_state.riesgo_recidiva = riesgo_recidiva
        st.session_state.factores_riesgo = factores_riesgo
        st.session_state.recomendacion = recomendacion
        st.session_state.detalles_calculo = detalles_calculo
        st.session_state.datos_clinicos = datos_clinicos
    
    # Mostrar resultados
    if st.session_state.calcular_presionado:
        st.markdown("---")
        st.header("📊 Resultados del Análisis")
        
        # Mostrar qué modelo se usó
        st.info(f"🤖 **Modelo utilizado:** {st.session_state.predicciones['modelo_usado']}")
        st.info(f"🎯 **Umbral de decisión:** {st.session_state.predicciones['umbral_usado']:.2f}")
        
        # Resultados del modelo predictivo
        st.subheader("🔮 Predicción de Recidiva con Machine Learning")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Probabilidad de Recidiva (ML)",
                     f"{st.session_state.predicciones['prob_recidiva']*100:.1f}%")
        
        with col2:
            st.metric("Clasificación ML",
                     "RECIDIVA" if st.session_state.predicciones['clase_predicha'] == 1 else "NO RECIDIVA",
                     delta="Alto riesgo" if st.session_state.predicciones['clase_predicha'] == 1 else "Bajo riesgo",
                     delta_color="inverse")
        
        with col3:
            st.metric("Score Tradicional", st.session_state.score)
        
        with col4:
            st.metric("Categoría", st.session_state.categoria_riesgo)
        
        # Interpretación
        prob_rec = st.session_state.predicciones['prob_recidiva']
        clase = st.session_state.predicciones['clase_predicha']
        
        if clase == 0:
            st.success(f"🟢 **Bajo riesgo de recidiva** según el modelo de Machine Learning (probabilidad: {prob_rec*100:.1f}%)")
        else:
            st.error(f"🔴 **Alto riesgo de recidiva detectado** según el modelo de Machine Learning (probabilidad: {prob_rec*100:.1f}%)")
        
        # Gráfico de probabilidades
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob_rec * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de Recidiva (%)", 'font': {'size': 24}},
            delta = {'reference': UMBRAL_OPTIMO * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
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
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.caption(f"Línea roja indica el umbral óptimo de decisión: {UMBRAL_OPTIMO*100:.0f}%")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones Terapéuticas")
        st.info(st.session_state.recomendacion)
        
        st.warning("""
        ⚠️ **Nota**: Esta herramienta está en fase de validación. 
        Los resultados deben interpretarse en contexto clínico completo.
        El umbral ha sido optimizado para maximizar la sensibilidad (detección de recidivas).
        """)

# ==================== PESTAÑA 2 ====================
with tab2:
    st.header("🤖 Interpretación Asistida por IA")
    
    if not st.session_state.calcular_presionado:
        st.info("👈 Calcula primero el score en la pestaña 'Diagnóstico y Análisis'")
    else:
        st.markdown(f"""
        ### 📋 Análisis del Modelo de Machine Learning
        
        El modelo XGBoost ha analizado **39 variables clínicas** para generar una predicción personalizada:
        
        - **Probabilidad de recidiva:** {st.session_state.predicciones['prob_recidiva']*100:.1f}%
        - **Modelo utilizado:** {st.session_state.predicciones['modelo_usado']}
        - **Categoría de riesgo tradicional:** {st.session_state.categoria_riesgo}
        
        ### 🔬 Variables Clave Analizadas
        
        El modelo ha considerado:
        - **Factores demográficos:** Edad, IMC
        - **Características tumorales:** Tipo histológico, grado, tamaño
        - **Invasión:** Infiltración miometrial, afectación linfovascular
        - **Marcadores moleculares:** POLE, p53, beta-catenina
        - **Estadificación:** FIGO 2023, ganglios, metástasis
        - **Tratamientos:** Cirugía, adyuvancia, quimioterapia
        
        ### 📊 Interpretación
        
        {chr(10).join(['- ' + f for f in st.session_state.factores_riesgo]) if st.session_state.factores_riesgo else 'Perfil de bajo riesgo'}
        
        ### 🎯 Recomendación Clínica
        
        {st.session_state.recomendacion}
        """)

# ==================== PESTAÑA 3 ====================
with tab3:
    st.header("🔢 Detalles Técnicos del Modelo")
    
    if not st.session_state.calcular_presionado:
        st.info("👈 Calcula primero el score en la pestaña 'Diagnóstico y Análisis'")
    else:
        st.markdown("### 📊 Features Utilizadas por el Modelo")
        
        st.markdown("""
        El modelo XGBoost utiliza **39 features** organizadas en:
        
        1. **Variables demográficas** (2): edad, imc
        2. **Histopatología** (5): tipo, grado, infiltración, tamaño, afectación linfovascular
        3. **Ecografía** (2): infiltración objetiva y subjetiva
        4. **Estadificación** (3): grupo de riesgo, estadio pre-IQ, FIGO2023
        5. **Tratamientos** (7): neoadyuvante, quirúrgico, ASA, RT, BQT, QT, sistémico
        6. **Marcadores moleculares** (3): POLE, p53, beta-catenina
        7. **Features temporales** (9): derivadas de fechas de nacimiento, diagnóstico y última visita
        8. **Otras** (5): histología definitiva, centinela, grupo riesgo definitivo, etc.
        """)
        
        # Tabla de score tradicional
        st.markdown("---")
        st.markdown("### 📏 Score Tradicional (Complementario)")
        
        df_calculo = pd.DataFrame(st.session_state.detalles_calculo)
        st.dataframe(df_calculo, use_container_width=True)
        
        st.markdown(f"**Score Total Tradicional:** {st.session_state.score} puntos")
        
        # Comparación
        st.markdown("---")
        st.markdown("### ⚖️ Comparación de Métodos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Método Tradicional (Score)")
            st.markdown(f"- **Puntuación:** {st.session_state.score}")
            st.markdown(f"- **Categoría:** {st.session_state.categoria_riesgo}")
            st.markdown(f"- **Riesgo estimado:** {st.session_state.riesgo_recidiva}")
        
        with col2:
            st.markdown("#### Modelo Machine Learning")
            st.markdown(f"- **Features:** 39 variables")
            st.markdown(f"- **Probabilidad:** {st.session_state.predicciones['prob_recidiva']}")
