import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="NEST - Endometrial Stratification Tool",
    page_icon="🔬",
    layout="wide"
)

# ==================== MODELO PREDICTIVO ====================

def logistic(x):
    return 1 / (1 + np.exp(-x))


def modelo_predictivo(score):
    """
    Modelo predictivo basado en regresión logística
    Variable objetivo primaria: recidiva
    Variable objetivo secundaria: exitus
    """
    # Parámetros iniciales (placeholder clínico)
    recidiva_alpha = -3.0
    recidiva_beta = 0.35

    exitus_alpha = -4.0
    exitus_beta = 0.30

    prob_recidiva = logistic(recidiva_alpha + recidiva_beta * score)
    prob_exitus = logistic(exitus_alpha + exitus_beta * score)

    return prob_recidiva, prob_exitus


# Título principal
st.title("🔬 NEST - NSMP Endometrial Stratification Tool")
st.markdown("### Calculadora de Riesgo para Cáncer Endometrial NSMP")
st.markdown("---")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📊 Diagnóstico y Análisis", "🤖 Interpretación IA", "🔢 Cálculos Detallados"])

# ==================== PESTAÑA 1: DIAGNÓSTICO Y ANÁLISIS ====================
with tab1:
    st.header("📋 Evaluación Clínica")
    
    # Información del proyecto
    with st.expander("ℹ️ Información del Proyecto"):
        st.markdown("""
        **NEST** es una herramienta de estratificación pronóstica para pacientes con cáncer endometrial 
        de perfil molecular **NSMP (No Specific Molecular Profile)**, que representa aproximadamente el 50% 
        de los casos de cáncer endometrial.
        
        **Objetivo**: Mejorar la estratificación de riesgo integrando factores clinicopatológicos y moleculares 
        para decisiones terapéuticas personalizadas.
        
        *Desarrollado por: Grup de Recerca en Patologies Ginecològiques i de la Mama - Hospital Sant Pau*
        """)

    # Formulario de entrada de datos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Datos Demográficos")
        edad = st.number_input(
            "Edad (años)",
            min_value=18,
            max_value=100,
            value=65
        )
        imc = st.number_input(
            "IMC (kg/m²)",
            min_value=15.0,
            max_value=60.0,
            value=28.0
        )
        
        st.subheader("Características Histológicas")
        tipo_histologico = st.selectbox(
            "Tipo Histológico",
            ["Endometrioide", "Seroso", "Células claras", "Carcinosarcoma", "Mixto"]
        )
        
        grado = st.selectbox(
            "Grado Histológico",
            ["G1 (Bien diferenciado)", "G2 (Moderadamente diferenciado)", "G3 (Pobremente diferenciado)"]
        )
        
        tamano_tumoral = st.number_input(
            "Tamaño Tumoral (cm)",
            min_value=0.1,
            max_value=20.0,
            value=3.0
        )
    
    with col2:
        st.subheader("Factores Pronósticos")
        
        infiltracion_miometrial = st.selectbox(
            "Infiltración Miometrial",
            ["< 50%", "≥ 50%", "Sin infiltración"]
        )
        
        afectacion_linfovascular = st.selectbox(
            "Afectación Linfovascular",
            ["Negativa", "Positiva", "Indeterminada"]
        )
        
        receptores_estrogenos = st.selectbox(
            "Receptores de Estrógenos",
            ["Positivos", "Negativos", "No disponible"]
        )
        
        receptores_progesterona = st.selectbox(
            "Receptores de Progesterona",
            ["Positivos", "Negativos", "No disponible"]
        )
        
        estadio_figo = st.selectbox(
            "Estadio FIGO",
            ["IA", "IB", "II", "IIIA", "IIIB", "IIIC1", "IIIC2", "IVA", "IVB"]
        )
    
    st.markdown("---")
    
    # Guardar datos en session_state para usar en otras pestañas
    if 'calcular_presionado' not in st.session_state:
        st.session_state.calcular_presionado = False
    
    if st.button("🔍 Calcular Score de Riesgo", type="primary", width='stretch'):
        st.session_state.calcular_presionado = True
        
        # Sistema de puntuación
        score = 0
        factores_riesgo = []
        detalles_calculo = []
        
        # Edad
        if edad > 70:
            score += 2
            factores_riesgo.append("Edad > 70 años (+2)")
            detalles_calculo.append({"Factor": "Edad", "Valor": edad, "Puntos": 2, "Criterio": "> 70 años"})
        elif edad > 60:
            score += 1
            factores_riesgo.append("Edad > 60 años (+1)")
            detalles_calculo.append({"Factor": "Edad", "Valor": edad, "Puntos": 1, "Criterio": "> 60 años"})
        else:
            detalles_calculo.append({"Factor": "Edad", "Valor": edad, "Puntos": 0, "Criterio": "≤ 60 años"})
        
        # IMC
        if imc >= 35:
            score += 2
            factores_riesgo.append("IMC ≥ 35 (+2)")
            detalles_calculo.append({"Factor": "IMC", "Valor": imc, "Puntos": 2, "Criterio": "≥ 35"})
        elif imc >= 30:
            score += 1
            factores_riesgo.append("IMC ≥ 30 (+1)")
            detalles_calculo.append({"Factor": "IMC", "Valor": imc, "Puntos": 1, "Criterio": "≥ 30"})
        else:
            detalles_calculo.append({"Factor": "IMC", "Valor": imc, "Puntos": 0, "Criterio": "< 30"})
        
        # Grado histológico
        if "G3" in grado:
            score += 3
            factores_riesgo.append("Grado G3 (+3)")
            detalles_calculo.append({"Factor": "Grado Histológico", "Valor": "G3", "Puntos": 3, "Criterio": "Pobremente diferenciado"})
        elif "G2" in grado:
            score += 1
            factores_riesgo.append("Grado G2 (+1)")
            detalles_calculo.append({"Factor": "Grado Histológico", "Valor": "G2", "Puntos": 1, "Criterio": "Moderadamente diferenciado"})
        else:
            detalles_calculo.append({"Factor": "Grado Histológico", "Valor": "G1", "Puntos": 0, "Criterio": "Bien diferenciado"})
        
        # Tipo histológico
        if tipo_histologico in ["Seroso", "Células claras", "Carcinosarcoma"]:
            score += 3
            factores_riesgo.append(f"Histología de alto riesgo: {tipo_histologico} (+3)")
            detalles_calculo.append({"Factor": "Tipo Histológico", "Valor": tipo_histologico, "Puntos": 3, "Criterio": "Alto riesgo"})
        else:
            detalles_calculo.append({"Factor": "Tipo Histológico", "Valor": tipo_histologico, "Puntos": 0, "Criterio": "Riesgo estándar"})
        
        # Tamaño tumoral
        if tamano_tumoral > 5:
            score += 2
            factores_riesgo.append("Tamaño tumoral > 5 cm (+2)")
            detalles_calculo.append({"Factor": "Tamaño Tumoral", "Valor": f"{tamano_tumoral} cm", "Puntos": 2, "Criterio": "> 5 cm"})
        elif tamano_tumoral > 2:
            score += 1
            factores_riesgo.append("Tamaño tumoral > 2 cm (+1)")
            detalles_calculo.append({"Factor": "Tamaño Tumoral", "Valor": f"{tamano_tumoral} cm", "Puntos": 1, "Criterio": "> 2 cm"})
        else:
            detalles_calculo.append({"Factor": "Tamaño Tumoral", "Valor": f"{tamano_tumoral} cm", "Puntos": 0, "Criterio": "≤ 2 cm"})
        
        # Infiltración miometrial
        if "≥ 50%" in infiltracion_miometrial:
            score += 3
            factores_riesgo.append("Infiltración miometrial ≥ 50% (+3)")
            detalles_calculo.append({"Factor": "Infiltración Miometrial", "Valor": "≥ 50%", "Puntos": 3, "Criterio": "Profunda"})
        elif "< 50%" in infiltracion_miometrial:
            detalles_calculo.append({"Factor": "Infiltración Miometrial", "Valor": "< 50%", "Puntos": 0, "Criterio": "Superficial"})
        else:
            detalles_calculo.append({"Factor": "Infiltración Miometrial", "Valor": "Sin infiltración", "Puntos": 0, "Criterio": "Ausente"})
        
        # Afectación linfovascular
        if afectacion_linfovascular == "Positiva":
            score += 3
            factores_riesgo.append("Afectación linfovascular positiva (+3)")
            detalles_calculo.append({"Factor": "Afectación Linfovascular", "Valor": "Positiva", "Puntos": 3, "Criterio": "Presente"})
        else:
            detalles_calculo.append({"Factor": "Afectación Linfovascular", "Valor": afectacion_linfovascular, "Puntos": 0, "Criterio": "Ausente/Indeterminada"})
        
        # Receptores hormonales
        if receptores_estrogenos == "Negativos":
            score += 2
            factores_riesgo.append("Receptores de estrógenos negativos (+2)")
            detalles_calculo.append({"Factor": "Receptores Estrógenos", "Valor": "Negativos", "Puntos": 2, "Criterio": "Negativo"})
        else:
            detalles_calculo.append({"Factor": "Receptores Estrógenos", "Valor": receptores_estrogenos, "Puntos": 0, "Criterio": "Positivo/No disponible"})
        
        if receptores_progesterona == "Negativos":
            score += 2
            factores_riesgo.append("Receptores de progesterona negativos (+2)")
            detalles_calculo.append({"Factor": "Receptores Progesterona", "Valor": "Negativos", "Puntos": 2, "Criterio": "Negativo"})
        else:
            detalles_calculo.append({"Factor": "Receptores Progesterona", "Valor": receptores_progesterona, "Puntos": 0, "Criterio": "Positivo/No disponible"})
        
        # Estadio FIGO
        estadio_puntos = 0
        if estadio_figo in ["IVA", "IVB"]:
            estadio_puntos = 5
            factores_riesgo.append(f"Estadio FIGO {estadio_figo} (+5)")
        elif estadio_figo in ["IIIC1", "IIIC2"]:
            estadio_puntos = 4
            factores_riesgo.append(f"Estadio FIGO {estadio_figo} (+4)")
        elif estadio_figo in ["IIIA", "IIIB"]:
            estadio_puntos = 3
            factores_riesgo.append(f"Estadio FIGO {estadio_figo} (+3)")
        elif estadio_figo == "II":
            estadio_puntos = 2
            factores_riesgo.append(f"Estadio FIGO {estadio_figo} (+2)")
        elif estadio_figo == "IB":
            estadio_puntos = 1
            factores_riesgo.append(f"Estadio FIGO {estadio_figo} (+1)")
        
        score += estadio_puntos
        detalles_calculo.append({"Factor": "Estadio FIGO", "Valor": estadio_figo, "Puntos": estadio_puntos, "Criterio": f"Estadio {estadio_figo}"})
        
        # Clasificación de riesgo
        if score <= 5:
            categoria_riesgo = "BAJO"
            riesgo_recidiva = "< 15%"
            recomendacion = "Vigilancia activa. Considerar omitir terapia adyuvante."
        elif score <= 10:
            categoria_riesgo = "INTERMEDIO"
            riesgo_recidiva = "15-30%"
            recomendacion = "Considerar braquiterapia vaginal. Evaluación individualizada."
        else:
            categoria_riesgo = "ALTO"
            riesgo_recidiva = "> 30%"
            recomendacion = "Terapia adyuvante recomendada: radioterapia ± quimioterapia sistémica."
        
        # Guardar en session_state
        st.session_state.score = score
        st.session_state.categoria_riesgo = categoria_riesgo
        st.session_state.riesgo_recidiva = riesgo_recidiva
        st.session_state.factores_riesgo = factores_riesgo
        st.session_state.recomendacion = recomendacion
        st.session_state.detalles_calculo = detalles_calculo
        st.session_state.datos_clinicos = {
            "edad": edad,
            "imc": imc,
            "tipo_histologico": tipo_histologico,
            "grado": grado,
            "tamano_tumoral": tamano_tumoral,
            "infiltracion_miometrial": infiltracion_miometrial,
            "afectacion_linfovascular": afectacion_linfovascular,
            "receptores_estrogenos": receptores_estrogenos,
            "receptores_progesterona": receptores_progesterona,
            "estadio_figo": estadio_figo
        }

        prob_recidiva, prob_exitus = modelo_predictivo(score)

        st.session_state.predicciones = {
            "prob_recidiva": prob_recidiva,
            "prob_exitus": prob_exitus
        }
    
    # Mostrar resultados si se ha calculado
    if st.session_state.calcular_presionado:
        st.markdown("---")
        st.header("📊 Resultados del Análisis de Riesgo")

        # ==================== RESULTADOS DEL MODELO PREDICTIVO ====================
        st.subheader("🔮 Modelo Predictivo de Eventos")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Probabilidad de Recidiva",
                f"{st.session_state.predicciones['prob_recidiva']*100:.1f}%"
            )

        with col2:
            st.metric(
                "Probabilidad de Exitus",
                f"{st.session_state.predicciones['prob_exitus']*100:.1f}%"
            )

        if st.session_state.predicciones["prob_recidiva"] < 0.15:
            st.success("🟢 Bajo riesgo de recidiva según el modelo predictivo")
        elif st.session_state.predicciones["prob_recidiva"] < 0.30:
            st.warning("🟡 Riesgo intermedio de recidiva según el modelo predictivo")
        else:
            st.error("🔴 Alto riesgo de recidiva según el modelo predictivo")

        # Métricas principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score Total", st.session_state.score)
        
        with col2:
            st.metric("Categoría de Riesgo", st.session_state.categoria_riesgo)
        
        with col3:
            st.metric("Riesgo de Recidiva Estimado", st.session_state.riesgo_recidiva)
        
        # Visualización del nivel de riesgo
        if st.session_state.categoria_riesgo == "BAJO":
            st.success(f"### ✅ RIESGO {st.session_state.categoria_riesgo}")
        elif st.session_state.categoria_riesgo == "INTERMEDIO":
            st.warning(f"### ⚠️ RIESGO {st.session_state.categoria_riesgo}")
        else:
            st.error(f"### 🔴 RIESGO {st.session_state.categoria_riesgo}")
        
        # Gráficos
        st.subheader("📈 Visualización de Riesgo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart para el score
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = st.session_state.score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Score de Riesgo", 'font': {'size': 24}},
                delta = {'reference': 10},
                gauge = {
                    'axis': {'range': [None, 25], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 5], 'color': '#90EE90'},
                        {'range': [5, 10], 'color': '#FFD700'},
                        {'range': [10, 25], 'color': '#FF6B6B'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': st.session_state.score
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, width='stretch')
        
        with col2:
            # Gráfico de barras con factores de riesgo
            df_detalles = pd.DataFrame(st.session_state.detalles_calculo)
            df_factores = df_detalles[df_detalles['Puntos'] > 0]
            
            if len(df_factores) > 0:
                fig_bar = px.bar(
                    df_factores,
                    x='Puntos',
                    y='Factor',
                    orientation='h',
                    title='Contribución de Factores de Riesgo',
                    labels={'Puntos': 'Puntos de Riesgo', 'Factor': ''},
                    color='Puntos',
                    color_continuous_scale=['#90EE90', '#FFD700', '#FF6B6B']
                )
                fig_bar.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info("No hay factores de riesgo significativos identificados")
        
        # Distribución de puntos por categoría
        st.subheader("📊 Distribución de Puntos por Categoría")
        
        categorias = {
            "Demográficos": ["Edad", "IMC"],
            "Histopatológicos": ["Grado Histológico", "Tipo Histológico", "Tamaño Tumoral"],
            "Invasión Tumoral": ["Infiltración Miometrial", "Afectación Linfovascular"],
            "Moleculares": ["Receptores Estrógenos", "Receptores Progesterona"],
            "Estadificación": ["Estadio FIGO"]
        }
        
        puntos_categoria = {}
        for cat, factores in categorias.items():
            puntos = sum([d['Puntos'] for d in st.session_state.detalles_calculo if d['Factor'] in factores])
            puntos_categoria[cat] = puntos
        
        fig_pie = px.pie(
            values=list(puntos_categoria.values()),
            names=list(puntos_categoria.keys()),
            title='Distribución de Puntos de Riesgo por Categoría',
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, width='stretch')
        
        # Factores de riesgo identificados
        st.subheader("🔍 Factores de Riesgo Identificados")
        if st.session_state.factores_riesgo:
            for factor in st.session_state.factores_riesgo:
                st.markdown(f"- {factor}")
        else:
            st.info("No se identificaron factores de riesgo significativos.")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones Terapéuticas")
        st.info(st.session_state.recomendacion)
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Nota Importante**: Esta herramienta está en fase de desarrollo y validación. 
        Los resultados deben interpretarse en el contexto clínico completo y no sustituyen 
        el juicio clínico profesional.
        """)

# ==================== PESTAÑA 2: INTERPRETACIÓN IA ====================
with tab2:
    st.header("🤖 Interpretación Asistida por IA")
    
    if not st.session_state.calcular_presionado:
        st.info("👈 Por favor, calcula primero el score de riesgo en la pestaña 'Diagnóstico y Análisis'")
    else:
        st.markdown("""
        Esta sección proporciona una interpretación detallada de los resultados del análisis de riesgo.
        """)
        
        st.markdown("---")
        st.markdown(f"""
        ### 📋 Resumen del Perfil de Riesgo
        
        El caso presenta un perfil de riesgo **{st.session_state.categoria_riesgo}** con un score total de **{st.session_state.score} puntos**, 
        lo que se traduce en un riesgo estimado de recidiva de **{st.session_state.riesgo_recidiva}**.
        
        ### 📊 Explicación de los Gráficos
        
        **Gauge de Score de Riesgo**: Este medidor visual muestra la puntuación total en una escala de 0-25 puntos.
        Las zonas de color representan:
        - Verde (0-5): Riesgo bajo - pacientes con excelente pronóstico
        - Amarillo (5-10): Riesgo intermedio - requiere evaluación individualizada
        - Rojo (10-25): Riesgo alto - necesita tratamiento adyuvante agresivo
        
        **Gráfico de Barras Horizontal**: Muestra la contribución individual de cada factor de riesgo al score total.
        
        **Gráfico Circular (Donut)**: Distribuye los puntos de riesgo en cinco categorías principales.
        
        ### 🔬 Interpretación de Factores de Riesgo
        
        {chr(10).join(['**' + f.split('(')[0] + '**: Este factor ha sido identificado como significativo.' for f in st.session_state.factores_riesgo]) if st.session_state.factores_riesgo else 'No se identificaron factores de riesgo significativos.'}
        
        ### 🎯 Implicaciones Clínicas y Pronósticas
        
        Con un riesgo **{st.session_state.categoria_riesgo}**, el enfoque terapéutico recomendado es: {st.session_state.recomendacion}
        
        ### 📅 Consideraciones para el Seguimiento
        
        **Protocolo de vigilancia recomendado**:
        - Revisiones clínicas cada 3-6 meses durante los primeros 2 años
        - Exploración física ginecológica completa en cada visita
        - Citología vaginal según indicación clínica
        - Imagen (TAC/RMN) en caso de sospecha de recidiva
        """)

# ==================== PESTAÑA 3: CÁLCULOS DETALLADOS ====================
with tab3:
    st.header("🔢 Cálculos Matemáticos Detallados")
    
    if not st.session_state.calcular_presionado:
        st.info("👈 Por favor, calcula primero el score de riesgo en la pestaña 'Diagnóstico y Análisis'")
    else:
        st.markdown("""
        Esta sección muestra el desglose matemático completo del score de riesgo.
        """)
        
        st.markdown("---")
        
        # Tabla detallada de cálculos
        st.subheader("📊 Tabla de Variables y Puntuaciones")
        
        df_calculo = pd.DataFrame(st.session_state.detalles_calculo)
        # Convertir todos los valores a string para evitar problemas de tipo
        df_calculo['Valor'] = df_calculo['Valor'].astype(str)
        st.dataframe(df_calculo, width='stretch', height=500)
        
        # Fórmula matemática
        st.markdown("---")
        st.subheader("📐 Fórmula de Cálculo del Score")
        
        formula_partes = []
        for detalle in st.session_state.detalles_calculo:
            if detalle['Puntos'] > 0:
                formula_partes.append(f"{detalle['Puntos']}")
        
        if formula_partes:
            formula = " + ".join(formula_partes)
            st.markdown(f"### Score Total = {formula} = **{st.session_state.score} puntos**")
        else:
            st.markdown(f"### Score Total = **{st.session_state.score} puntos**")
        
        # Sistema de clasificación
        st.markdown("---")
        st.subheader("📏 Sistema de Clasificación de Riesgo")
        
        clasificacion_df = pd.DataFrame({
            "Categoría": ["BAJO", "INTERMEDIO", "ALTO"],
            "Rango de Score": ["0 - 5 puntos", "6 - 10 puntos", "≥ 11 puntos"],
            "Riesgo de Recidiva": ["< 15%", "15 - 30%", "> 30%"],
            "Recomendación": [
                "Vigilancia activa",
                "Braquiterapia vaginal (individualizar)",
                "Radioterapia ± Quimioterapia"
            ]
        })
        
        st.table(clasificacion_df)
        
        st.info(f"**Este caso se clasifica en la categoría de RIESGO {st.session_state.categoria_riesgo}** con {st.session_state.score} puntos.")
        
        # Opciones de exportación
        st.markdown("---")
        st.subheader("💾 Exportar Datos")
        
        csv_calculo = df_calculo.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Tabla de Cálculos (CSV)",
            data=csv_calculo,
            file_name=f"NEST_calculos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>NEST - NSMP Endometrial Stratification Tool</strong></p>
    <p>Grup de Recerca en Patologies Ginecològiques i de la Mama</p>
    <p>Hospital de la Santa Creu i Sant Pau - #BitsxlaMarató 2024</p>
</div>
""", unsafe_allow_html=True)