import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, root_mean_squared_error, r2_score

# Balanceo
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# SHAP para interpretabilidad
import shap

print("="*80)
print("🔬 ANÁLISIS COMPLETO CON NESTED CROSS-VALIDATION Y SHAP")
print("   Modelo: Random Forest Simple")
print("   Objetivos: recidiva | recidiva_exitus | diferencia_dias_reci_exit")
print("="*80)

# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================
print("\n📂 PASO 1: Cargando datos...")
file_path = 'Dataset_NEST_Final_Reclassificat.csv'
df = pd.read_csv(file_path)

# Corregir target binario
df['recidiva'] = df['recidiva'].apply(lambda x: 1 if x >= 1 else 0)

print(f"   ✓ Total de casos: {len(df)}")
print(f"   ✓ Recidiva: {df['recidiva'].value_counts().to_dict()}")
print(f"   ✓ Recidiva+Éxitus: {df['recidiva_exitus'].value_counts().to_dict()}")
print(f"   ✓ Días hasta recidiva: {df['diferencia_dias_reci_exit'].describe()[['mean', 'std', 'min', 'max']].to_dict()}")

# =============================================================================
# PASO 2: DEFINIR VARIABLES
# =============================================================================
print("\n🎯 PASO 2: Definiendo variables predictoras...")


features_numericas = [
    'edad',
    'imc',
    'valor_de_ca125'
]

features_categoricas = [
    'tipo_histologico',
    'grado_histologi', 
    'metasta_distan',
    'grupo_riesgo',
    'estadiaje_pre_i',
    'ecotv_infiltsub'
]


# Crear grupos de variables para interpretabilidad
grupos_variables = {
    'Demográficas': ['edad', 'imc'],
    'Clínicas': ['valor_de_ca125'],
    'Patológicas': ['tipo_histologico', 'grado_histologi'],
    'Estadificación': ['metasta_distan', 'grupo_riesgo', 'estadiaje_pre_i'],
    'Imagen': ['ecotv_infiltsub']
}


print(f"   ✓ Variables numéricas: {features_numericas}")
print(f"   ✓ Variables categóricas: {features_categoricas}")
print(f"   ✓ Grupos definidos: {list(grupos_variables.keys())}")

# =============================================================================
# PASO 3: PREPARAR PIPELINE
# =============================================================================
print("\n🔧 PASO 3: Construyendo pipeline de preprocesamiento...")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features_numericas),
        ('cat', categorical_transformer, features_categoricas)
    ])

print("   ✓ Imputación: media (numéricas) / moda (categóricas)")
print("   ✓ Escalado: StandardScaler")
print("   ✓ Encoding: OneHotEncoder con drop='first'")

# =============================================================================
# FUNCIÓN PARA NESTED CROSS-VALIDATION
# =============================================================================
def nested_cross_validation(X, y, pipeline, param_grid, task_type='classification', cv_outer=5, cv_inner=3):
    """
    NESTED CROSS-VALIDATION:
    
    ¿Qué es? Dos niveles de cross-validation anidados:
    - OUTER LOOP (externo): Divide datos en 5 partes para EVALUAR el modelo final
    - INNER LOOP (interno): Dentro de cada iteración externa, hace otro CV 
      para OPTIMIZAR hiperparámetros sin contaminar la evaluación
    
    ¿Por qué? 
    - Si solo haces un CV y optimizas hiperparámetros, tus resultados son 
      demasiado optimistas (has "espiado" los datos de validación)
    - Con nested CV, el outer loop nunca ve los datos del inner loop
    
    Retorna:
    - scores_outer: Resultados reales del modelo (sin sesgo de optimización)
    - best_params_list: Los mejores hiperparámetros encontrados en cada fold
    """
    
    outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42) if task_type == 'classification' else KFold(n_splits=cv_outer, shuffle=True, random_state=42)
    
    scores_outer = []
    best_params_list = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y if task_type == 'classification' else None)):
        print(f"      Fold {fold_idx + 1}/{cv_outer}...", end=" ")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # INNER LOOP: Buscar mejores hiperparámetros
        inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=42) if task_type == 'classification' else KFold(n_splits=cv_inner, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='roc_auc' if task_type == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_params_list.append(grid_search.best_params_)
        
        # OUTER LOOP: Evaluar con el mejor modelo encontrado
        y_pred = grid_search.predict(X_val)
        
        if task_type == 'classification':
            y_proba = grid_search.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            print(f"AUC = {score:.4f}")
        else:
            score = root_mean_squared_error(y_val, y_pred)  # RMSE
            print(f"RMSE = {score:.2f}")
        
        scores_outer.append(score)
    
    return scores_outer, best_params_list

# =============================================================================
# PASO 4: NESTED CV PARA CADA OBJETIVO
# =============================================================================
print("\n" + "="*80)
print("🔄 PASO 4: NESTED CROSS-VALIDATION")
print("   ¿Qué hace? Evalúa el modelo de forma honesta SIN sesgo de optimización")
print("="*80)

# Hiperparámetros a probar (Random Forest Simple)
param_grid_rf = {
    'classifier__n_estimators': [20, 30, 50],
    'classifier__max_depth': [2, 3],
    'classifier__min_samples_split': [10, 15],
    'classifier__min_samples_leaf': [5, 8]
}

resultados_objetivos = {}

# ---------------- OBJETIVO 1: RECIDIVA ----------------
print("\n📊 Objetivo 1: RECIDIVA (clasificación binaria)")

X = df[features_numericas + features_categoricas]
y_recidiva = df['recidiva']

# Split 70/30
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_recidiva, test_size=0.30, random_state=42, stratify=y_recidiva
)

pipeline_rf_class = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('classifier', RandomForestClassifier(
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    ))
])

print("\n   🔄 Ejecutando Nested CV (5 outer folds, 3 inner folds)...")
scores_recidiva, params_recidiva = nested_cross_validation(
    X_train_val, y_train_val, 
    pipeline_rf_class, 
    param_grid_rf,
    task_type='classification',
    cv_outer=5,
    cv_inner=3
)

print(f"\n   ✅ Resultados Nested CV:")
print(f"      AUC medio: {np.mean(scores_recidiva):.4f} (±{np.std(scores_recidiva):.4f})")
print(f"      AUCs por fold: {[f'{s:.4f}' for s in scores_recidiva]}")

resultados_objetivos['recidiva'] = {
    'scores': scores_recidiva,
    'params': params_recidiva,
    'X_train': X_train_val,
    'y_train': y_train_val,
    'X_test': X_test,
    'y_test': y_test,
    'pipeline': pipeline_rf_class,
    'task': 'classification'
}

# ---------------- OBJETIVO 2: RECIDIVA + ÉXITUS ----------------
print("\n📊 Objetivo 2: RECIDIVA + ÉXITUS (clasificación binaria)")

y_recidiva_exitus = df['recidiva_exitus']

X_train_val2, X_test2, y_train_val2, y_test2 = train_test_split(
    X, y_recidiva_exitus, test_size=0.30, random_state=42, stratify=y_recidiva_exitus
)

print("\n   🔄 Ejecutando Nested CV...")
scores_exitus, params_exitus = nested_cross_validation(
    X_train_val2, y_train_val2, 
    pipeline_rf_class, 
    param_grid_rf,
    task_type='classification',
    cv_outer=5,
    cv_inner=3
)

print(f"\n   ✅ Resultados Nested CV:")
print(f"      AUC medio: {np.mean(scores_exitus):.4f} (±{np.std(scores_exitus):.4f})")

resultados_objetivos['recidiva_exitus'] = {
    'scores': scores_exitus,
    'params': params_exitus,
    'X_train': X_train_val2,
    'y_train': y_train_val2,
    'X_test': X_test2,
    'y_test': y_test2,
    'pipeline': pipeline_rf_class,
    'task': 'classification'
}

# ---------------- OBJETIVO 3: DÍAS HASTA RECIDIVA ----------------
print("\n📊 Objetivo 3: DÍAS HASTA RECIDIVA (regresión)")

# Filtrar solo casos con recidiva
df_recidiva = df[df['recidiva'] == 1].copy()
X_reg = df_recidiva[features_numericas + features_categoricas]
y_dias = df_recidiva['diferencia_dias_reci_exit']

print(f"   ℹ️  Casos con recidiva: {len(df_recidiva)}")

if len(df_recidiva) > 20:  # Solo si hay suficientes datos
    from sklearn.model_selection import KFold
    
    X_train_val3, X_test3, y_train_val3, y_test3 = train_test_split(
        X_reg, y_dias, test_size=0.30, random_state=42
    )
    
    pipeline_rf_reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            max_features='sqrt',
            random_state=42
        ))
    ])
    
    param_grid_reg = {
        'regressor__n_estimators': [20, 30, 50],
        'regressor__max_depth': [2, 3, 4],
        'regressor__min_samples_split': [5, 10],
        'regressor__min_samples_leaf': [3, 5]
    }
    
    print("\n   🔄 Ejecutando Nested CV...")
    scores_dias, params_dias = nested_cross_validation(
        X_train_val3, y_train_val3, 
        pipeline_rf_reg, 
        param_grid_reg,
        task_type='regression',
        cv_outer=5,
        cv_inner=3
    )
    
    print(f"\n   ✅ Resultados Nested CV:")
    print(f"      RMSE medio: {np.mean(scores_dias):.2f} (±{np.std(scores_dias):.2f}) días")
    
    resultados_objetivos['diferencia_dias'] = {
        'scores': scores_dias,
        'params': params_dias,
        'X_train': X_train_val3,
        'y_train': y_train_val3,
        'X_test': X_test3,
        'y_test': y_test3,
        'pipeline': pipeline_rf_reg,
        'task': 'regression'
    }
else:
    print("   ⚠️  Muy pocos datos para regresión confiable")

# =============================================================================
# PASO 5: GRÁFICO DE OVERFITTING
# =============================================================================
print("\n" + "="*80)
print("📈 PASO 5: VISUALIZACIÓN DE OVERFITTING")
print("   ¿Qué muestra? Comparación entre rendimiento en train vs test")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (objetivo, datos) in enumerate(resultados_objetivos.items()):
    ax = axes[idx]
    
    # Entrenar modelo final con mejores parámetros promedio
    # (en práctica, usarías los parámetros más comunes de todos los folds)
    pipeline = datos['pipeline']
    pipeline.fit(datos['X_train'], datos['y_train'])
    
    if datos['task'] == 'classification':
        # Train score
        y_train_pred = pipeline.predict_proba(datos['X_train'])[:, 1]
        train_score = roc_auc_score(datos['y_train'], y_train_pred)
        
        # Test score
        y_test_pred = pipeline.predict_proba(datos['X_test'])[:, 1]
        test_score = roc_auc_score(datos['y_test'], y_test_pred)
        
        metric_name = 'AUC'
    else:
        # Train score
        y_train_pred = pipeline.predict(datos['X_train'])
        train_score = root_mean_squared_error(datos['y_train'], y_train_pred)
        
        # Test score
        y_test_pred = pipeline.predict(datos['X_test'])
        test_score = root_mean_squared_error(datos['y_test'], y_test_pred)
        
        metric_name = 'RMSE'
    
    # Gráfico
    categories = ['Train', 'Test']
    scores = [train_score, test_score]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Añadir valores en las barras
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Gap
    gap = abs(train_score - test_score)
    ax.axhline(y=train_score, color='gray', linestyle='--', alpha=0.3)
    ax.text(0.5, (train_score + test_score)/2, 
            f'Gap: {gap:.3f}', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_title(f'{objetivo.replace("_", " ").title()}\n{metric_name}', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric_name)
    ax.grid(axis='y', alpha=0.3)
    
    # Interpretación
    if datos['task'] == 'classification':
        if gap < 0.05:
            status = '✅ Sin overfitting'
        elif gap < 0.10:
            status = '⚠️ Overfitting leve'
        else:
            status = '❌ Overfitting alto'
    else:
        if gap < 50:
            status = '✅ Buen ajuste'
        elif gap < 100:
            status = '⚠️ Ajuste moderado'
        else:
            status = '❌ Mal ajuste'
    
    ax.text(0.5, ax.get_ylim()[1] * 0.95, status, 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n   ✓ Gráfico guardado: overfitting_analysis.png")

# =============================================================================
# PASO 6: ANÁLISIS SHAP (INTERPRETABILIDAD)
# =============================================================================
print("\n" + "="*80)
print("🔍 PASO 6: ANÁLISIS SHAP - INTERPRETABILIDAD DEL MODELO")
print("   ¿Qué es SHAP? Muestra cuánto contribuye cada variable a la predicción")
print("="*80)

# Analizar solo RECIDIVA (el más importante clínicamente)
print("\n   Analizando RECIDIVA...")

datos_recidiva = resultados_objetivos['recidiva']
pipeline_final = datos_recidiva['pipeline']
pipeline_final.fit(datos_recidiva['X_train'], datos_recidiva['y_train'])

# Preprocesar datos para SHAP
X_train_preprocessed = pipeline_final.named_steps['preprocessor'].transform(datos_recidiva['X_train'])
X_train_smote, _ = pipeline_final.named_steps['smote'].fit_resample(X_train_preprocessed, datos_recidiva['y_train'])

# Tomar muestra para SHAP (si hay muchos datos)
# Convertir a array denso si es sparse
if hasattr(X_train_smote, 'toarray'):
    X_train_smote = X_train_smote.toarray()

# Tomar muestra para SHAP (si hay muchos datos)
sample_size = min(100, X_train_smote.shape[0])
X_sample = X_train_smote[np.random.choice(len(X_train_smote), sample_size, replace=False)]

# Crear explainer
modelo_rf = pipeline_final.named_steps['classifier']
explainer = shap.TreeExplainer(modelo_rf)
shap_values = explainer.shap_values(X_sample)

# Si es clasificación binaria, tomar clase positiva
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Nombres de features
feature_names = pipeline_final.named_steps['preprocessor'].get_feature_names_out()
feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

# ----- GRÁFICO 1: IMPORTANCIA GLOBAL (Summary Plot) -----
print("\n   📊 Generando gráfico SHAP - Importancia global...")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
plt.title('SHAP: Impacto de cada variable en la predicción de RECIDIVA', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("      ✓ Interpretación:")
print("         - Eje X: Cuánto cambia la predicción")
print("         - Color ROJO: Valor alto de la variable")
print("         - Color AZUL: Valor bajo de la variable")
print("         - Ejemplo: Si 'edad' tiene puntos rojos a la derecha →")
print("                    mayor edad aumenta el riesgo de recidiva")

# ----- GRÁFICO 2: IMPORTANCIA POR GRUPOS -----
print("\n   📊 Generando gráfico - Importancia por GRUPOS de variables...")

# Calcular importancia absoluta por feature
importancia_features = np.abs(shap_values).mean(axis=0)
importancia_df = pd.DataFrame({
    'feature': feature_names,
    'importancia': importancia_features
})

# Asignar cada feature a su grupo
def asignar_grupo(feature_name):
    for grupo, variables in grupos_variables.items():
        for var in variables:
            if var in feature_name.lower():
                return grupo
    return 'Otros'

importancia_df['grupo'] = importancia_df['feature'].apply(asignar_grupo)

# Agrupar por categoría
importancia_grupos = importancia_df.groupby('grupo')['importancia'].sum().sort_values(ascending=False)

# Gráfico de barras
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Importancia por grupo
colors_grupos = sns.color_palette("husl", len(importancia_grupos))
importancia_grupos.plot(kind='barh', ax=ax1, color=colors_grupos, edgecolor='black')
ax1.set_title('Importancia SHAP por GRUPO de Variables', fontsize=14, fontweight='bold')
ax1.set_xlabel('Importancia acumulada (SHAP)')
ax1.set_ylabel('Grupo de variables')
ax1.grid(axis='x', alpha=0.3)

# Subplot 2: Top 10 variables individuales
top_10_features = importancia_df.nlargest(10, 'importancia')
ax2.barh(range(len(top_10_features)), top_10_features['importancia'], 
         color=[colors_grupos[list(importancia_grupos.index).index(g)] for g in top_10_features['grupo']],
         edgecolor='black')
ax2.set_yticks(range(len(top_10_features)))
ax2.set_yticklabels(top_10_features['feature'])
ax2.set_title('Top 10 Variables Individuales', fontsize=14, fontweight='bold')
ax2.set_xlabel('Importancia SHAP')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('shap_grupos.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n   ✓ Gráficos guardados: shap_summary.png, shap_grupos.png")

# ----- TABLA RESUMEN -----
print("\n   📋 TABLA RESUMEN - Importancia por grupos:")
print("   " + "-"*50)
for grupo, valor in importancia_grupos.items():
    porcentaje = (valor / importancia_grupos.sum()) * 100
    print(f"   {grupo:<20} {valor:>8.4f}  ({porcentaje:>5.1f}%)")
print("   " + "-"*50)

# =============================================================================
# PASO 7: RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("📝 RESUMEN FINAL")
print("="*80)

print("\n🎯 RESULTADOS NESTED CROSS-VALIDATION:")
for objetivo, datos in resultados_objetivos.items():
    scores = datos['scores']
    if datos['task'] == 'classification':
        print(f"\n   {objetivo.upper()}:")
        print(f"      AUC medio: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        print(f"      Rango: {min(scores):.4f} - {max(scores):.4f}")
    else:
        print(f"\n   {objetivo.upper()}:")
        print(f"      RMSE medio: {np.mean(scores):.2f} (±{np.std(scores):.2f}) días")

print("\n🔍 VARIABLES MÁS IMPORTANTES (según SHAP):")
print("   Top 5:")
for i, (idx, row) in enumerate(top_10_features.head(5).iterrows(), 1):
    print(f"   {i}. {row['feature']} (Grupo: {row['grupo']})")

print("\n✅ ARCHIVOS GENERADOS:")
print("   • overfitting_analysis.png → Comparación Train vs Test")
print("   • shap_summary.png → Impacto individual de cada variable")
print("   • shap_grupos.png → Importancia por grupos de variables")

print("\n💡 INTERPRETACIÓN:")
print("   ✓ Nested CV asegura resultados honestos (sin sesgo de optimización)")
print("   ✓ SHAP muestra QUÉ variables son importantes y CÓMO afectan")
print("   ✓ Los gráficos de overfitting muestran si el modelo generaliza bien")

print("\n" + "="*80)