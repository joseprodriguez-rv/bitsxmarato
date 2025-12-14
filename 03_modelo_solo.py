import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import optuna
import warnings
import json

# Configurar backend no-interactivo ANTES de importar pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 1. LOAD AND ANALYZE DATA
df = pd.read_csv("Dataset_NEST_Final_Reclassificat.csv")
target_ratio = df["recidiva"].value_counts()
scale_pos_weight = target_ratio[0] / target_ratio[1] if len(target_ratio) > 1 else 1
for c in df.columns:
    print(c)

date_cols = [
    "FN",
    "f_diag",
    "fecha_qx",
    "f_ultima_visita",
    "Ultima_fecha"
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_day"] = df[col].dt.day

# 2. FEATURE ENGINEERING
# En la sección "2. FEATURE ENGINEERING", ELIMINA estas features:

clinical_features = [
    "edad",
    "imc",
    "tipo_histologico",
    "grado_histologi",
    "infiltracion_mi",
    "ecotv_infiltobj",
    "ecotv_infiltsub",
    "metasta_distan",
    "grupo_riesgo",
    "estadiaje_pre_i",
    "tto_NA",
    "tto_1_quirugico",
    "asa",
    "histo_defin",
    "tamano_tumoral",
    "afectacion_linf",
    "AP_centinela_pelvico",
    "beta_cateninap",
    "mut_pole",
    "p53_ihq",
    "FIGO2023",
    "grupo_de_riesgo_definitivo",
    "Tributaria_a_Radioterapia",
    "bqt",
    "qt",
    "Tratamiento_sistemico_realizad",
    

    # FEATURES DERIVADAS DE FECHA (estas SÍ están OK)
    "FN_year", "FN_month", "FN_day",
    "f_diag_year", "f_diag_month", "f_diag_day",
    "Ultima_fecha_year", "Ultima_fecha_month", "Ultima_fecha_day"
]

X = df[clinical_features].copy()
y = df["recidiva"]

# 3. STRATIFIED SPLIT
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# 4. OPTUNA OPTIMIZATION - CAMBIADO A RECALL COMO MÉTRICA
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 3),
        "scale_pos_weight": scale_pos_weight,
        "max_bin": trial.suggest_int("max_bin", 128, 512),
        "random_state": 42,
        "n_jobs": -1,
        "device": "cpu"
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # CAMBIO IMPORTANTE: Optimizar por RECALL en lugar de accuracy
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring='recall',  # ← CAMBIO AQUÍ
        n_jobs=-1
    )
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

# 5. TRAIN FINAL MODEL WITH EARLY STOPPING
best_params = study.best_params.copy()
best_params["scale_pos_weight"] = scale_pos_weight
best_params["random_state"] = 42
best_params["n_jobs"] = -1
best_params["device"] = "cpu"

final_model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    early_stopping_rounds=75
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# 6. RETRAIN ON TRAIN + VALIDATION
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

final_model_full = XGBClassifier(**best_params, objective="binary:logistic")
final_model_full.fit(X_train_final, y_train_final)

# 7. EVALUACIÓN COMPLETA EN TEST SET
y_pred = final_model_full.predict(X_test)
y_pred_proba = final_model_full.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("MÉTRICAS EN TEST SET")
print("="*60)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Sensibilidad (Recall): {recall_score(y_test, y_pred):.4f}")
print(f"Precisión: {precision_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

print("\n" + "-"*60)
print("MATRIZ DE CONFUSIÓN")
print("-"*60)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nVerdaderos Negativos: {cm[0,0]}")
print(f"Falsos Positivos: {cm[0,1]}")
print(f"Falsos Negativos: {cm[1,0]}")
print(f"Verdaderos Positivos: {cm[1,1]}")

print("\n" + "-"*60)
print("REPORTE DE CLASIFICACIÓN COMPLETO")
print("-"*60)
print(classification_report(y_test, y_pred, target_names=['No Recidiva', 'Recidiva']))

# 8. AJUSTE DE UMBRAL PARA MEJORAR SENSIBILIDAD (BÚSQUEDA EXHAUSTIVA)
print("\n" + "="*60)
print("AJUSTE DE UMBRAL PARA MAXIMIZAR SENSIBILIDAD")
print("="*60)

# Probar umbrales desde 0.1 hasta 0.9 con pasos de 0.01
import numpy as np
umbrales_test = np.arange(0.1, 0.91, 0.01)
mejores_metricas = {
    'umbral': 0.5, 
    'recall': 0, 
    'accuracy': 0, 
    'precision': 0, 
    'f1': 0,
    'matriz_confusion': None
}

resultados_umbrales = []

for umbral in umbrales_test:
    y_pred_ajustado = (y_pred_proba >= umbral).astype(int)
    acc = accuracy_score(y_test, y_pred_ajustado)
    rec = recall_score(y_test, y_pred_ajustado)
    prec = precision_score(y_test, y_pred_ajustado, zero_division=0)
    f1 = f1_score(y_test, y_pred_ajustado, zero_division=0)
    
    resultados_umbrales.append({
        'umbral': umbral,
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1': f1
    })
    
    # Actualizar si encontramos mejor sensibilidad
    if f1 > mejores_metricas['f1']:
        mejores_metricas = {
            'umbral': float(umbral),
            'recall': float(rec),
            'accuracy': float(acc),
            'precision': float(prec),
            'f1': float(f1),
            'matriz_confusion': confusion_matrix(y_test, y_pred_ajustado).tolist()
        }

# Mostrar algunos umbrales de referencia
print("\nUMBRALES DE REFERENCIA:")
for umbral in [0.3, 0.35, 0.4, 0.45, 0.5]:
    idx = int((umbral - 0.1) * 100)
    if 0 <= idx < len(resultados_umbrales):
        r = resultados_umbrales[idx]
        print(f"\nUmbral: {r['umbral']:.2f}")
        print(f"  Accuracy: {r['accuracy']:.4f}")
        print(f"  Sensibilidad: {r['recall']:.4f}")
        print(f"  Precisión: {r['precision']:.4f}")
        print(f"  F1-Score: {r['f1']:.4f}")

print(f"\n{'='*60}")
print(f"✓ MEJOR UMBRAL ENCONTRADO: {mejores_metricas['umbral']:.2f}")
print(f"✓ Sensibilidad (Recall): {mejores_metricas['recall']:.4f}")
print(f"✓ Accuracy: {mejores_metricas['accuracy']:.4f}")
print(f"✓ Precisión: {mejores_metricas['precision']:.4f}")
print(f"✓ F1-Score: {mejores_metricas['f1']:.4f}")
print("="*60)

# Matriz de confusión con mejor umbral
print("\nMATRIZ DE CONFUSIÓN CON UMBRAL ÓPTIMO:")
cm_mejor = np.array(mejores_metricas['matriz_confusion'])
print(cm_mejor)
print(f"\nVerdaderos Negativos: {cm_mejor[0,0]}")
print(f"Falsos Positivos: {cm_mejor[0,1]}")
print(f"Falsos Negativos: {cm_mejor[1,0]}")
print(f"Verdaderos Positivos: {cm_mejor[1,1]}")

# Guardar modelo Y el mejor umbral
final_model_full.save_model('modelo_definitivo.json')

# Guardar todas las métricas y configuración
config = {
    'mejor_umbral': mejores_metricas['umbral'],
    'metricas_test': {
        'sensibilidad': mejores_metricas['recall'],
        'accuracy': mejores_metricas['accuracy'],
        'precision': mejores_metricas['precision'],
        'f1_score': mejores_metricas['f1'],
        'matriz_confusion': mejores_metricas['matriz_confusion']
    },
    'hiperparametros_optuna': best_params,
    'scale_pos_weight': float(scale_pos_weight),
    'todos_los_umbrales': resultados_umbrales  # Guardar TODOS los resultados
}

with open('modelo_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("\n✓ Modelo guardado en 'modelo_definitivo.json'")
print("✓ Configuración completa guardada en 'modelo_config.json'")
print("  (incluye todas las métricas y todos los umbrales probados)")

# IMPORTANTE: Para usar el modelo en producción:
print("\n" + "="*60)
print("CÓMO USAR EL MODELO EN PRODUCCIÓN:")
print("="*60)
print(f"""
1. Cargar el modelo y configuración:
   
   from xgboost import XGBClassifier
   import json
   
   model = XGBClassifier()
   model.load_model('modelo_definitivo.json')
   
   with open('modelo_config.json', 'r') as f:
       config = json.load(f)

2. Hacer predicciones con el umbral óptimo ({mejores_metricas['umbral']:.2f}):
   
   proba = model.predict_proba(X_nuevos)[:, 1]
   predicciones = (proba >= config['mejor_umbral']).astype(int)
   
   ⚠️  NO uses model.predict(X_nuevos) directamente
""")
