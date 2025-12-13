import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier, plot_importance
from xgboost import XGBClassifier
import optuna
import warnings
import shap
from lime.lime_tabular import LimeTabularExplainer

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

        # Feature engineering de fechas
        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_day"] = df[col].dt.day



# 2. FEATURE ENGINEERING
#variables más relevantes segun ellos
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
    "estado",
    "est_pcte",
    "libre_enferm",
    "OS_MESES",
    "DFS_MESES",

    # FEATURES DERIVADAS DE FECHA
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


# 4. OPTUNA OPTIMIZATION
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
        "device": "cpu"  # Cambio de cuda a cpu
    }

    model = XGBClassifier(**params)
   
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring='accuracy',
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
best_params["device"] = "cpu"  # Cambio de cuda a cpu

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

print("Model trained on full dataset (train + validation)")