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



# 7. FEATURE IMPORTANCE AND SHAP PREPARATION
feature_names = X_train_final.columns.tolist()

importances = final_model_full.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

top_feats = feat_imp_df['feature'].iloc[:4].tolist()
print("\nTop 4 features:", top_feats)

print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(final_model_full)

# Reducir muestra para evitar problemas de memoria
X_shap = X_train_final.sample(n=min(500, len(X_train_final)), random_state=42)
shap_values = explainer.shap_values(X_shap)

print(f"SHAP values computed: {shap_values.shape}")


# 8. SHAP SUMMARY PLOT
try:
    print("\nGenerating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="dot", show=False)
    plt.title("SHAP Summary Plot - Global Feature Importance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ SHAP Summary guardado como 'shap_summary.png'")
except Exception as e:
    print(f"⚠ SHAP Summary falló: {str(e)}")
    plt.close()


# 9. SHAP DEPENDENCE PLOTS
try:
    print("\nGenerating individual Dependence Plots...")
    for i, feat in enumerate(top_feats[:4], 1):
        try:
            print(f"  {i}. {feat}...")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            shap.dependence_plot(
                ind=feat,
                shap_values=shap_values,
                features=X_shap,
                interaction_index="auto",
                ax=ax,
                show=False, 
                alpha=0.5,
                dot_size=20
            )
            
            ax.set_xlabel(f"{feat}", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"SHAP Value Impact", fontsize=12, fontweight='bold')
            ax.set_title(f"SHAP Dependence Plot: {feat}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.savefig(f'shap_dependence_{i}_{feat}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Guardado como 'shap_dependence_{i}_{feat}.png'")
        except Exception as e:
            print(f"  ✗ Error con {feat}: {str(e)}")
            plt.close()
except Exception as e:
    print(f"⚠ Dependence plots fallaron: {str(e)}")


# 10. PARTIAL DEPENDENCE PLOTS
try:
    print("\nGenerating Partial Dependence Plots...")
    pdp_feats = top_feats[:3]
    
    # Filtrar features válidas para PDP
    valid_pdp_feats = []
    for feat in pdp_feats:
        unique_vals = X_train_final[feat].nunique()
        val_range = X_train_final[feat].max() - X_train_final[feat].min()
        
        if unique_vals > 10 and val_range > 0:
            valid_pdp_feats.append(feat)
            print(f"  ✓ {feat}: {unique_vals} unique values, range={val_range:.2f}")
        else:
            print(f"  ✗ Skipping {feat}: {unique_vals} unique values, range={val_range}")
    
    if valid_pdp_feats:
        # Usar muestra para acelerar
        X_pdp = X_train_final.sample(n=min(1000, len(X_train_final)), random_state=42)
        
        fig, ax = plt.subplots(len(valid_pdp_feats), 1, figsize=(8, 4*len(valid_pdp_feats)))
        if len(valid_pdp_feats) == 1:
            ax = [ax]
            
        for i, feat in enumerate(valid_pdp_feats):
            try:
                PartialDependenceDisplay.from_estimator(
                    final_model_full,
                    X_pdp,
                    features=[feat],
                    kind='average',
                    ax=ax[i],
                    grid_resolution=50,
                    n_jobs=1
                )
                ax[i].set_title(f"PDP: {feat}", fontsize=12, fontweight='bold')
                print(f"  ✓ PDP generado para {feat}")
            except ValueError as e:
                print(f"  ✗ Error en {feat}: {str(e)}")
                ax[i].text(0.5, 0.5, f'Error generando PDP para {feat}', 
                          ha='center', va='center', fontsize=10)
                ax[i].set_xlim(0, 1)
                ax[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('partial_dependence_plots.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("✓ PDP guardado como 'partial_dependence_plots.png'")
    else:
        print("⚠ No hay features válidas para generar PDP")
except Exception as e:
    print(f"⚠ PDP falló: {str(e)}")
    plt.close()


# 12. SHAP WATERFALL PLOT
try:
    print("\nGenerating SHAP Waterfall Plot...")
    if len(X_test) > 0:
        idx = X_test.index[0]
        x_instance = X_test.loc[[idx]]
        shap_vals_instance = explainer.shap_values(x_instance)
        
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_vals_instance[0], 
            feature_names=x_instance.columns, 
            max_display=15
        )
        plt.tight_layout()
        plt.savefig('shap_waterfall.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("✓ Waterfall guardado como 'shap_waterfall.png'")
    else:
        print("⚠ X_test vacío, saltando waterfall")
except Exception as e:
    print(f"⚠ Waterfall falló: {str(e)}")
    plt.close()


# 13. SHAP FORCE PLOT
try:
    print("\nGenerating SHAP Force Plot...")
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_vals_instance[0], x_instance)
    print("✓ Force plot generado (revisar en notebook/HTML)")
except Exception as e:
    print(f"⚠ Force plot falló (normal en scripts sin Jupyter): {str(e)}")


# 14. LIME EXPLAINER
try:
    print("\nGenerating LIME Explanation...")
    if len(X_test) > 0:
        X_train_np = X_train_final.values
        feature_names = X_train_final.columns.tolist()
        class_names = ['no', 'yes']

        explainer_lime = LimeTabularExplainer(
            X_train_np,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
            random_state=42
        )

        i = X_test.index[0]

        exp = explainer_lime.explain_instance(
            X_test.loc[i].values,
            final_model_full.predict_proba,
            num_features=8,
            top_labels=2
        )

        print("Available labels:", exp.available_labels())
        label = exp.available_labels()[0]

        print("\nLIME Explanation:")
        print(exp.as_list(label=label))

        fig = exp.as_pyplot_figure(label=label)
        fig.tight_layout()
        plt.savefig('lime_explanation.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("✓ LIME guardado como 'lime_explanation.png'")
    else:
        print("⚠ X_test vacío, saltando LIME")
except Exception as e:
    print(f"⚠ LIME falló: {str(e)}")
    plt.close()


print("\n" + "="*60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("="*60)
print("\nArchivos generados:")
print("  - shap_summary.png")
print("  - shap_dependence_*.png")
print("  - partial_dependence_plots.png")
print("  - shap_waterfall.png")
print("  - lime_explanation.png")



# 8. EVALUACIÓN FINAL EN TEST
print("\n" + "=" * 60)
print("EVALUACIÓN EN TEST SET")
print("=" * 60)

# Predicciones
test_preds = final_model_full.predict(X_test)
test_proba = final_model_full.predict_proba(X_test)[:, 1]

# Métricas
acc = accuracy_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_proba)

print(f"\nAccuracy en TEST: {acc:.4f}")
print(f"ROC-AUC en TEST: {auc:.4f}")
print(f"\nMEJORA vs baseline: {(acc - 0.83) / 0.83 * 100:+.2f}%")
print("\nInforme de clasificación:")
print(classification_report(y_test, test_preds))


# 9. LEARNING CURVE - LOSS/TAMAÑO DEL DATASET
from sklearn.model_selection import learning_curve

print("\n" + "=" * 60)
print("GENERANDO LEARNING CURVES")
print("=" * 60)

train_sizes, train_scores, val_scores = learning_curve(
    final_model_full,
    X_train_final,
    y_train_final,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(12, 5))


train_sizes_loss, train_scores_loss, val_scores_loss = learning_curve(
    final_model_full,
    X_train_final,
    y_train_final,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_log_loss',
    random_state=42
)

# Convertir a positivo (neg_log_loss es negativo)
train_mean_loss = -np.mean(train_scores_loss, axis=1)
train_std_loss = np.std(train_scores_loss, axis=1)
val_mean_loss = -np.mean(val_scores_loss, axis=1)
val_std_loss = np.std(val_scores_loss, axis=1)

plt.plot(train_sizes_loss, train_mean_loss, 'o-', color='r', label='Training Loss')
plt.plot(train_sizes_loss, val_mean_loss, 'o-', color='g', label='Validation Loss')
plt.fill_between(train_sizes_loss, train_mean_loss - train_std_loss, train_mean_loss + train_std_loss, alpha=0.15, color='r')
plt.fill_between(train_sizes_loss, val_mean_loss - val_std_loss, val_mean_loss + val_std_loss, alpha=0.15, color='g')
plt.xlabel('Tamaño del Training Set')
plt.ylabel('Log Loss')
plt.title('Learning Curve - Loss')
plt.legend(loc='best')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("learning_curve_loss.png", dpi=100)
plt.close()

print("Learning Curves generadas exitosamente")

# Feature importance del XGBoost base
plt.figure(figsize=(10, 6))
plot_importance(final_model_full, max_num_features=20, importance_type='weight')
plt.title("Orden de importancia de las features")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=100)
plt.close()


# Comparación de modelos
print("\n" + "=" * 60)
print("EVALUACIÓN FINAL")
print("=" * 60)

print(f"XGBoost : {accuracy_score(y_test, final_model_full.predict(X_test)):.4f}")

print("\n" + "=" * 60)
print("¡PIPELINE COMPLETADO!")
print("=" * 60)
