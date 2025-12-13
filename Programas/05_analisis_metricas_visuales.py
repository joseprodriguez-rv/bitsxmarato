import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)

# --- CONFIGURACIÓN DE ESTILO ---
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# --- 1. CARGA Y PREPARACIÓN ---
print("🔄 Cargando datos...")
try:
    df = pd.read_csv('Dataset_NEST_Final_Reclassificat.csv')
except FileNotFoundError:
    print("❌ Error: No se encuentra 'Dataset_NEST_Final_Reclassificat.csv'")
    exit()

# Variables definidas por el usuario
features = ['edad', 'imc', 'grado_histologi', 'infiltracion_mi', 'afectacion_linf', 'FIGO2023']
target = 'recidiva'

# Limpieza básica
df = df[features + [target]].dropna()
X = df[features]
y = df[target]

# División Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. ENTRENAMIENTO ---
# Modelo ajustado para evitar overfitting excesivo (max_depth=5)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1] # Probabilidad de clase 1 (Recidiva)

# --- 3. CÁLCULO DE MÉTRICAS ---
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
auc = roc_auc_score(y_test, y_prob_test)

print(f"\n📊 RESULTADOS RÁPIDOS:")
print(f"   - Accuracy Test: {acc_test:.1%}")
print(f"   - AUC-ROC:       {auc:.3f}")
print(f"   - Gap (Overfit): {acc_train - acc_test:.1%} (Ideal < 10-15%)")

# --- 4. GENERACIÓN DE GRÁFICAS CLARAS ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Evaluación del Modelo NEST (Random Forest)\nAccuracy Test: {acc_test:.1%} | AUC: {auc:.3f}', fontsize=16, fontweight='bold')

# GRÁFICA 1: Matriz de Confusión con Porcentajes
cm = confusion_matrix(y_test, y_pred_test)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalizar por filas
labels = [f"{v1}\n({v2:.1%})" for v1, v2 in zip(cm.flatten(), cm_percent.flatten())]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=axes[0,0], cbar=False, annot_kws={"size": 14})
axes[0,0].set_title('1. Matriz de Confusión (Predicciones)', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Predicho por el Modelo')
axes[0,0].set_ylabel('Realidad Clínica')
axes[0,0].set_xticklabels(['Sano (No Recidiva)', 'Recidiva'])
axes[0,0].set_yticklabels(['Sano', 'Recidiva'])

# GRÁFICA 2: Curva ROC (Capacidad de Discriminación)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
axes[0,1].plot(fpr, tpr, color='#e74c3c', lw=3, label=f'AUC = {auc:.2f}')
axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Azar (0.5)')
axes[0,1].fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
axes[0,1].set_xlim([0.0, 1.0])
axes[0,1].set_ylim([0.0, 1.05])
axes[0,1].set_xlabel('Tasa de Falsos Positivos')
axes[0,1].set_ylabel('Tasa de Verdaderos Positivos')
axes[0,1].set_title('2. Curva ROC (Sensibilidad vs Falsos)', fontsize=14, fontweight='bold')
axes[0,1].legend(loc="lower right")
axes[0,1].grid(True, alpha=0.3)

# GRÁFICA 3: Histograma de Probabilidades (Seguridad del Modelo)
sns.histplot(x=y_prob_test, hue=y_test, element="step", stat="density", common_norm=False, 
             palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[1,0], bins=15, alpha=0.6)
axes[1,0].set_title('3. Densidad de Probabilidad (Separación de Clases)', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Probabilidad asignada de Recidiva (0 a 1)')
axes[1,0].set_ylabel('Densidad de Pacientes')
axes[1,0].legend(title='Realidad', labels=['Recidiva', 'Sano'])
axes[1,0].axvline(0.5, color='gray', linestyle='--', linewidth=1)
axes[1,0].text(0.05, axes[1,0].get_ylim()[1]*0.9, "Zona Segura", color="#2ecc71", fontweight='bold')
axes[1,0].text(0.80, axes[1,0].get_ylim()[1]*0.9, "Zona Peligro", color="#e74c3c", fontweight='bold')

# GRÁFICA 4: Curva de Aprendizaje (Overfitting/Underfitting)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

axes[1,1].plot(train_sizes, train_mean, 'o-', color="#3498db", label="Entrenamiento", lw=2)
axes[1,1].plot(train_sizes, test_mean, 'o-', color="#2ecc71", label="Validación (Test)", lw=2)

# Sombras de desviación estándar
axes[1,1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#3498db")
axes[1,1].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="#2ecc71")

axes[1,1].set_title('4. Curva de Aprendizaje (Estabilidad)', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Número de muestras de entrenamiento')
axes[1,1].set_ylabel('Accuracy')
axes[1,1].legend(loc="best")
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Gráficas generadas. Úsalas en tu presentación para demostrar la robustez del modelo.")