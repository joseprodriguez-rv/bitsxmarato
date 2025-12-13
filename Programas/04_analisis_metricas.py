import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_score, recall_score, f1_score)

# --- 1. CARGA Y PREPARACIÓN ---
print("🔄 Cargando datos...")
try:
    df = pd.read_csv('Dataset_NEST_Final_Reclassificat.csv')
except FileNotFoundError:
    print("❌ Error: No se encuentra 'Dataset_NEST_Final_Reclassificat.csv'")
    exit()

# Definir variables (las mismas que en tu App)
features = ['edad', 'imc', 'grado_histologi', 'infiltracion_mi', 'afectacion_linf', 'FIGO2023']
target = 'recidiva'

# Limpiar nulos
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Dividir en Train (80%) y Test (20%)
# random_state fijo para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ Datos cargados: {len(df)} pacientes.")
print(f"   - Entrenamiento: {len(X_train)}")
print(f"   - Prueba (Test): {len(X_test)}")

# --- 2. ENTRENAMIENTO DEL MODELO ---
# Usamos parámetros para controlar el overfitting (max_depth limitado)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]

# --- 3. DIVERSAS MÉTRICAS (TRAIN VS TEST) ---
print("\n" + "="*40)
print("📊 ANÁLISIS DE OVERFITTING / UNDERFITTING")
print("="*40)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"Accuracy en Entrenamiento: {acc_train:.2%}")
print(f"Accuracy en Test:          {acc_test:.2%}")

gap = acc_train - acc_test
print(f"Diferencia (Gap):          {gap:.2%}")

if gap > 0.15:
    print("⚠️ ALERTA: Posible OVERFITTING (El modelo memoriza el train pero falla en test).")
    print("   -> Solución: Reduce 'max_depth', aumenta datos o usa más regularización.")
elif acc_train < 0.60:
    print("⚠️ ALERTA: Posible UNDERFITTING (El modelo no aprende lo suficiente).")
    print("   -> Solución: Usa un modelo más complejo o añade más variables relevantes.")
else:
    print("✅ El modelo parece equilibrado.")

# --- 4. MÉTRICAS DETALLADAS ---
print("\n" + "="*40)
print("📈 MÉTRICAS DETALLADAS (En conjunto de Test)")
print("="*40)
print(classification_report(y_test, y_pred_test, target_names=['No Recidiva', 'Recidiva']))

# AUC-ROC
auc = roc_auc_score(y_test, y_prob_test)
print(f"⭐ AUC-ROC: {auc:.3f} (Ideal > 0.8)")

# --- 5. VISUALIZACIÓN ---
plt.figure(figsize=(15, 5))

# Gráfico 1: Matriz de Confusión
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Recidiva', 'Recidiva'], 
            yticklabels=['No Recidiva', 'Recidiva'])
plt.title('Matriz de Confusión (Test)')
plt.ylabel('Real')
plt.xlabel('Predicho')

# Gráfico 2: Curva ROC
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos (1 - Especificidad)')
plt.ylabel('Verdaderos Positivos (Sensibilidad)')
plt.title('Curva ROC')
plt.legend(loc="lower right")

# Gráfico 3: Curva de Aprendizaje (Visualizar Overfitting)
# Esto tarda un poco más, calcula cómo mejora el modelo al darle más datos
plt.subplot(1, 3, 3)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación Cruzada")
plt.title("Curva de Aprendizaje")
plt.xlabel("Tamaño del dataset")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()

plt.tight_layout()
plt.show()