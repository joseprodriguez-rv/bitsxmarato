import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Herramienta necesaria para 3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. CARGA DE DATOS
# ---------------------------------------------------------
file_path = 'datos.csv' 
df = pd.read_csv(file_path)

# 2. SELECCIÓN DE VARIABLES
# ---------------------------------------------------------
features_numericas = ['edad', 'imc', 'valor_de_ca125']
features_categoricas = [
    'tipo_histologico', 'Grado', 'metasta_distan', 
    'grupo_riesgo', 'estadiaje_pre_i', 'ecotv_infiltsub'
]
target = 'recidiva' 

cols_to_use = features_numericas + features_categoricas + [target]
df_pca = df[cols_to_use].copy()

# 3. LIMPIEZA DE DATOS
# ---------------------------------------------------------
imputer_num = SimpleImputer(strategy='mean')
df_pca[features_numericas] = imputer_num.fit_transform(df_pca[features_numericas])

imputer_cat = SimpleImputer(strategy='most_frequent')
df_pca[features_categoricas] = imputer_cat.fit_transform(df_pca[features_categoricas])
df_pca[features_categoricas] = df_pca[features_categoricas].astype(str)

# 4. PREPROCESAMIENTO
# ---------------------------------------------------------
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features_numericas),
        ('cat', categorical_transformer, features_categoricas)
    ])

# 5. APLICAR PCA (¡CAMBIOS AQUÍ!)
# ---------------------------------------------------------
# Cambiamos a 3 componentes
pca = PCA(n_components=3)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('pca', pca)])

X_pca = clf.fit_transform(df_pca.drop(columns=[target]))

# IMPORTANTE: Ahora el DataFrame debe tener 3 nombres de columnas
df_result = pd.DataFrame(data = X_pca, columns = ['Componente 1', 'Componente 2', 'Componente 3'])

# Añadimos la columna de recidiva
df_result['Recidiva'] = df_pca[target].values

# 6. RESULTADOS Y VISUALIZACIÓN 3D
# ---------------------------------------------------------

# Calcular varianza
var = pca.explained_variance_ratio_
print(f"Varianza Comp 1: {var[0]:.2%}")
print(f"Varianza Comp 2: {var[1]:.2%}")
print(f"Varianza Comp 3: {var[2]:.2%}")
print(f"Total retenido: {sum(var):.2%}")

# Configurar gráfico 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Definir colores para Recidiva (0 y 1)
# Asumimos que 0 es No (Azul) y 1 es Sí (Rojo), ajusta según tus datos
targets = df_result['Recidiva'].unique()
colors = ['#1f77b4', '#d62728', '#2ca02c'] # Azul, Rojo, Verde (por si hay más clases)

for target_val, color in zip(targets, colors):
    indices = df_result['Recidiva'] == target_val
    ax.scatter(
        df_result.loc[indices, 'Componente 1'],
        df_result.loc[indices, 'Componente 2'],
        df_result.loc[indices, 'Componente 3'],
        c=color,
        label=f'Recidiva: {target_val}',
        s=60,            # Tamaño del punto
        alpha=0.7,       # Transparencia
        edgecolors='w'   # Borde blanco para resaltar
    )

# Etiquetas de los ejes con su varianza
ax.set_xlabel(f'Comp. 1 ({var[0]:.2%})')
ax.set_ylabel(f'Comp. 2 ({var[1]:.2%})')
ax.set_zlabel(f'Comp. 3 ({var[2]:.2%})')
ax.set_title('PCA 3D: Análisis de Pacientes', fontsize=16)
ax.legend()

plt.show()

# =========================================================
# 7. ANÁLISIS NUMÉRICO DE LOS RESULTADOS (LOADINGS & SCORES)
# =========================================================

print("\n" + "="*50)
print("ANÁLISIS DE FACTORES (LOADINGS)")
print("¿Qué variables originales pesan más en cada Componente?")
print("="*50)

# 1. RECUPERAR NOMBRES DE LAS VARIABLES
# Como usamos OneHotEncoder, las variables cambiaron de nombre (ej: 'Grado' -> 'Grado_1', 'Grado_2')
# Recuperamos los nombres generados por el preprocesador
feature_names = clf.named_steps['preprocessor'].get_feature_names_out()

# 2. EXTRAER LOS COEFICIENTES (LOADINGS) DEL PCA
# Esto nos dice cuánto contribuye cada variable a cada componente
pca_loadings = pd.DataFrame(
    clf.named_steps['pca'].components_.T, # Transponemos para que filas=variables
    index=feature_names,
    columns=['C1', 'C2', 'C3']
)

# 3. MOSTRAR LAS VARIABLES MÁS INFLUYENTES PARA EL COMPONENTE 1
# El C1 suele ser el más importante. Vamos a ver qué lo define.
print("\n--- Top 5 Variables que definen el COMPONENTE 1 ---")
# Ordenamos por valor absoluto para ver las que más pesan (sea positivo o negativo)
top_c1 = pca_loadings.iloc[abs(pca_loadings['C1']).argsort()[::-1]]
print(top_c1['C1'].head(5))

print("\n--- Top 5 Variables que definen el COMPONENTE 2 ---")
top_c2 = pca_loadings.iloc[abs(pca_loadings['C2']).argsort()[::-1]]
print(top_c2['C2'].head(5))


# 4. CALIDAD DE LA SEPARACIÓN (SILHOUETTE SCORE)
# =========================================================
from sklearn.metrics import silhouette_score

print("\n" + "="*50)
print("CALIDAD DE LA AGRUPACIÓN")
print("="*50)

# Calculamos el Silhouette Score usando las 3 dimensiones del PCA y la etiqueta Recidiva
# Valor cercano a 1: Separación perfecta
# Valor cercano a 0: Los grupos están solapados (mezclados)
# Valor negativo: Datos mal asignados
sil_score = silhouette_score(df_result[['Componente 1', 'Componente 2', 'Componente 3']], df_result['Recidiva'])

print(f"Silhouette Score (Recidiva vs No Recidiva): {sil_score:.4f}")

if sil_score > 0.5:
    print(">> CONCLUSIÓN: Existe una separación MUY CLARA entre grupos.")
elif sil_score > 0.2:
    print(">> CONCLUSIÓN: Hay cierta separación, pero con zonas de solapamiento.")
else:
    print(">> CONCLUSIÓN: Los grupos están muy mezclados. Es difícil distinguirlos solo con estas variables.")

# 5. COMPARACIÓN DE MEDIAS (CENTROIDES)
# =========================================================
print("\n" + "="*50)
print("PERFIL PROMEDIO EN PCA")
print("="*50)
# Agrupamos por Recidiva y vemos dónde cae el "centro" de cada grupo en el mapa 3D
centroides = df_result.groupby('Recidiva')[['Componente 1', 'Componente 2', 'Componente 3']].mean()
print(centroides)