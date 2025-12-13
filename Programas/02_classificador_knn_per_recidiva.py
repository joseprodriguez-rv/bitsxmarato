features = [
    'edad', 'imc', 'grado_histologi', 'valor_de_ca125',
    'infiltracion_mi', 'estadiaje_pre_i', 'tamano_tumoral',
    'OS_MESES', 'DFS_MESES', 'diferencia_dias_reci_exit'
]
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Carregar les dades
print("Carregant dades...")
df = pd.read_csv('Dataset_NEST_Final.csv')
print(f"✓ Dades carregades: {len(df)} registres\n")

print("="*80)
print("CLASSIFICADOR KNN PER RECIDIVA I RECIDIVA_EXITUS")
print("="*80)

# 1. CREAR/VERIFICAR DIFERENCIA_DIAS_RECI_EXIT
print("\n1. Verificant i recalculant diferencia_dias_reci_exit...")

# Convertir dates si no ho estan
date_cols = ['f_diag', 'fecha_de_recidi', 'f_muerte', 'Ultima_fecha']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Recalcular diferencia_dias_reci_exit segons la definició correcta
if 'f_diag' in df.columns:
    def calcular_diferencia_correcta(row):
        """
        Dies des del diagnòstic fins a recidiva/mort/últim seguiment
        (el que passi primer)
        """
        try:
            if pd.isna(row['f_diag']):
                return np.nan
            
            # Buscar la data final (el que passi primer)
            dates_finals = []
            if pd.notna(row.get('fecha_de_recidi')):
                dates_finals.append(row['fecha_de_recidi'])
            if pd.notna(row.get('f_muerte')):
                dates_finals.append(row['f_muerte'])
            if pd.notna(row.get('Ultima_fecha')):
                dates_finals.append(row['Ultima_fecha'])
            
            if dates_finals:
                # Agafar la primera data que passi
                data_final = min(dates_finals)
                diferencia = (data_final - row['f_diag']).days
                return diferencia
        except:
            pass
        return np.nan
    
    df['diferencia_dias_reci_exit'] = df.apply(calcular_diferencia_correcta, axis=1)
    casos_amb_dif = df['diferencia_dias_reci_exit'].notna().sum()
    pct = (casos_amb_dif / len(df)) * 100
    print(f"   ✓ Variable recalculada correctament")
    print(f"   - Disponibilitat: {casos_amb_dif}/{len(df)} ({pct:.1f}%)")
else:
    print("   ✗ No es pot calcular (falta 'f_diag')")
    casos_amb_dif = 0
    pct = 0

# 2. PREPARAR FEATURES PER KNN
print("\n2. Preparant features per KNN...")

features = [
    'edad', 'imc', 'grado_histologi', 'valor_de_ca125',
    'infiltracion_mi', 'estadiaje_pre_i', 'tamano_tumoral',
    'OS_MESES', 'DFS_MESES', 'diferencia_dias_reci_exit'
]

print(f"   - Features seleccionades: {len(features)}")

# Verificar disponibilitat de cada feature
print("\n   Disponibilitat de features:")
for f in features:
    if f in df.columns:
        disponibilitat = df[f].notna().sum()
        percentatge = (disponibilitat / len(df)) * 100
        status = "✓" if percentatge >= 50 else "⚠️"
        print(f"     {status} {f:30} → {disponibilitat:3}/{len(df)} ({percentatge:5.1f}%)")
    else:
        print(f"     ✗ {f:30} → NO DISPONIBLE")

# 3. RECLASSIFICAR RECIDIVA
print("\n" + "="*80)
print("RECLASSIFICACIÓ DE RECIDIVA")
print("="*80)

# Guardar valors originals
df['recidiva_original'] = df['recidiva']

# Separar training (0, 1) i test (2 - desconeguts)
train_recidiva = df[df['recidiva'].isin([0, 1])].copy()
test_recidiva = df[df['recidiva'] == 2].copy()

print(f"\nCasos training (recidiva 0 o 1): {len(train_recidiva)}")
print(f"Casos test (recidiva = 2): {len(test_recidiva)}")

if len(test_recidiva) > 0 and len(train_recidiva) > 0:
    # Preparar dades de training
    X_train = train_recidiva[features].copy()
    y_train = train_recidiva['recidiva'].copy()
    
    # Preparar dades de test
    X_test = test_recidiva[features].copy()
    
    # Imputar valors perduts amb la mediana
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Normalitzar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Entrenar KNN
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Prediccions
    predictions = knn.predict(X_test_scaled)
    probabilities = knn.predict_proba(X_test_scaled)
    
    # Assignar prediccions
    test_indices = test_recidiva.index
    for i, idx in enumerate(test_indices):
        df.loc[idx, 'recidiva'] = predictions[i]
        df.loc[idx, 'recidiva_confidence'] = probabilities[i].max()
    
    print(f"\n✓ Reclassificació completada!")
    print(f"  - Casos reclassificats a 0: {(predictions == 0).sum()}")
    print(f"  - Casos reclassificats a 1: {(predictions == 1).sum()}")
    print(f"  - Confiança mitjana: {probabilities.max(axis=1).mean():.2%}")
else:
    print("\n✗ No hi ha casos a reclassificar per recidiva")

# 4. RECLASSIFICAR RECIDIVA_EXITUS
print("\n" + "="*80)
print("RECLASSIFICACIÓ DE RECIDIVA_EXITUS")
print("="*80)

# Comprovar si la columna existeix
if 'recidiva_exitus' not in df.columns:
    print("\n⚠️  La columna 'recidiva_exitus' no existeix al dataset")
    print("   Columnes disponibles relacionades amb recidiva:")
    recidiva_cols = [col for col in df.columns if 'recid' in col.lower()]
    for col in recidiva_cols:
        print(f"     · {col}")
    print("\n   Saltant la reclassificació de recidiva_exitus...")
    train_exitus = pd.DataFrame()
    test_exitus = pd.DataFrame()
else:
    # Guardar valors originals
    df['recidiva_exitus_original'] = df['recidiva_exitus']
    
    # Separar training (0, 1) i test (2 - desconeguts)
    train_exitus = df[df['recidiva_exitus'].isin([0, 1])].copy()
    test_exitus = df[df['recidiva_exitus'] == 2].copy()

if 'recidiva_exitus' in df.columns:
    print(f"\nCasos training (recidiva_exitus 0 o 1): {len(train_exitus)}")
    print(f"Casos test (recidiva_exitus = 2): {len(test_exitus)}")

if len(test_exitus) > 0 and len(train_exitus) > 0 and 'recidiva_exitus' in df.columns:
    # Preparar dades de training
    X_train = train_exitus[features].copy()
    y_train = train_exitus['recidiva_exitus'].copy()
    
    # Preparar dades de test
    X_test = test_exitus[features].copy()
    
    # Imputar valors perduts amb la mediana
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Normalitzar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Entrenar KNN
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Prediccions
    predictions = knn.predict(X_test_scaled)
    probabilities = knn.predict_proba(X_test_scaled)
    
    # Assignar prediccions
    test_indices = test_exitus.index
    for i, idx in enumerate(test_indices):
        df.loc[idx, 'recidiva_exitus'] = predictions[i]
        df.loc[idx, 'recidiva_exitus_confidence'] = probabilities[i].max()
    
    print(f"\n✓ Reclassificació completada!")
    print(f"  - Casos reclassificats a 0: {(predictions == 0).sum()}")
    print(f"  - Casos reclassificats a 1: {(predictions == 1).sum()}")
    print(f"  - Confiança mitjana: {probabilities.max(axis=1).mean():.2%}")
else:
    print("\n✗ No hi ha casos a reclassificar per recidiva_exitus")

# 5. RESUM FINAL
print("\n" + "="*80)
print("RESUM FINAL")
print("="*80)

print("\nDistribució RECIDIVA:")
print(f"  Original valor 0: {(df['recidiva_original'] == 0).sum()}")
print(f"  Original valor 1: {(df['recidiva_original'] == 1).sum()}")
print(f"  Original valor 2: {(df['recidiva_original'] == 2).sum()}")
print(f"  ---")
print(f"  Nova valor 0: {(df['recidiva'] == 0).sum()}")
print(f"  Nova valor 1: {(df['recidiva'] == 1).sum()}")
print(f"  Nova valor 2: {(df['recidiva'] == 2).sum()}")

print("\nDistribució RECIDIVA_EXITUS:")
if 'recidiva_exitus_original' in df.columns and 'recidiva_exitus' in df.columns:
    print(f"  Original valor 0: {(df['recidiva_exitus_original'] == 0).sum()}")
    print(f"  Original valor 1: {(df['recidiva_exitus_original'] == 1).sum()}")
    print(f"  Original valor 2: {(df['recidiva_exitus_original'] == 2).sum()}")
    print(f"  ---")
    print(f"  Nova valor 0: {(df['recidiva_exitus'] == 0).sum()}")
    print(f"  Nova valor 1: {(df['recidiva_exitus'] == 1).sum()}")
    print(f"  Nova valor 2: {(df['recidiva_exitus'] == 2).sum()}")
else:
    print("  ⚠️  Columna no disponible al dataset")

# 6. MOSTRAR CASOS RECLASSIFICATS
print("\n" + "="*80)
print("CASOS RECLASSIFICATS")
print("="*80)

reclassified = df[
    ((df['recidiva_original'] == 2) & (df['recidiva'] != 2))
].copy()

if 'recidiva_exitus' in df.columns and 'recidiva_exitus_original' in df.columns:
    reclassified_exitus = df[
        ((df['recidiva_exitus_original'] == 2) & (df['recidiva_exitus'] != 2))
    ]
    reclassified = pd.concat([reclassified, reclassified_exitus]).drop_duplicates()

if len(reclassified) > 0:
    print(f"\nTotal casos reclassificats: {len(reclassified)}")
    print("\nMostra dels primers 20 casos:")
    
    cols_mostrar = ['codigo_participante', 'recidiva_original', 'recidiva', 'recidiva_confidence']
    
    if 'recidiva_exitus' in df.columns:
        cols_mostrar.extend(['recidiva_exitus_original', 'recidiva_exitus', 'recidiva_exitus_confidence'])
    
    # Preparar per mostrar
    available_cols = [col for col in cols_mostrar if col in reclassified.columns]
    display_df = reclassified[available_cols].head(20).copy()
    
    # Formatar confiança
    if 'recidiva_confidence' in display_df.columns:
        display_df['recidiva_confidence'] = display_df['recidiva_confidence'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        )
    if 'recidiva_exitus_confidence' in display_df.columns:
        display_df['recidiva_exitus_confidence'] = display_df['recidiva_exitus_confidence'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        )
    
    print(display_df.to_string(index=False))
else:
    print("\nNo s'han reclassificat casos.")

# 7. GUARDAR RESULTATS
output_file = 'Dataset_NEST_Final_Reclassificat.csv'
df.to_csv(output_file, index=False)

print("\n" + "="*80)
print(f"✓ Resultats guardats a: {output_file}")
print("="*80)

# 8. ESTADÍSTIQUES ADDICIONALS
print("\nESTADÍSTIQUES ADICIONALS:")
print(f"  - Total registres: {len(df)}")
print(f"  - Casos amb diferencia_dias_reci_exit: {df['diferencia_dias_reci_exit'].notna().sum()}")
if df['diferencia_dias_reci_exit'].notna().sum() > 0:
    print(f"  - Mitjana diferencia_dias_reci_exit: {df['diferencia_dias_reci_exit'].mean():.1f} dies")
    print(f"  - Mediana diferencia_dias_reci_exit: {df['diferencia_dias_reci_exit'].median():.1f} dies")

print("\n" + "="*80)
print("PROCÉS COMPLETAT!")
print("="*80)

print("\n💾 Fitxer generat: Dataset_NEST_Final_Reclassificat.csv")
print("   Aquest fitxer conté:")
print("   - diferencia_dias_reci_exit (nova variable)")
print("   - recidiva i recidiva_exitus reclassificats")
print("   - recidiva_original i recidiva_exitus_original (valors originals)")
print("   - recidiva_confidence i recidiva_exitus_confidence (confiança de les prediccions)")
print("\n✓ Pots obrir-lo amb Excel o tornar a carregar-lo amb pandas!")