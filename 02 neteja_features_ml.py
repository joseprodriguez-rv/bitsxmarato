import pandas as pd
import numpy as np

# ============================================================================
# 1. CARREGAR DADES
# ============================================================================
print("="*80)
print("NETEJA I SELECCIÓ DE FEATURES PER MACHINE LEARNING")
print("="*80)

df = pd.read_csv('Dataset_NEST_Final_Reclassificat.csv')
print(f"\n✓ Dades carregades: {len(df)} pacients, {len(df.columns)} columnes")

# ============================================================================
# 2. MAPATGE: Noms originals → Variables clíniques importants
# ============================================================================
print("\n" + "="*80)
print("MAPATGE DE VARIABLES CLÍNIQUES")
print("="*80)

# Segons la teva llista de presentació
mapatge_variables = {
    # DEMOGRÀFIQUES
    'FN': 'Fecha_nacimiento',
    'edad': 'Edad',
    'imc': 'IMC',
    
    # DIAGNÒSTIC INICIAL
    'f_diag': 'Fecha_diagnostico',
    'tipo_histologico': 'Tipo_histologico',
    'grado_histologi': 'Grado_histologico',
    'valor_de_ca125': 'CA125',
    
    # CARACTERÍSTIQUES TUMOR
    'infiltracion_mi': 'Infiltracion_miometrial',
    'metasta_distan': 'Metastasis_distancia',
    'tamano_tumoral': 'Tamano_tumoral',
    'afectacion_linf': 'Afectacion_linfovascular',
    
    # ESTADIATGE PRE-QUIRÚRGIC
    'grupo_riesgo': 'Riesgo_preIQ',
    'estadiaje_pre_i': 'Estadiaje_preIQ',
    
    # TRACTAMENT
    'tto_NA': 'Tratamiento_neoadyuvante',
    'tto_1_quirugico': 'Tratamiento_quirurgico',
    'asa': 'ASA',
    
    # ANATOMIA PATOLÒGICA
    'histo_defin': 'Histologia_definitiva',
    'AP_centinela_pelvico': 'AP_centinela_pelvico',
    'AP_ganPelv': 'AP_ganglios_pelvicos',
    'AP_glanPaor': 'AP_ganglios_paraaorticos',
    
    # RECEPTORS
    'recep_est_porcent': 'Receptores_estrogenos',
    'rece_de_Ppor': 'Receptores_progesterona',
    
    # GENÈTICA
    'mut_pole': 'Mutacion_POLE',
    'p53_ihq': 'p53',
    'beta_cateninap': 'Beta_catenina',
    
    # ESTADIATGE DEFINITIU
    'FIGO2023': 'FIGO_2023',
    'grupo_de_riesgo_definitivo': 'Grupo_riesgo_definitivo',
    
    # TRACTAMENT ADJUVANT
    'Tributaria_a_Radioterapia': 'Tributaria_radioterapia',
    'rdt': 'Radioterapia',
    'bqt': 'Braquiterapia',
    'qt': 'Quimioterapia',
    
    # SEGUIMENT
    'Ultima_fecha': 'Fecha_ultima_visita',
    'recidiva': 'Recidiva',
    'estado': 'Estado_actual',
    'est_pcte': 'Estado_paciente',
    'causa_muerte': 'Causa_muerte',
    'f_muerte': 'Fecha_muerte',
    'libre_enferm': 'Libre_enfermedad',
    
    # RECIDIVA (DETALL)
    'numero_de_recid': 'Numero_recidivas',
    'num_recidiva': 'Num_recidiva',
    'fecha_de_recidi': 'Fecha_recidiva',
    'loc_recidiva_r01': 'Lugar_recidiva',
    'tto_recidiva': 'Tratamiento_recidiva',
    'Tt_recidiva_qx': 'Tratamiento_qx_recidiva',
    'Reseccion_macroscopica_com': 'Reseccion_completa',
    
    # VARIABLES CALCULADES
    'OS_MESES': 'Supervivencia_global_meses',
    'DFS_MESES': 'Supervivencia_libre_enfermedad_meses',
    'diferencia_dias_reci_exit': 'Dias_recidiva_exitus'
}

# ============================================================================
# 3. ANÀLISI DE DISPONIBILITAT
# ============================================================================
print("\n📊 Anàlisi de disponibilitat de variables:")
print("-"*80)

disponibilitat = []

for col_orig, col_clean in mapatge_variables.items():
    if col_orig in df.columns:
        n_disponibles = df[col_orig].notna().sum()
        pct = (n_disponibles / len(df)) * 100
        disponibilitat.append({
            'Variable_original': col_orig,
            'Variable_neta': col_clean,
            'N_disponibles': n_disponibles,
            'Percentatge': pct,
            'Utilitat': 'ALTA' if pct >= 80 else 'MITJANA' if pct >= 50 else 'BAIXA'
        })

df_disp = pd.DataFrame(disponibilitat).sort_values('Percentatge', ascending=False)

print("\n✅ Variables amb ALTA disponibilitat (≥80%):")
print(df_disp[df_disp['Utilitat'] == 'ALTA'][['Variable_neta', 'N_disponibles', 'Percentatge']].to_string(index=False))

print("\n⚠️  Variables amb MITJANA disponibilitat (50-80%):")
mitjanes = df_disp[df_disp['Utilitat'] == 'MITJANA']
if len(mitjanes) > 0:
    print(mitjanes[['Variable_neta', 'N_disponibles', 'Percentatge']].to_string(index=False))
else:
    print("   (Cap)")

print("\n❌ Variables amb BAIXA disponibilitat (<50%) - NO recomanades per ML:")
baixes = df_disp[df_disp['Utilitat'] == 'BAIXA']
if len(baixes) > 0:
    print(baixes[['Variable_neta', 'N_disponibles', 'Percentatge']].to_string(index=False))

# ============================================================================
# 4. SELECCIÓ DE FEATURES PER ML
# ============================================================================
print("\n" + "="*80)
print("SELECCIÓ DE FEATURES PER MACHINE LEARNING")
print("="*80)

# Criteri: Només variables amb ≥50% de disponibilitat
features_seleccionades_orig = df_disp[df_disp['Percentatge'] >= 50]['Variable_original'].tolist()
features_seleccionades_netes = df_disp[df_disp['Percentatge'] >= 50]['Variable_neta'].tolist()

print(f"\n✓ {len(features_seleccionades_orig)} variables seleccionades (≥50% disponibilitat)")

# Excloure variables objectiu i dates
variables_objectiu = ['recidiva', 'recidiva_exitus', 'estado', 'est_pcte', 'causa_muerte', 'libre_enferm']
variables_dates = ['FN', 'f_diag', 'fecha_de_recidi', 'f_muerte', 'Ultima_fecha', 'fecha_qx']
variables_id = ['codigo_participante']

features_ml = [f for f in features_seleccionades_orig 
               if f not in variables_objectiu 
               and f not in variables_dates 
               and f not in variables_id]

print(f"\n✓ {len(features_ml)} features finals per ML (excloses: objectius, dates, IDs)")

# ============================================================================
# 5. AGRUPACIÓ PER TIPUS CLÍNIC
# ============================================================================
print("\n" + "="*80)
print("AGRUPACIÓ DE FEATURES PER TIPUS CLÍNIC")
print("="*80)

grupos_variables = {
    'Demogràfiques': ['edad', 'imc'],
    'Diagnòstic_inicial': ['tipo_histologico', 'grado_histologi', 'valor_de_ca125'],
    'Característiques_tumor': ['infiltracion_mi', 'metasta_distan', 'tamano_tumoral', 'afectacion_linf'],
    'Estadiatge': ['grupo_riesgo', 'estadiaje_pre_i', 'FIGO2023', 'grupo_de_riesgo_definitivo'],
    'Tractament': ['tto_NA', 'tto_1_quirugico', 'asa', 'Tributaria_a_Radioterapia', 'rdt', 'bqt', 'qt'],
    'Anatomia_patològica': ['histo_defin', 'AP_centinela_pelvico', 'AP_ganPelv', 'AP_glanPaor'],
    'Biomarcadors': ['recep_est_porcent', 'rece_de_Ppor', 'mut_pole', 'p53_ihq', 'beta_cateninap'],
    'Supervivència': ['OS_MESES', 'DFS_MESES', 'diferencia_dias_reci_exit']
}

print("\nDistribució per grups:")
for grup, vars_grup in grupos_variables.items():
    vars_disponibles = [v for v in vars_grup if v in features_ml]
    print(f"  📁 {grup:25} → {len(vars_disponibles):2} variables")

# ============================================================================
# 6. SEPARAR FEATURES NUMÈRIQUES I CATEGÒRIQUES
# ============================================================================
print("\n" + "="*80)
print("CLASSIFICACIÓ: NUMÈRIQUES vs CATEGÒRIQUES")
print("="*80)

features_numericas = []
features_categoricas = []

for feat in features_ml:
    if feat in df.columns:
        # Criteri: Si té <15 valors únics i no és numèric continu → Categòrica
        n_unique = df[feat].nunique()
        dtype = df[feat].dtype
        
        if dtype in ['object', 'category'] or n_unique <= 10:
            features_categoricas.append(feat)
        else:
            features_numericas.append(feat)

print(f"\n✓ {len(features_numericas)} features NUMÈRIQUES:")
for f in features_numericas:
    print(f"   · {f}")

print(f"\n✓ {len(features_categoricas)} features CATEGÒRIQUES:")
for f in features_categoricas:
    n_cat = df[f].nunique()
    print(f"   · {f:30} → {n_cat} categories")

# ============================================================================
# 7. GUARDAR CONFIGURACIÓ
# ============================================================================
print("\n" + "="*80)
print("GUARDANT CONFIGURACIÓ")
print("="*80)

# Crear diccionari de configuració
config = {
    'features_numericas': features_numericas,
    'features_categoricas': features_categoricas,
    'features_todas': features_ml,
    'grupos_variables': grupos_variables,
    'variables_objectiu': variables_objectiu
}

# Guardar com a Python
with open('config_ml_features.py', 'w', encoding='utf-8') as f:
    f.write("# Configuració de Features per Machine Learning\n")
    f.write("# Generat automàticament\n\n")
    
    f.write(f"features_numericas = {features_numericas}\n\n")
    f.write(f"features_categoricas = {features_categoricas}\n\n")
    f.write(f"features_todas = {features_ml}\n\n")
    f.write(f"grupos_variables = {grupos_variables}\n\n")
    f.write(f"variables_objectiu = {variables_objectiu}\n")

print("✓ Configuració guardada a: config_ml_features.py")

# Guardar resum en CSV
df_disp.to_csv('resum_disponibilitat_variables.csv', index=False)
print("✓ Resum disponibilitat guardat a: resum_disponibilitat_variables.csv")

# ============================================================================
# 8. RESUM FINAL
# ============================================================================
print("\n" + "="*80)
print("✅ RESUM FINAL")
print("="*80)
print(f"Total pacients:              {len(df)}")
print(f"Features disponibles:        {len(features_seleccionades_orig)}")
print(f"Features per ML:             {len(features_ml)}")
print(f"  - Numèriques:              {len(features_numericas)}")
print(f"  - Categòriques:            {len(features_categoricas)}")
print(f"\nVariables objectiu:          {len(variables_objectiu)}")
for obj in variables_objectiu:
    if obj in df.columns:
        print(f"  - {obj:20} → Disponibilitat: {(df[obj].notna().sum()/len(df)*100):.1f}%")

print("\n💡 RECOMANACIONS:")
print("   1. Utilitza 'features_numericas' i 'features_categoricas' per al preprocessor")
print("   2. Variables amb <50% disponibilitat → No recomanades")
print("   3. 'grupos_variables' per anàlisi SHAP agrupat")
print("   4. Importa config_ml_features.py al teu codi de ML")

print("\n" + "="*80)
print("✅ PROCÉS COMPLETAT!")
print("="*80)
