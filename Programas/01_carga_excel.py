import pandas as pd
import numpy as np
import os
import sys

# --- 1. CONFIGURACIÓ ---
carpeta = os.path.dirname(os.path.abspath(__file__))
fitxer_entrada = os.path.join(carpeta, 'datos.xlsx')
fitxer_csv = os.path.join(carpeta, 'Dataset_NEST_Final.csv')
fitxer_pdf = os.path.join(carpeta, 'Vista_Datos.pdf')

print(f"⏳ Llegint dades de: {fitxer_entrada}")

try:
    df = pd.read_excel(fitxer_entrada, engine='openpyxl')
except ImportError:
    print("❌ ERROR: Falta 'openpyxl'. Instal·la'l amb: pip install openpyxl")
    sys.exit()
except FileNotFoundError:
    print("❌ ERROR: No trobo 'datos.xlsx'.")
    sys.exit()

print(f"✓ Llegides {len(df)} files i {len(df.columns)} columnes")

# --- 2. VERIFICAR SI DIFERENCIA_DIAS_RECI_EXIT EXISTEIX ---
print("\n🔍 Verificant columnes crítiques...")
if 'diferencia_dias_reci_exit' in df.columns:
    print("   ✓ diferencia_dias_reci_exit trobada!")
    print(f"     Valors disponibles: {df['diferencia_dias_reci_exit'].notna().sum()}/{len(df)}")
else:
    print("   ⚠️  diferencia_dias_reci_exit NO trobada a l'Excel original")

if 'recidiva_exitus' in df.columns:
    print("   ✓ recidiva_exitus trobada!")
    print(f"     Valors disponibles: {df['recidiva_exitus'].notna().sum()}/{len(df)}")
else:
    print("   ⚠️  recidiva_exitus NO trobada a l'Excel original")

# --- 3. SELECCIÓ I CÀLCULS ---
print("\n⚙️ Processant dades i calculant supervivència...")

variables_desitjades = [
    'codigo_participante', 'FN', 'edad', 'imc', 'f_diag', 'fecha_qx',
    'tipo_histologico', 'grado_histologi', 'valor_de_ca125', 
    'infiltracion_mi', 'ecotv_infiltobj', 'ecotv_infiltsub',
    'metasta_distan', 'grupo_riesgo', 'estadiaje_pre_i', 
    'tto_NA', 'tto_1_quirugico', 'asa',
    'histo_defin', 'tamano_tumoral', 'afectacion_linf',
    'AP_centinela_pelvico', 'AP_ganPelv', 'AP_glanPaor',
    'recep_est_porcent', 'rece_de_Ppor', 'beta_cateninap', 
    'estudio_genetico', 'mut_pole', 'p53_ihq', 
    'FIGO2023', 'estadificacion', 'grupo_de_riesgo_definitivo',
    'Tributaria_a_Radioterapia', 'rdt', 'bqt', 'qt', 'Tratamiento_sistemico_realizad',
    'visita_control', 'Ultima_fecha', 
    'recidiva', 'recidiva_exitus', 'diferencia_dias_reci_exit',  # ← AFEGIDES AQUÍ!
    'estado', 'est_pcte',
    'causa_muerte', 'f_muerte', 'libre_enferm', 
    'numero_de_recid', 'num_recidiva', 'fecha_de_recidi',
    'loc_recidiva', 'loc_recidiva_r01', 
    'tto_recidiva', 'Tt_recidiva_qx', 'Reseccion_macroscopica_com'
]

cols_existents = [c for c in variables_desitjades if c in df.columns]
print(f"   Variables seleccionades: {len(cols_existents)}/{len(variables_desitjades)}")

# Mostrar quines NO s'han trobat
cols_no_trobades = [c for c in variables_desitjades if c not in df.columns]
if cols_no_trobades:
    print(f"   ⚠️  Variables NO trobades a l'Excel ({len(cols_no_trobades)}):")
    for col in cols_no_trobades:
        print(f"      - {col}")

df_final = df[cols_existents].copy()

# Conversió dates
cols_data = ['fecha_qx', 'fecha_de_recidi', 'f_muerte', 'visita_control', 'Ultima_fecha']
for col in cols_data:
    if col in df_final.columns:
        df_final[col] = pd.to_datetime(df_final[col], errors='coerce')

# Càlcul OS i DFS
if 'fecha_qx' in df_final.columns:
    # OS
    data_fin_os = df_final['f_muerte'].fillna(df_final.get('visita_control', df_final.get('Ultima_fecha')))
    df_final['OS_MESES'] = (data_fin_os - df_final['fecha_qx']).dt.days / 30.44
    
    # DFS
    data_fin_dfs = df_final['fecha_de_recidi'].fillna(df_final.get('visita_control', df_final.get('Ultima_fecha')))
    df_final['DFS_MESES'] = (data_fin_dfs - df_final['fecha_qx']).dt.days / 30.44

# --- 4. SI DIFERENCIA_DIAS_RECI_EXIT NO EXISTIA, INTENTAR CREAR-LA ---
if 'diferencia_dias_reci_exit' not in df_final.columns:
    print("\n⚙️ diferencia_dias_reci_exit no existeix, intentant calcular-la...")
    if 'fecha_de_recidi' in df_final.columns and 'f_muerte' in df_final.columns:
        def calcular_dif(row):
            try:
                if pd.notna(row['fecha_de_recidi']) and pd.notna(row['f_muerte']):
                    return (row['f_muerte'] - row['fecha_de_recidi']).days
            except:
                pass
            return np.nan
        
        df_final['diferencia_dias_reci_exit'] = df_final.apply(calcular_dif, axis=1)
        print(f"   ✓ Calculada: {df_final['diferencia_dias_reci_exit'].notna().sum()} casos")
    else:
        print("   ✗ No es pot calcular (falten dates)")

# --- 5. EXPORTAR A CSV ---
print(f"\n💾 Guardant CSV a: {fitxer_csv}")
df_final.to_csv(fitxer_csv, index=False, sep=',')
print(f"   ✓ Guardat amb {len(df_final)} files i {len(df_final.columns)} columnes")

# --- 6. RESUM FINAL ---
print("\n" + "="*80)
print("✅ PROCÉS FINALITZAT")
print("="*80)
print(f"📂 CSV: {fitxer_csv}")
print(f"\n📊 VARIABLES CRÍTIQUES AL CSV FINAL:")
if 'recidiva' in df_final.columns:
    print(f"   ✓ recidiva: {df_final['recidiva'].notna().sum()}/{len(df_final)}")
if 'recidiva_exitus' in df_final.columns:
    print(f"   ✓ recidiva_exitus: {df_final['recidiva_exitus'].notna().sum()}/{len(df_final)}")
if 'diferencia_dias_reci_exit' in df_final.columns:
    print(f"   ✓ diferencia_dias_reci_exit: {df_final['diferencia_dias_reci_exit'].notna().sum()}/{len(df_final)}")
if 'OS_MESES' in df_final.columns:
    print(f"   ✓ OS_MESES: {df_final['OS_MESES'].notna().sum()}/{len(df_final)}")
if 'DFS_MESES' in df_final.columns:
    print(f"   ✓ DFS_MESES: {df_final['DFS_MESES'].notna().sum()}/{len(df_final)}")
print("="*80)