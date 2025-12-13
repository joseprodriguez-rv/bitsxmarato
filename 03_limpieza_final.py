import pandas as pd

print("🧹 INICIANDO LIMPIEZA DE DATOS 'DESCONOCIDOS' (Valor 2)...")

# 1. Cargar los archivos que generaste en el paso anterior
X = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')

# (Opcional) Si generaste el de tiempo, cárgalo también
try:
    y_time = pd.read_csv('y_time.csv')
    tiene_tiempo = True
except:
    tiene_tiempo = False
    print("⚠️ No se encontró archivo de tiempos, trabajando solo con X e y.")

# Comprobación de seguridad: Deben tener las mismas filas
if len(X) != len(y):
    print("❌ ERROR GRAVE: X e y no tienen el mismo tamaño. Vuelve a ejecutar el paso 02.")
else:
    print(f"📉 Datos iniciales: {len(y)} pacientes.")
    
    # 2. IDENTIFICAR LAS FILAS "MALAS" (Donde recidiva es 2)
    # Creamos un filtro (máscara) con las filas que QUEREMOS mantener (las que NO son 2)
    filtro_buenos = y['recidiva'] != 2
    
    # 3. APLICAR EL FILTRO A TODO
    X_limpio = X[filtro_buenos]
    y_limpio = y[filtro_buenos]
    
    print(f"📉 Datos después de limpiar: {len(y_limpio)} pacientes.")
    print(f"🗑️ Se han eliminado {len(y) - len(y_limpio)} pacientes con estado 'Desconocido'.")
    
    # Verificación final
    print("Valores únicos en 'y' ahora:", y_limpio['recidiva'].unique()) # Debería salir solo 0 y 1
    
    # 4. GUARDAR
    X_limpio.to_csv('X_train_final.csv', index=False)
    y_limpio.to_csv('y_train_final.csv', index=False)
    
    if tiene_tiempo:
        y_time_limpio = y_time[filtro_buenos]
        y_time_limpio.to_csv('y_time_final.csv', index=False)

    print("\n✅ ¡LISTO! Usa 'X_train_final.csv' y 'y_train_final.csv' para el modelo.")