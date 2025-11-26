# scripts/verificar_dataset_final.py

import pandas as pd
import numpy as np
from pathlib import Path

def verificar_dataset_final(csv_path: str = "datos_salida/finales/dataset_entrenamiento_final.csv"):
    """
    Verificación exhaustiva del dataset final antes del entrenamiento.
    Detecta problemas de leakage, escalas, features faltantes, etc.
    """
    
    print("="*100)
    print("🔍 VERIFICACIÓN EXHAUSTIVA DEL DATASET FINAL")
    print("="*100)
    
    # 1. CARGAR DATASET
    print("\n[1] CARGANDO DATASET...")
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    print(f"   ✅ Cargado: {len(df)} filas, {df.shape[1]} columnas")
    
    # 2. COLUMNAS Y TARGET
    print("\n[2] ANÁLISIS DE COLUMNAS:")
    print(f"   Total de columnas: {df.shape[1]}")
    print(f"   Columnas: {list(df.columns)}\n")
    
    # Columnas auxiliares esperadas
    aux_cols = ['tm_id', 'ss_id', 'nombre_jugador', 'season']
    target_col = 'riesgo_fichaje'
    
    print("   Columnas auxiliares detectadas:")
    for col in aux_cols:
        status = "✅" if col in df.columns else "❌"
        print(f"      {status} {col}")
    
    print(f"\n   Target '{target_col}':")
    if target_col in df.columns:
        print(f"      ✅ Presente")
        print(f"      Distribución: {df[target_col].value_counts().to_dict()}")
    else:
        print(f"      ❌ NO ENCONTRADO")
        return
    
    # 3. FEATURES PARA EL MODELO
    print("\n[3] FEATURES DISPONIBLES PARA EL MODELO:")
    exclude = {'tm_id', 'ss_id', 'nombre_jugador', 'season', 'riesgo_fichaje'}
    features = [c for c in df.columns if c not in exclude]
    
    print(f"   Total de features: {len(features)}")
    print(f"   Features: {features}\n")
    
    # 4. VERIFICAR VARIABLES PROBLEMÁTICAS
    print("[4] VERIFICACIÓN DE VARIABLES CRÍTICAS:")
    
    # A. score_ponderado
    if 'score_ponderado' in features:
        print("   🔴 CRÍTICO: 'score_ponderado' está en el dataset")
        print("      ⚠️ ESTO ES DATA LEAKAGE - DEBE SER ELIMINADO")
    else:
        print("   ✅ 'score_ponderado' NO está en el dataset (correcto)")
    
    # B. Variables con sufijo _futuro
    futuro_cols = [c for c in features if '_futuro' in c]
    if futuro_cols:
        print(f"   🔴 CRÍTICO: {len(futuro_cols)} columnas con '_futuro' detectadas:")
        for col in futuro_cols:
            print(f"      - {col}")
        print("      ⚠️ ESTO ES DATA LEAKAGE - DEBEN SER ELIMINADAS")
    else:
        print("   ✅ No hay columnas con '_futuro' (correcto)")
    
    # C. minutesPlayed
    if 'minutesPlayed' in features:
        print("\n   ⚠️ 'minutesPlayed' está en las features:")
        
        # Estadísticas
        min_val = df['minutesPlayed'].min()
        max_val = df['minutesPlayed'].max()
        mean_val = df['minutesPlayed'].mean()
        print(f"      - Rango: [{min_val:.0f}, {max_val:.0f}]")
        print(f"      - Media: {mean_val:.0f}")
        
        # Correlación con target
        corr = df[['minutesPlayed', target_col]].corr().iloc[0, 1]
        print(f"      - Correlación con {target_col}: {corr:.3f}")
        
        if abs(corr) > 0.7:
            print(f"      🔴 ALERTA: Correlación muy alta ({corr:.3f})")
            print(f"         Posible leakage o proxy del target")
        elif abs(corr) > 0.4:
            print(f"      ⚠️ Correlación moderada ({corr:.3f})")
            print(f"         Verificar que sea del PASADO")
        else:
            print(f"      ✅ Correlación baja ({corr:.3f}) - Parece seguro")
    else:
        print("\n   ✅ 'minutesPlayed' NO está en features")
    
    # 5. TIPOS DE DATOS
    print("\n[5] TIPOS DE DATOS:")
    print(f"   Numéricas: {df[features].select_dtypes(include=[np.number]).shape[1]}")
    print(f"   Categóricas: {df[features].select_dtypes(include=['object']).shape[1]}")
    
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        print(f"   ⚠️ Columnas categóricas detectadas (deberían ser numéricas):")
        for col in cat_cols:
            print(f"      - {col}: {df[col].unique()[:5]}")
    
    # 6. VALORES NULOS
    print("\n[6] VALORES NULOS:")
    nulos = df[features].isnull().sum()
    nulos_presentes = nulos[nulos > 0]
    
    if len(nulos_presentes) == 0:
        print("   ✅ No hay valores nulos en las features")
    else:
        print(f"   ⚠️ {len(nulos_presentes)} features con valores nulos:")
        for col, count in nulos_presentes.items():
            pct = (count / len(df)) * 100
            print(f"      - {col}: {count} ({pct:.1f}%)")
    
    # 7. ESCALAS DE FEATURES NUMÉRICAS
    print("\n[7] ANÁLISIS DE ESCALAS (Features numéricas):")
    numeric_features = df[features].select_dtypes(include=[np.number]).columns
    
    escalas = []
    for col in numeric_features:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        std_val = df[col].std()
        rango = max_val - min_val
        
        escalas.append({
            'feature': col,
            'min': min_val,
            'max': max_val,
            'rango': rango,
            'mean': mean_val,
            'std': std_val
        })
    
    df_escalas = pd.DataFrame(escalas).sort_values('rango', ascending=False)
    
    print("\n   Top 10 features por rango (mayor variabilidad):")
    print(df_escalas.head(10)[['feature', 'min', 'max', 'rango']].to_string(index=False))
    
    # Detectar si necesitan normalización
    rangos_grandes = df_escalas[df_escalas['rango'] > 100]
    rangos_pequeños = df_escalas[df_escalas['rango'] < 10]
    
    print(f"\n   Features con rango > 100: {len(rangos_grandes)}")
    print(f"   Features con rango < 10: {len(rangos_pequeños)}")
    
    if len(rangos_grandes) > 0 and len(rangos_pequeños) > 0:
        print("\n   ⚠️ RECOMENDACIÓN: Las features tienen escalas muy diferentes")
        print("      → StandardScaler necesario para LogReg, AdaBoost, NB")
        print("      → NO necesario para RandomForest, XGBoost, LightGBM")
    
    # 8. DUMMIES DE POSICIÓN Y NACIONALIDAD
    print("\n[8] VARIABLES DUMMIES:")
    
    dummies_esperadas = {
        'posición': ['pos_Delantero', 'pos_Mediocampista'],
        'nacionalidad': ['nac_Perú', 'nac_Argentina', 'nac_Colombia', 'nac_Uruguay']
    }
    
    for categoria, dummies in dummies_esperadas.items():
        print(f"\n   {categoria.capitalize()}:")
        for dummy in dummies:
            if dummy in features:
                unicos = df[dummy].unique()
                conteo = df[dummy].value_counts().to_dict()
                print(f"      ✅ {dummy}: valores={unicos}, distribución={conteo}")
            else:
                print(f"      ❌ {dummy}: NO ENCONTRADA")
    
    # 9. DISTRIBUCIÓN DE CLASES POR TEMPORADA
    print("\n[9] DISTRIBUCIÓN DE CLASES POR TEMPORADA:")
    
    temporadas = sorted(df['season'].unique(), key=lambda s: tuple(map(int, s.split('_'))))
    
    for temp in temporadas:
        df_temp = df[df['season'] == temp]
        conteo = df_temp[target_col].value_counts().to_dict()
        total = len(df_temp)
        pct_1 = (conteo.get(1.0, 0) / total) * 100 if total > 0 else 0
        print(f"   {temp}: n={total:>3} | Riesgo Alto={conteo.get(1.0, 0):>3} ({pct_1:.1f}%) | Bajo={conteo.get(0.0, 0):>3}")
    
    # 10. CORRELACIÓN ALTA ENTRE FEATURES (Multicolinealidad)
    print("\n[10] DETECCIÓN DE MULTICOLINEALIDAD:")
    
    numeric_feats = df[features].select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_feats].corr().abs()
    
    # Encontrar pares con correlación > 0.9
    umbral = 0.9
    high_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > umbral:
                high_corr.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlacion': corr_matrix.iloc[i, j]
                })
    
    if high_corr:
        print(f"   ⚠️ {len(high_corr)} pares de features con correlación > {umbral}:")
        for pair in high_corr[:10]:  # Mostrar solo top 10
            print(f"      - {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlacion']:.3f}")
    else:
        print(f"   ✅ No hay multicolinealidad severa (correlación > {umbral})")
    
    # 11. RESUMEN FINAL
    print("\n" + "="*100)
    print("📊 RESUMEN Y RECOMENDACIONES:")
    print("="*100)
    
    problemas = []
    warnings = []
    ok = []
    
    # Verificaciones
    if 'score_ponderado' in features:
        problemas.append("🔴 score_ponderado presente (DATA LEAKAGE)")
    else:
        ok.append("✅ score_ponderado ausente")
    
    if futuro_cols:
        problemas.append(f"🔴 {len(futuro_cols)} columnas con '_futuro' (DATA LEAKAGE)")
    else:
        ok.append("✅ No hay columnas '_futuro'")
    
    if 'minutesPlayed' in features:
        corr = df[['minutesPlayed', target_col]].corr().iloc[0, 1]
        if abs(corr) > 0.7:
            problemas.append(f"🔴 minutesPlayed con correlación alta ({corr:.3f})")
        elif abs(corr) > 0.4:
            warnings.append(f"⚠️ minutesPlayed con correlación moderada ({corr:.3f})")
        else:
            ok.append(f"✅ minutesPlayed con correlación baja ({corr:.3f})")
    
    if len(nulos_presentes) > 0:
        warnings.append(f"⚠️ {len(nulos_presentes)} features con valores nulos")
    else:
        ok.append("✅ No hay valores nulos")
    
    if len(rangos_grandes) > 0 and len(rangos_pequeños) > 0:
        warnings.append("⚠️ Features con escalas muy diferentes (normalización recomendada)")
    
    # Mostrar resumen
    if problemas:
        print("\n🔴 PROBLEMAS CRÍTICOS:")
        for p in problemas:
            print(f"   {p}")
    
    if warnings:
        print("\n⚠️ ADVERTENCIAS:")
        for w in warnings:
            print(f"   {w}")
    
    if ok:
        print("\n✅ VERIFICACIONES PASADAS:")
        for o in ok:
            print(f"   {o}")
    
    print("\n" + "="*100)
    print(f"💾 Dataset analizado: {csv_path}")
    print(f"📊 {len(features)} features disponibles para modelado")
    print("="*100 + "\n")
    
    return {
        'n_features': len(features),
        'features': features,
        'problemas': problemas,
        'warnings': warnings
    }

if __name__ == "__main__":
    verificar_dataset_final()