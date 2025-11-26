import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json

from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_COLS = {
    "tm_id", "ss_id", "nombre_jugador", "season",
    "riesgo_fichaje", "score_ponderado", "posicion_grupo"
}

# =============================================================================
# 1. FEATURE IMPORTANCE DEL MODELO
# =============================================================================

def analizar_feature_importance(model, feature_names, threshold=0.005):
    """
    Identifica features con importancia < threshold (0.5% por defecto).
    
    Args:
        model: Modelo LightGBM entrenado
        feature_names: Lista de nombres de features
        threshold: Umbral mínimo de importancia (0.005 = 0.5%)
    
    Returns:
        dict con features irrelevantes y DataFrame completo
    """
    logger.info("=" * 80)
    logger.info("1️⃣ ANÁLISIS DE FEATURE IMPORTANCE")
    logger.info("=" * 80)
    
    # Obtener importancia
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'steps'):  # Pipeline
        clf = model.steps[-1][1]
        importances = clf.feature_importances_
    else:
        raise ValueError("Modelo no soporta feature_importances_")
    
    # Normalizar a suma = 1
    importances_norm = importances / importances.sum()
    
    # Crear DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances_norm,
        'importance_pct': importances_norm * 100
    }).sort_values('importance', ascending=False)
    
    # Identificar features irrelevantes
    irrelevant = df_importance[df_importance['importance'] < threshold]['feature'].tolist()
    
    logger.info(f"✅ Total features: {len(feature_names)}")
    logger.info(f"⚠️ Features con importancia < {threshold*100:.1f}%: {len(irrelevant)}")
    logger.info(f"📊 Top 10 features más importantes:")
    for idx, row in df_importance.head(10).iterrows():
        logger.info(f"   {row['feature']:40s} {row['importance_pct']:6.2f}%")
    
    if irrelevant:
        logger.info(f"\n🚨 Features IRRELEVANTES (importancia < {threshold*100:.1f}%):")
        for feat in irrelevant:
            imp_pct = df_importance[df_importance['feature'] == feat]['importance_pct'].values[0]
            logger.info(f"   {feat:40s} {imp_pct:6.3f}%")
    
    return {
        'irrelevant_features': irrelevant,
        'df_importance': df_importance,
        'threshold': threshold
    }

# =============================================================================
# 2. VARIANZA CASI CERO
# =============================================================================

def analizar_varianza_baja(df, feature_names, threshold=0.01):
    """
    Identifica features con varianza muy baja (casi constantes).
    
    Args:
        df: DataFrame con los datos
        feature_names: Lista de features numéricas
        threshold: Umbral mínimo de varianza
    
    Returns:
        dict con features de baja varianza
    """
    logger.info("\n" + "=" * 80)
    logger.info("2️⃣ ANÁLISIS DE VARIANZA")
    logger.info("=" * 80)
    
    variances = {}
    low_variance = []
    
    for feat in feature_names:
        if feat in df.columns:
            var = df[feat].var()
            variances[feat] = var
            if var < threshold:
                low_variance.append(feat)
    
    df_variance = pd.DataFrame({
        'feature': list(variances.keys()),
        'variance': list(variances.values())
    }).sort_values('variance')
    
    logger.info(f"✅ Features analizadas: {len(variances)}")
    logger.info(f"⚠️ Features con varianza < {threshold}: {len(low_variance)}")
    
    if low_variance:
        logger.info(f"\n🚨 Features con VARIANZA MUY BAJA:")
        for feat in low_variance:
            var = variances[feat]
            logger.info(f"   {feat:40s} varianza = {var:.6f}")
    else:
        logger.info("✅ Todas las features tienen varianza suficiente")
    
    return {
        'low_variance_features': low_variance,
        'df_variance': df_variance,
        'threshold': threshold
    }

# =============================================================================
# 3. CORRELACIÓN ALTA (REDUNDANCIA)
# =============================================================================

def analizar_correlacion_alta(df, feature_names, threshold=0.90):
    """
    Identifica pares de features altamente correlacionadas (redundantes).
    
    Args:
        df: DataFrame con los datos
        feature_names: Lista de features numéricas
        threshold: Umbral de correlación (0.90 = 90%)
    
    Returns:
        dict con pares correlacionados y sugerencias de eliminación
    """
    logger.info("\n" + "=" * 80)
    logger.info("3️⃣ ANÁLISIS DE CORRELACIÓN (MULTICOLINEALIDAD)")
    logger.info("=" * 80)
    
    # Filtrar solo features que existen en df
    features_disponibles = [f for f in feature_names if f in df.columns]
    
    # Calcular matriz de correlación
    corr_matrix = df[features_disponibles].corr().abs()
    
    # Encontrar pares altamente correlacionados
    high_corr_pairs = []
    features_to_drop = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > threshold:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': corr_val
                })
                # Sugerencia: eliminar feat2 (arbitrario, luego se puede ajustar)
                features_to_drop.add(feat2)
    
    logger.info(f"✅ Features analizadas: {len(features_disponibles)}")
    logger.info(f"⚠️ Pares con correlación > {threshold*100:.0f}%: {len(high_corr_pairs)}")
    
    if high_corr_pairs:
        logger.info(f"\n🚨 PARES ALTAMENTE CORRELACIONADOS:")
        for pair in high_corr_pairs:
            logger.info(f"   {pair['feature1']:30s} ↔ {pair['feature2']:30s} | r = {pair['correlation']:.3f}")
        
        logger.info(f"\n💡 SUGERENCIA: Considerar eliminar estas features (una de cada par):")
        for feat in features_to_drop:
            logger.info(f"   - {feat}")
    else:
        logger.info("✅ No se encontraron pares con correlación muy alta")
    
    return {
        'high_corr_pairs': high_corr_pairs,
        'suggested_to_drop': list(features_to_drop),
        'corr_matrix': corr_matrix,
        'threshold': threshold
    }

# =============================================================================
# 4. FEATURES SPARSE (MAYORMENTE CEROS)
# =============================================================================

def analizar_features_sparse(df, feature_names, threshold=0.90):
    """
    Identifica features que son 0 en > threshold% de los registros.
    
    Args:
        df: DataFrame con los datos
        feature_names: Lista de features numéricas
        threshold: Porcentaje de ceros (0.90 = 90%)
    
    Returns:
        dict con features sparse
    """
    logger.info("\n" + "=" * 80)
    logger.info("4️⃣ ANÁLISIS DE FEATURES SPARSE (MAYORMENTE CEROS)")
    logger.info("=" * 80)
    
    sparse_features = []
    sparsity_stats = {}
    
    for feat in feature_names:
        if feat in df.columns:
            zero_pct = (df[feat] == 0).sum() / len(df)
            sparsity_stats[feat] = zero_pct
            if zero_pct > threshold:
                sparse_features.append(feat)
    
    df_sparsity = pd.DataFrame({
        'feature': list(sparsity_stats.keys()),
        'zero_percentage': list(sparsity_stats.values())
    }).sort_values('zero_percentage', ascending=False)
    
    logger.info(f"✅ Features analizadas: {len(sparsity_stats)}")
    logger.info(f"⚠️ Features con > {threshold*100:.0f}% ceros: {len(sparse_features)}")
    
    if sparse_features:
        logger.info(f"\n🚨 Features SPARSE (mayoría ceros):")
        for feat in sparse_features:
            pct = sparsity_stats[feat] * 100
            logger.info(f"   {feat:40s} {pct:.1f}% ceros")
    else:
        logger.info("✅ No hay features excesivamente sparse")
    
    return {
        'sparse_features': sparse_features,
        'df_sparsity': df_sparsity,
        'threshold': threshold
    }

# =============================================================================
# 5. RESUMEN Y RECOMENDACIONES
# =============================================================================

def generar_recomendaciones(results_dict):
    """
    Consolida todos los análisis y genera lista final de features a eliminar.
    
    Args:
        results_dict: Diccionario con resultados de todos los análisis
    
    Returns:
        dict con recomendaciones finales
    """
    logger.info("\n" + "=" * 80)
    logger.info("5️⃣ RESUMEN Y RECOMENDACIONES FINALES")
    logger.info("=" * 80)
    
    # Consolidar features problemáticas
    all_problematic = set()
    
    # 1. Irrelevantes por importancia
    irrelevant = set(results_dict.get('importance', {}).get('irrelevant_features', []))
    all_problematic.update(irrelevant)
    
    # 2. Varianza baja
    low_var = set(results_dict.get('variance', {}).get('low_variance_features', []))
    all_problematic.update(low_var)
    
    # 3. Correlación alta
    high_corr = set(results_dict.get('correlation', {}).get('suggested_to_drop', []))
    all_problematic.update(high_corr)
    
    # 4. Sparse
    sparse = set(results_dict.get('sparsity', {}).get('sparse_features', []))
    all_problematic.update(sparse)
    
    # Generar tabla resumen
    summary_data = []
    for feat in all_problematic:
        reasons = []
        if feat in irrelevant:
            reasons.append("Importancia < 0.5%")
        if feat in low_var:
            reasons.append("Varianza baja")
        if feat in high_corr:
            reasons.append("Correlación alta")
        if feat in sparse:
            reasons.append("Mayoría ceros")
        
        summary_data.append({
            'feature': feat,
            'razones': ' | '.join(reasons),
            'num_razones': len(reasons)
        })
    
    df_summary = pd.DataFrame(summary_data).sort_values('num_razones', ascending=False)
    
    logger.info(f"\n📊 RESUMEN:")
    logger.info(f"   Total features originales: {results_dict.get('total_features', 'N/A')}")
    logger.info(f"   Features problemáticas identificadas: {len(all_problematic)}")
    logger.info(f"   Features recomendadas a MANTENER: {results_dict.get('total_features', 42) - len(all_problematic)}")
    
    logger.info(f"\n🚨 FEATURES RECOMENDADAS PARA ELIMINACIÓN:")
    logger.info(f"   (Ordenadas por número de problemas detectados)\n")
    for idx, row in df_summary.iterrows():
        logger.info(f"   [{row['num_razones']} problemas] {row['feature']:35s} → {row['razones']}")
    
    logger.info(f"\n💡 RECOMENDACIÓN FINAL:")
    logger.info(f"   Eliminar las {len(all_problematic)} features listadas arriba")
    logger.info(f"   Nuevo total de features: {results_dict.get('total_features', 42) - len(all_problematic)}")
    logger.info(f"   Ratio registros/features: {896 / (results_dict.get('total_features', 42) - len(all_problematic)):.1f}")
    
    return {
        'features_to_drop': list(all_problematic),
        'df_summary': df_summary,
        'num_to_drop': len(all_problematic)
    }

# =============================================================================
# 6. VISUALIZACIONES
# =============================================================================

def generar_visualizaciones(results_dict, output_dir):
    """Genera gráficos de los análisis."""
    logger.info("\n" + "=" * 80)
    logger.info("6️⃣ GENERANDO VISUALIZACIONES")
    logger.info("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance
    if 'importance' in results_dict:
        df_imp = results_dict['importance']['df_importance']
        threshold = results_dict['importance']['threshold']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Top 25 features
        df_plot = df_imp.head(25)
        colors = ['red' if x < threshold else 'steelblue' for x in df_plot['importance']]
        
        ax.barh(range(len(df_plot)), df_plot['importance_pct'], color=colors, edgecolor='black')
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.axvline(threshold * 100, color='red', linestyle='--', linewidth=2, label=f'Umbral {threshold*100:.1f}%')
        ax.set_xlabel('Importancia (%)', fontweight='bold')
        ax.set_title('Top 25 Features por Importancia\n(Rojas: < 0.5% → Candidatas a eliminación)', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '01_feature_importance.png', dpi=300)
        plt.close()
        logger.info(f"   ✅ Guardado: 01_feature_importance.png")
    
    # 2. Correlation Heatmap (Solo top 20)
    if 'correlation' in results_dict:
        corr_matrix = results_dict['correlation']['corr_matrix']
        df_imp = results_dict['importance']['df_importance']
        
        # Top 20 features por importancia
        top_20_features = df_imp.head(20)['feature'].tolist()
        top_20_features = [f for f in top_20_features if f in corr_matrix.columns]
        
        corr_subset = corr_matrix.loc[top_20_features, top_20_features]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                    square=True, linewidths=0.5, cbar_kws={'label': 'Correlación'}, ax=ax)
        ax.set_title('Matriz de Correlación (Top 20 Features)\nVerde = Positiva | Rojo = Negativa', 
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / '02_correlation_heatmap.png', dpi=300)
        plt.close()
        logger.info(f"   ✅ Guardado: 02_correlation_heatmap.png")
    
    # 3. Sparsity (% de ceros)
    if 'sparsity' in results_dict:
        df_sparse = results_dict['sparsity']['df_sparsity']
        threshold = results_dict['sparsity']['threshold']
        
        # Top 20 más sparse
        df_plot = df_sparse.head(20)
        colors = ['red' if x > threshold else 'steelblue' for x in df_plot['zero_percentage']]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(df_plot)), df_plot['zero_percentage'] * 100, color=colors, edgecolor='black')
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.axvline(threshold * 100, color='red', linestyle='--', linewidth=2, label=f'Umbral {threshold*100:.0f}%')
        ax.set_xlabel('Porcentaje de Ceros (%)', fontweight='bold')
        ax.set_title('Top 20 Features Más Sparse\n(Rojas: > 90% ceros → Candidatas a eliminación)', 
                     fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '03_sparsity_analysis.png', dpi=300)
        plt.close()
        logger.info(f"   ✅ Guardado: 03_sparsity_analysis.png")
    
    logger.info(f"\n✅ Todas las visualizaciones guardadas en: {output_dir}")

# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Análisis de relevancia de features")
    parser.add_argument("--csv", type=str, default="datos_salida/finales/dataset_etiquetado.csv", 
                        help="Ruta al dataset etiquetado")
    parser.add_argument("--model", type=str, default="model_artifacts/lgbm_model.joblib",
                        help="Ruta al modelo entrenado")
    parser.add_argument("--metadata", type=str, default="model_artifacts/model_metadata.json",
                        help="Ruta al metadata del modelo")
    parser.add_argument("--output", type=str, default="reports/feature_analysis",
                        help="Directorio de salida para reportes")
    args = parser.parse_args()
    
    # Cargar datos
    logger.info("📂 Cargando dataset...")
    df = pd.read_csv(args.csv, sep=';', encoding='utf-8-sig')
    logger.info(f"   ✅ Dataset cargado: {len(df)} registros, {df.shape[1]} columnas")
    
    # Cargar modelo y metadata
    logger.info("\n🤖 Cargando modelo y metadata...")
    import joblib
    model = joblib.load(args.model)
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['features_list']
    logger.info(f"   ✅ Modelo cargado: {metadata['model_name']}")
    logger.info(f"   ✅ Features en el modelo: {len(feature_names)}")
    
    # Ejecutar análisis
    results = {}
    results['total_features'] = len(feature_names)
    
    # 1. Feature Importance
    results['importance'] = analizar_feature_importance(model, feature_names, threshold=0.005)
    
    # 2. Varianza
    results['variance'] = analizar_varianza_baja(df, feature_names, threshold=0.01)
    
    # 3. Correlación
    results['correlation'] = analizar_correlacion_alta(df, feature_names, threshold=0.90)
    
    # 4. Sparsity
    results['sparsity'] = analizar_features_sparse(df, feature_names, threshold=0.90)
    
    # 5. Recomendaciones
    recommendations = generar_recomendaciones(results)
    results['recommendations'] = recommendations
    
    # 6. Visualizaciones
    generar_visualizaciones(results, args.output)
    
    # 7. Guardar reporte JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'total_features': results['total_features'],
        'features_to_drop': recommendations['features_to_drop'],
        'num_to_drop': recommendations['num_to_drop'],
        'new_num_features': results['total_features'] - recommendations['num_to_drop'],
        'new_ratio': 896 / (results['total_features'] - recommendations['num_to_drop'])
    }
    
    with open(output_dir / 'feature_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # 8. Guardar lista de features a eliminar
    with open(output_dir / 'features_to_drop.txt', 'w') as f:
        for feat in recommendations['features_to_drop']:
            f.write(f"{feat}\n")
    
    # 9. Guardar resumen en CSV
    recommendations['df_summary'].to_csv(output_dir / 'feature_problems_summary.csv', index=False)
    
    logger.info(f"\n✅ ANÁLISIS COMPLETADO")
    logger.info(f"   📁 Reportes guardados en: {output_dir}")
    logger.info(f"   📄 feature_analysis_report.json")
    logger.info(f"   📄 features_to_drop.txt")
    logger.info(f"   📄 feature_problems_summary.csv")
    logger.info(f"   🖼️ 3 visualizaciones (.png)")

if __name__ == "__main__":
    main()