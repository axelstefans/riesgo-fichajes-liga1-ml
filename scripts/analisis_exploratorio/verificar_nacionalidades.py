import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cargar datasets
df_crudo = pd.read_csv("datos_salida/brutos/datos_brutos_merged.csv", sep=";", encoding="utf-8-sig")
df_final = pd.read_csv("datos_salida/finales/dataset_etiquetado.csv", sep=";", encoding="utf-8-sig")

print("=" * 80)
print("📊 ANÁLISIS DE NACIONALIDADES")
print("=" * 80)

# ===============================================
# 1. DATASET CRUDO (Antes del preprocesamiento)
# ===============================================
print("\n1️⃣ DATASET CRUDO (ORIGINAL):")
print(f"   Total registros: {len(df_crudo)}")

if 'nacionalidad_str' in df_crudo.columns:
    nac_counts_crudo = df_crudo['nacionalidad_str'].value_counts()
    nac_pct_crudo = df_crudo['nacionalidad_str'].value_counts(normalize=True) * 100
    
    print(f"\n   Top 10 Nacionalidades:")
    for nac, count in nac_counts_crudo.head(10).items():
        pct = nac_pct_crudo[nac]
        print(f"   {nac:20s} {count:4d} ({pct:5.2f}%)")
    
    print(f"\n   Total nacionalidades únicas: {len(nac_counts_crudo)}")
else:
    print("   ⚠️ Columna 'nacionalidad_str' no encontrada")

# ===============================================
# 2. DATASET FINAL (Después del preprocesamiento)
# ===============================================
print("\n" + "=" * 80)
print("2️⃣ DATASET FINAL (PREPROCESADO):")
print(f"   Total registros: {len(df_final)}")

# Verificar one-hot encoding
nac_cols = [col for col in df_final.columns if col.startswith('nac_')]
print(f"\n   Columnas de nacionalidad encontradas: {len(nac_cols)}")

if nac_cols:
    print(f"\n   Distribución de nacionalidades (one-hot encoding):")
    for col in sorted(nac_cols):
        count = df_final[col].sum()
        pct = (count / len(df_final)) * 100
        zero_pct = ((df_final[col] == 0).sum() / len(df_final)) * 100
        print(f"   {col:30s} {int(count):4d} jugadores ({pct:5.2f}%) | {zero_pct:5.1f}% ceros")
    
    # Baseline (jugadores sin ninguna de estas nacionalidades)
    baseline_mask = df_final[nac_cols].sum(axis=1) == 0
    baseline_count = baseline_mask.sum()
    baseline_pct = (baseline_count / len(df_final)) * 100
    print(f"\n   Baseline (Otras nacionalidades): {baseline_count:4d} jugadores ({baseline_pct:5.2f}%)")
else:
    print("   ⚠️ No se encontraron columnas de nacionalidad con prefijo 'nac_'")

# ===============================================
# 3. COMPARACIÓN CRUDO vs FINAL
# ===============================================
print("\n" + "=" * 80)
print("3️⃣ COMPARACIÓN CRUDO vs FINAL:")
print("=" * 80)

if 'nacionalidad_str' in df_crudo.columns and nac_cols:
    print("\n   Nacionalidades seleccionadas para one-hot encoding:")
    nacionalidades_importantes = ['Perú', 'Argentina', 'Colombia', 'Uruguay']
    
    for nac in nacionalidades_importantes:
        # Contar en dataset crudo
        count_crudo = (df_crudo['nacionalidad_str'] == nac).sum()
        pct_crudo = (count_crudo / len(df_crudo)) * 100
        
        # Contar en dataset final
        col_name = f'nac_{nac}'
        if col_name in df_final.columns:
            count_final = df_final[col_name].sum()
            pct_final = (count_final / len(df_final)) * 100
            
            diff = count_final - count_crudo
            diff_pct = ((count_final - count_crudo) / count_crudo * 100) if count_crudo > 0 else 0
            
            print(f"\n   {nac}:")
            print(f"      Crudo:  {count_crudo:4d} ({pct_crudo:5.2f}%)")
            print(f"      Final:  {int(count_final):4d} ({pct_final:5.2f}%)")
            print(f"      Cambio: {diff:+4.0f} ({diff_pct:+5.1f}%)")

# ===============================================
# 4. VISUALIZACIÓN
# ===============================================
print("\n" + "=" * 80)
print("4️⃣ GENERANDO VISUALIZACIONES...")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Dataset Crudo (Top 10)
if 'nacionalidad_str' in df_crudo.columns:
    top_10_crudo = df_crudo['nacionalidad_str'].value_counts().head(10)
    axes[0].barh(range(len(top_10_crudo)), top_10_crudo.values, color='steelblue', edgecolor='black')
    axes[0].set_yticks(range(len(top_10_crudo)))
    axes[0].set_yticklabels(top_10_crudo.index)
    axes[0].set_xlabel('Cantidad de Jugadores', fontweight='bold')
    axes[0].set_title('Top 10 Nacionalidades (Dataset Crudo Original)', fontweight='bold', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Agregar porcentajes
    for i, (nac, count) in enumerate(top_10_crudo.items()):
        pct = (count / len(df_crudo)) * 100
        axes[0].text(count + 5, i, f'{pct:.1f}%', va='center', fontweight='bold')

# Gráfico 2: Dataset Final (One-hot encoding)
if nac_cols:
    nac_counts_final = {col.replace('nac_', ''): df_final[col].sum() for col in nac_cols}
    
    # Agregar baseline
    baseline_mask = df_final[nac_cols].sum(axis=1) == 0
    nac_counts_final['Otras (Baseline)'] = baseline_mask.sum()
    
    # Ordenar
    nac_counts_sorted = dict(sorted(nac_counts_final.items(), key=lambda x: x[1], reverse=True))
    
    axes[1].barh(range(len(nac_counts_sorted)), list(nac_counts_sorted.values()), 
                 color='coral', edgecolor='black')
    axes[1].set_yticks(range(len(nac_counts_sorted)))
    axes[1].set_yticklabels(list(nac_counts_sorted.keys()))
    axes[1].set_xlabel('Cantidad de Jugadores', fontweight='bold')
    axes[1].set_title('Distribución de Nacionalidades (Dataset Final One-Hot)', fontweight='bold', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Agregar porcentajes
    for i, (nac, count) in enumerate(nac_counts_sorted.items()):
        pct = (count / len(df_final)) * 100
        axes[1].text(count + 5, i, f'{pct:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
output_path = Path("reports/feature_analysis/nacionalidades_comparacion.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Visualización guardada: {output_path}")

# ===============================================
# 5. CONCLUSIONES
# ===============================================
print("\n" + "=" * 80)
print("5️⃣ CONCLUSIONES:")
print("=" * 80)

if nac_cols:
    # Calcular % de ceros por columna
    zero_percentages = {}
    for col in nac_cols:
        zero_pct = (df_final[col] == 0).sum() / len(df_final) * 100
        zero_percentages[col] = zero_pct
    
    print("\n   ¿Son 'sparse' las columnas de nacionalidad?")
    print("   (Una columna es 'sparse' si tiene muchos ceros, PERO esto es NORMAL para one-hot encoding)")
    
    for col in sorted(nac_cols):
        zero_pct = zero_percentages[col]
        jugadores_count = df_final[col].sum()
        
        if zero_pct > 90:
            status = "⚠️ SPARSE"
        elif zero_pct > 80:
            status = "⚠️ Moderado"
        else:
            status = "✅ Normal"
        
        print(f"\n   {col}:")
        print(f"      {zero_pct:.1f}% ceros → {status}")
        print(f"      Representa {int(jugadores_count)} jugadores ({100-zero_pct:.1f}%)")
        print(f"      ¿Eliminar? {'NO - Representa una nacionalidad válida' if jugadores_count > 20 else 'CONSIDERAR - Muy pocos casos'}")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)