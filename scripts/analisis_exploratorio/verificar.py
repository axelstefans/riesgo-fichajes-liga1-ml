import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import Config
from core.constants import POS_MAP

INPUT_FILE = Path("datos_salida/brutos/datos_brutos_merged.csv")
OUTPUT_DIR = Path("reports/verificacion_outliers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def cargar_datos():
    print(f"📂 Cargando datos desde: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
    print(f"✅ Datos cargados: {len(df)} registros\n")
    return df

def mapear_posiciones(df):
    df['posicion_agrupada'] = df['posicion'].map(POS_MAP)
    return df

def analizar_outliers_goals(df):
    print("=" * 100)
    print("🎯 ANÁLISIS DE OUTLIERS EN GOALS")
    print("=" * 100)
    
    umbral_delantero = 23
    umbral_mediocampista = 12
    umbral_defensa = 10
    
    outliers_delantero = df[(df['posicion_agrupada'] == 'Delantero') & (df['goals'] > umbral_delantero)].copy()
    outliers_medio = df[(df['posicion_agrupada'] == 'Mediocampista') & (df['goals'] > umbral_mediocampista)].copy()
    outliers_defensa = df[(df['posicion_agrupada'] == 'Defensa') & (df['goals'] > umbral_defensa)].copy()
    
    print(f"\n🔴 DELANTEROS CON MÁS DE {umbral_delantero} GOLES:")
    print("-" * 100)
    if len(outliers_delantero) > 0:
        outliers_delantero['goles_por_90'] = (outliers_delantero['goals'] / (outliers_delantero['minutesPlayed'] / 90)).round(2)
        outliers_delantero['partidos_estimados'] = (outliers_delantero['minutesPlayed'] / 90).round(1)
        
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'posicion', 'goals', 'totalShots', 'shotsOnTarget', 
                'minutesPlayed', 'partidos_estimados', 'goles_por_90', 'season', 'club_origen']
        print(outliers_delantero[cols].to_string(index=False))
        
        print(f"\n💡 INTERPRETACIÓN:")
        for idx, row in outliers_delantero.iterrows():
            goles_partido = row['goles_por_90']
            if goles_partido > 1.0:
                print(f"   ⚠️ {row['nombre_jugador']}: {row['goals']} goles en {row['partidos_estimados']:.1f} partidos (~{goles_partido:.2f} goles/90min)")
                print(f"      → VERIFICAR: Tasa extremadamente alta. Posible duplicado o error.")
            else:
                print(f"   ✅ {row['nombre_jugador']}: {row['goals']} goles en {row['partidos_estimados']:.1f} partidos (~{goles_partido:.2f} goles/90min)")
                print(f"      → NORMAL: Goleador excepcional pero realista.")
    else:
        print("   ✅ No se encontraron outliers extremos en delanteros.")
    
    print(f"\n🟡 MEDIOCAMPISTAS CON MÁS DE {umbral_mediocampista} GOLES:")
    print("-" * 100)
    if len(outliers_medio) > 0:
        outliers_medio['goles_por_90'] = (outliers_medio['goals'] / (outliers_medio['minutesPlayed'] / 90)).round(2)
        outliers_medio['partidos_estimados'] = (outliers_medio['minutesPlayed'] / 90).round(1)
        
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'posicion', 'goals', 'totalShots', 'minutesPlayed', 
                'partidos_estimados', 'goles_por_90', 'season', 'club_origen']
        print(outliers_medio[cols].to_string(index=False))
    else:
        print("   ✅ No se encontraron outliers extremos en mediocampistas.")
    
    print(f"\n🔵 DEFENSAS CON MÁS DE {umbral_defensa} GOLES:")
    print("-" * 100)
    if len(outliers_defensa) > 0:
        outliers_defensa['goles_por_90'] = (outliers_defensa['goals'] / (outliers_defensa['minutesPlayed'] / 90)).round(2)
        outliers_defensa['partidos_estimados'] = (outliers_defensa['minutesPlayed'] / 90).round(1)
        
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'posicion', 'goals', 'totalShots', 'minutesPlayed', 
                'partidos_estimados', 'goles_por_90', 'season', 'club_origen']
        print(outliers_defensa[cols].to_string(index=False))
    else:
        print("   ✅ No se encontraron outliers extremos en defensas.")
    
    outliers_goals = pd.concat([outliers_delantero, outliers_medio, outliers_defensa], ignore_index=True)
    if len(outliers_goals) > 0:
        outliers_goals.to_csv(OUTPUT_DIR / "outliers_goals.csv", index=False, sep=';', encoding='utf-8-sig')
        print(f"\n💾 Outliers de goals guardados en: {OUTPUT_DIR / 'outliers_goals.csv'}")

def analizar_outliers_totalshots(df):
    print("\n" + "=" * 100)
    print("🎯 ANÁLISIS DE OUTLIERS EN TOTALSHOTS")
    print("=" * 100)
    
    umbral_delantero = 110
    umbral_mediocampista = 90
    umbral_defensa = 60
    
    outliers_delantero = df[(df['posicion_agrupada'] == 'Delantero') & (df['totalShots'] > umbral_delantero)].copy()
    outliers_medio = df[(df['posicion_agrupada'] == 'Mediocampista') & (df['totalShots'] > umbral_mediocampista)].copy()
    outliers_defensa = df[(df['posicion_agrupada'] == 'Defensa') & (df['totalShots'] > umbral_defensa)].copy()
    
    print(f"\n🔴 DELANTEROS CON MÁS DE {umbral_delantero} TIROS:")
    print("-" * 100)
    if len(outliers_delantero) > 0:
        outliers_delantero['tiros_por_90'] = (outliers_delantero['totalShots'] / (outliers_delantero['minutesPlayed'] / 90)).round(2)
        outliers_delantero['partidos_estimados'] = (outliers_delantero['minutesPlayed'] / 90).round(1)
        
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'posicion', 'totalShots', 'shotsOnTarget', 'goals',
                'minutesPlayed', 'partidos_estimados', 'tiros_por_90', 'season', 'club_origen']
        print(outliers_delantero[cols].to_string(index=False))
        
        print(f"\n💡 INTERPRETACIÓN:")
        for idx, row in outliers_delantero.iterrows():
            tiros_partido = row['tiros_por_90']
            if tiros_partido > 5.0:
                print(f"   ⚠️ {row['nombre_jugador']}: {row['totalShots']} tiros (~{tiros_partido:.2f} tiros/90min)")
                print(f"      → ALTO pero posible en delanteros rematadores compulsivos.")
            else:
                print(f"   ✅ {row['nombre_jugador']}: {row['totalShots']} tiros (~{tiros_partido:.2f} tiros/90min)")
                print(f"      → NORMAL para delantero activo.")
    else:
        print("   ✅ No se encontraron outliers extremos.")
    
    print(f"\n🟡 MEDIOCAMPISTAS CON MÁS DE {umbral_mediocampista} TIROS:")
    print("-" * 100)
    if len(outliers_medio) > 0:
        outliers_medio['tiros_por_90'] = (outliers_medio['totalShots'] / (outliers_medio['minutesPlayed'] / 90)).round(2)
        outliers_medio['partidos_estimados'] = (outliers_medio['minutesPlayed'] / 90).round(1)
        
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'posicion', 'totalShots', 'shotsOnTarget', 'goals',
                'minutesPlayed', 'partidos_estimados', 'tiros_por_90', 'season', 'club_origen']
        print(outliers_medio[cols].to_string(index=False))
    else:
        print("   ✅ No se encontraron outliers extremos.")
    
    outliers_shots = pd.concat([outliers_delantero, outliers_medio, outliers_defensa], ignore_index=True)
    if len(outliers_shots) > 0:
        outliers_shots.to_csv(OUTPUT_DIR / "outliers_totalshots.csv", index=False, sep=';', encoding='utf-8-sig')
        print(f"\n💾 Outliers de totalShots guardados en: {OUTPUT_DIR / 'outliers_totalshots.csv'}")

def analizar_duplicados_jugador_temporada(df):
    print("\n" + "=" * 100)
    print("🎯 ANÁLISIS DE DUPLICADOS (Jugador × Temporada)")
    print("=" * 100)
    
    duplicados = df.groupby(['tm_id', 'season']).size().reset_index(name='count')
    duplicados = duplicados[duplicados['count'] > 1].sort_values('count', ascending=False)
    
    if len(duplicados) > 0:
        print(f"\n⚠️ Se encontraron {len(duplicados)} jugadores con múltiples registros en la MISMA temporada:")
        print("-" * 100)
        
        for idx, row in duplicados.head(10).iterrows():
            tm_id = row['tm_id']
            season = row['season']
            count = row['count']
            
            registros = df[(df['tm_id'] == tm_id) & (df['season'] == season)]
            
            print(f"\n🔍 Jugador: {registros.iloc[0]['nombre_jugador']} (tm_id: {tm_id}) - Temporada: {season}")
            print(f"   Cantidad de registros: {count}")
            
            cols = ['ss_id', 'club_origen', 'goals', 'assists', 'minutesPlayed', 'appearances', 'totalShots']
            print(registros[cols].to_string(index=False))
            
            if registros['club_origen'].nunique() > 1:
                print(f"   💡 CAUSA: Jugador cambió de club en la misma temporada → NORMAL (fichaje mid-season)")
            else:
                print(f"   ⚠️ CAUSA: Mismo club, múltiples registros → POSIBLE ERROR DE SCRAPING")
                print(f"   🔧 ACCIÓN: Verificar en SofaScore si son stats de diferentes competencias (Liga vs Copa)")
        
        duplicados_full = df.merge(duplicados, on=['tm_id', 'season'], how='inner')
        duplicados_full.to_csv(OUTPUT_DIR / "duplicados_jugador_temporada.csv", index=False, sep=';', encoding='utf-8-sig')
        print(f"\n💾 Duplicados guardados en: {OUTPUT_DIR / 'duplicados_jugador_temporada.csv'}")
    else:
        print("\n✅ No se encontraron duplicados (jugador × temporada).")
        print("   Todos los jugadores tienen 1 registro por temporada.")

def analizar_registros_sospechosos(df):
    print("\n" + "=" * 100)
    print("🎯 ANÁLISIS DE REGISTROS SOSPECHOSOS")
    print("=" * 100)
    
    print("\n🔍 CASO 1: Jugadores con minutesPlayed > 90 pero TODAS las stats en 0")
    print("-" * 100)
    sospechosos_1 = df[
        (df['minutesPlayed'] > 90) &
        (df['goals'] == 0) &
        (df['assists'] == 0) &
        (df['totalShots'] == 0) &
        (df['totalPasses'] == 0) &
        (df['tackles'] == 0)
    ].copy()
    
    if len(sospechosos_1) > 0:
        print(f"⚠️ Se encontraron {len(sospechosos_1)} registros sospechosos:")
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'posicion', 'minutesPlayed', 'appearances', 'season', 'club_origen']
        print(sospechosos_1[cols].head(10).to_string(index=False))
        print(f"\n🔧 ACCIÓN: Estos registros serán eliminados en el preprocesamiento (filtro de registros problemáticos).")
        
        sospechosos_1.to_csv(OUTPUT_DIR / "registros_sospechosos_stats_cero.csv", index=False, sep=';', encoding='utf-8-sig')
        print(f"💾 Guardado en: {OUTPUT_DIR / 'registros_sospechosos_stats_cero.csv'}")
    else:
        print("✅ No se encontraron registros con stats todas en 0.")
    
    print("\n🔍 CASO 2: Jugadores con minutesPlayed == 0 pero con stats > 0")
    print("-" * 100)
    sospechosos_2 = df[
        (df['minutesPlayed'] == 0) &
        ((df['goals'] > 0) | (df['assists'] > 0) | (df['totalShots'] > 0))
    ].copy()
    
    if len(sospechosos_2) > 0:
        print(f"⚠️ Se encontraron {len(sospechosos_2)} registros imposibles (stats sin minutos):")
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'goals', 'assists', 'totalShots', 'minutesPlayed', 'season']
        print(sospechosos_2[cols].head(10).to_string(index=False))
        print(f"\n🔧 ACCIÓN: ERROR DE SCRAPING. Eliminar estos registros.")
        
        sospechosos_2.to_csv(OUTPUT_DIR / "registros_sospechosos_stats_sin_minutos.csv", index=False, sep=';', encoding='utf-8-sig')
        print(f"💾 Guardado en: {OUTPUT_DIR / 'registros_sospechosos_stats_sin_minutos.csv'}")
    else:
        print("✅ No se encontraron registros con stats sin minutos.")
    
    print("\n🔍 CASO 3: Jugadores con conversion absurda (goals > totalShots)")
    print("-" * 100)
    sospechosos_3 = df[df['goals'] > df['totalShots']].copy()
    
    if len(sospechosos_3) > 0:
        print(f"⚠️ Se encontraron {len(sospechosos_3)} registros IMPOSIBLES (más goles que tiros):")
        cols = ['tm_id', 'ss_id', 'nombre_jugador', 'goals', 'totalShots', 'shotsOnTarget', 'season', 'club_origen']
        print(sospechosos_3[cols].to_string(index=False))
        print(f"\n🔧 ACCIÓN: ERROR DE SCRAPING CRÍTICO. Revisar código de extracción.")
        
        sospechosos_3.to_csv(OUTPUT_DIR / "registros_sospechosos_goals_mayor_shots.csv", index=False, sep=';', encoding='utf-8-sig')
        print(f"💾 Guardado en: {OUTPUT_DIR / 'registros_sospechosos_goals_mayor_shots.csv'}")
    else:
        print("✅ No se encontraron registros con goals > totalShots.")

def generar_links_sofascore(df_outliers):
    print("\n" + "=" * 100)
    print("🔗 GENERACIÓN DE LINKS PARA VERIFICACIÓN EN SOFASCORE")
    print("=" * 100)
    
    if len(df_outliers) == 0:
        print("✅ No hay outliers para verificar.")
        return
    
    links_file = OUTPUT_DIR / "links_verificacion_sofascore.txt"
    
    with open(links_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("LINKS PARA VERIFICACIÓN MANUAL EN SOFASCORE\n")
        f.write("=" * 100 + "\n\n")
        f.write("INSTRUCCIONES:\n")
        f.write("1. Abre cada link en tu navegador\n")
        f.write("2. Verifica las estadísticas mostradas en SofaScore\n")
        f.write("3. Compara con los valores en el CSV\n")
        f.write("4. Si hay discrepancia → Error de scraping (revisar código)\n")
        f.write("5. Si coinciden → Outlier real (jugador excepcional)\n\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, row in df_outliers.iterrows():
            ss_id = row['ss_id']
            ss_season_id = row.get('ss_season_id', 'DESCONOCIDO')
            nombre = row['nombre_jugador']
            goals = row['goals']
            season = row['season']
            
            url = f"https://www.sofascore.com/player/{ss_id}/{ss_season_id}"
            
            f.write(f"Jugador: {nombre}\n")
            f.write(f"Goles en dataset: {goals}\n")
            f.write(f"Temporada: {season}\n")
            f.write(f"Link: {url}\n")
            f.write(f"tm_id: {row['tm_id']} | ss_id: {ss_id}\n")
            f.write("-" * 100 + "\n\n")
    
    print(f"✅ Links de verificación guardados en: {links_file}")
    print(f"\n💡 PRÓXIMO PASO:")
    print(f"   1. Abre el archivo: {links_file}")
    print(f"   2. Verifica cada link manualmente en SofaScore")
    print(f"   3. Si encuentras errores, reporta los ss_id problemáticos")

def main():
    print("\n" + "=" * 100)
    print("🔍 VERIFICACIÓN DE OUTLIERS Y CALIDAD DE DATOS")
    print("=" * 100 + "\n")
    
    df = cargar_datos()
    df = mapear_posiciones(df)
    
    analizar_outliers_goals(df)
    analizar_outliers_totalshots(df)
    analizar_duplicados_jugador_temporada(df)
    analizar_registros_sospechosos(df)
    
    outliers_goals = df[
        ((df['posicion_agrupada'] == 'Delantero') & (df['goals'] > 23)) |
        ((df['posicion_agrupada'] == 'Mediocampista') & (df['goals'] > 12)) |
        ((df['posicion_agrupada'] == 'Defensa') & (df['goals'] > 10))
    ].copy()
    
    generar_links_sofascore(outliers_goals)
    
    print("\n" + "=" * 100)
    print("✅ VERIFICACIÓN COMPLETADA")
    print("=" * 100)
    print(f"\n📁 Todos los reportes guardados en: {OUTPUT_DIR}")
    print("\n📋 Archivos generados:")
    print(f"   - outliers_goals.csv")
    print(f"   - outliers_totalshots.csv")
    print(f"   - duplicados_jugador_temporada.csv (si aplica)")
    print(f"   - registros_sospechosos_*.csv (si aplica)")
    print(f"   - links_verificacion_sofascore.txt")

if __name__ == "__main__":
    main()