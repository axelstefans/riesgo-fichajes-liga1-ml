import os
import pandas as pd
import logging
from config import Config
from pipeline_recoleccion import obtener_fichajes_tm, mapear_ids_ss, encontrar_temporada_previa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generar_paths(file_suffix, tm_saison_id):
    sufijo_final = f"{tm_saison_id}_{int(tm_saison_id) + 1}"
    
    return {
        'fichajes_tm': f"{Config.DIR_ENTRADA}/fichajes_tm_{file_suffix}.csv",
        'fichajes_mapeados': f"{Config.DIR_ENTRADA}/fichajes_mapeados_{file_suffix}.csv",
        'fichajes_final': f"{Config.DIR_ENTRADA}/fichajes_{sufijo_final}.csv", # <-- CAMBIO APLICADO AQUÍ
        'fichajes_fallidos': f"{Config.DIR_DEBUG}/log_recoleccion_fallidos_{file_suffix}.csv"
    }

def cargar_o_crear_csv(path, funcion_crear, *args):
    if os.path.exists(path):
        logger.info(f"Archivo '{os.path.basename(path)}' ya existe. Cargando desde disco.")
        df = pd.read_csv(path, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING, dtype=str)
        if not df.empty:
            return df
        logger.warning(f"Archivo '{os.path.basename(path)}' existe pero está vacío. Se intentará recrearlo.")
    
    df = funcion_crear(*args)
    if df is not None and not df.empty:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
        logger.info(f"Guardados {len(df)} registros en '{os.path.basename(path)}'.")
        return df
    
    return None

if __name__ == "__main__":
    
    if not Config.TEMPORADAS:
        logger.error("El diccionario 'TEMPORADAS' en config.py está vacío. No hay nada que procesar.")
        exit(1)

    resumen_temporadas = []
    
    for temporada_nombre, config_temporada in Config.TEMPORADAS.items():
        
        logger.info(f"--- INICIANDO FASE 1: RECOLECCIÓN DE IDS PARA {temporada_nombre} ---")
        
        paths = generar_paths(config_temporada['file_suffix'], config_temporada['tm_saison_id'])
        resultado_temporada = {
            "temporada": temporada_nombre, 
            "fase_completada": "No iniciado", 
            "registros_finales": 0,
            "registros_fallidos": 0
        }
        
        try:
            logger.info("Paso 1.1: Obteniendo fichajes desde Transfermarkt...")
            df_fichajes = cargar_o_crear_csv(paths['fichajes_tm'], obtener_fichajes_tm, config_temporada['tm_saison_id'], Config.DRIVER_VERSION)
            if df_fichajes is None or df_fichajes.empty:
                logger.warning("No se pudieron obtener fichajes. Saltando temporada.")
                resultado_temporada["fase_completada"] = "Paso 1.1 (Fallido)"
                resumen_temporadas.append(resultado_temporada)
                continue
            
            logger.info("Paso 1.2: Mapeando IDs de Sofascore...")
            df_mapeado = cargar_o_crear_csv(paths['fichajes_mapeados'], mapear_ids_ss, df_fichajes, Config.DRIVER_VERSION)
            if df_mapeado is None or df_mapeado.empty:
                logger.warning("No se pudieron mapear IDs. Saltando temporada.")
                resultado_temporada["fase_completada"] = "Paso 1.2 (Fallido)"
                resumen_temporadas.append(resultado_temporada)
                continue
            
            logger.info("Paso 1.3: Encontrando IDs de temporada previa...")
            if os.path.exists(paths['fichajes_final']):
                logger.info(f"Archivo final '{os.path.basename(paths['fichajes_final'])}' ya existe. Omitiendo este paso.")
                df_exitosos = cargar_o_crear_csv(paths['fichajes_final'], lambda: None)
                if df_exitosos is not None and not df_exitosos.empty:
                    resultado_temporada["fase_completada"] = "Completa (Cargada)"
                    resultado_temporada["registros_finales"] = len(df_exitosos)
                else:
                    os.remove(paths['fichajes_final'])
                    logger.warning("Archivo final cargado estaba vacío. Se forzará la recreación.")
                    df_exitosos, df_fallidos = encontrar_temporada_previa(df_mapeado, config_temporada['anio_fichaje'], Config.DRIVER_VERSION)
            else:
                df_exitosos, df_fallidos = encontrar_temporada_previa(df_mapeado, config_temporada['anio_fichaje'], Config.DRIVER_VERSION)
            
            if df_exitosos is not None and not df_exitosos.empty:
                df_exitosos.to_csv(paths['fichajes_final'], index=False, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
                logger.info(f"Guardados {len(df_exitosos)} registros listos para la siguiente fase.")
                resultado_temporada["registros_finales"] = len(df_exitosos)
                resultado_temporada["fase_completada"] = "Completa"
            else:
                resultado_temporada["fase_completada"] = "Completa (Sin exitosos)"
            
            if df_fallidos is not None and not df_fallidos.empty:
                os.makedirs(os.path.dirname(paths['fichajes_fallidos']), exist_ok=True)
                df_fallidos.to_csv(paths['fichajes_fallidos'], index=False, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
                logger.info(f"Guardados {len(df_fallidos)} registros fallidos para revisión.")
                resultado_temporada["registros_fallidos"] = len(df_fallidos)

            logger.info(f"--- PROCESO PARA TEMPORADA {temporada_nombre} COMPLETADO ---\n")
        
        except Exception as e:
            logger.error(f"Error procesando temporada {temporada_nombre}: {e}", exc_info=True)
            resultado_temporada["fase_completada"] = "Error inesperado"
        
        resumen_temporadas.append(resultado_temporada)
    
    logger.info("=" * 80)
    logger.info(f"{'RESUMEN FINAL DE LA FASE 1: RECOLECCIÓN DE IDS':^80}")
    logger.info("=" * 80)
    logger.info(f"{'TEMPORADA':<15} {'ESTADO':<30} {'EXITOSOS':<15} {'FALLIDOS':<15}")
    logger.info("-" * 80)
    for r in resumen_temporadas:
        logger.info(f"{r['temporada']:<15} {r['fase_completada']:<30} {r['registros_finales']:<15} {r['registros_fallidos']:<15}")
    logger.info("=" * 80)
    logger.info("PROCESO COMPLETO FINALIZADO")