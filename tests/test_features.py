"""
tests/test_features.py
Unit test suite validating core feature engineering mathematics and contextual encodings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

# Guarantee project root and scripts directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core.features import crear_features_contextuales, crear_features_numericas


def test_crear_features_numericas_math_and_zero_division():
    """
    Validates:
    1. Exact per-90 math on 17 metrics: stat_p90 = stat / (minutesPlayed / 90).
    2. Conversion percentages: accuratePassesPercentage, goalConversionPercentage, penaltyConversionPercentage.
    3. Safe coercion and zero-division handling (minutesPlayed=0 or totalShots=0 must yield 0.0, no NaN/inf).
    """
    mock_input = pd.DataFrame(
        [
            {
                # Player 1: 1800 minutes = 20 full 90s
                "minutesPlayed": 1800,
                "goals": 10,
                "assists": 4,
                "shotsOnTarget": 15,
                "shotsOffTarget": 5,
                "blockedShots": 2,
                "keyPasses": 20,
                "successfulDribbles": 8,
                "offsides": 4,
                "wasFouled": 12,
                "clearances": 6,
                "dribbledPast": 2,
                "fouls": 14,
                "aerialDuelsWon": 10,
                "accurateFinalThirdPasses": 40,
                "accurateLongBalls": 18,
                "accurateCrosses": 6,
                "yellowCards": 2,
                # Percentages basis
                "totalPasses": 500,
                "accuratePasses": 400,
                "totalShots": 40,
                "penaltiesTaken": 5,
                "penaltyGoals": 4,
            },
            {
                # Player 2: Edge-case zero minutes and zero volumes
                "minutesPlayed": 0,
                "goals": 0,
                "assists": 0,
                "shotsOnTarget": 0,
                "shotsOffTarget": 0,
                "blockedShots": 0,
                "keyPasses": 0,
                "successfulDribbles": 0,
                "offsides": 0,
                "wasFouled": 0,
                "clearances": 0,
                "dribbledPast": 0,
                "fouls": 0,
                "aerialDuelsWon": 0,
                "accurateFinalThirdPasses": 0,
                "accurateLongBalls": 0,
                "accurateCrosses": 0,
                "yellowCards": 0,
                "totalPasses": 0,
                "accuratePasses": 0,
                "totalShots": 0,
                "penaltiesTaken": 0,
                "penaltyGoals": 0,
            },
        ]
    )

    df_result = crear_features_numericas(mock_input)

    # 1. Assert conversion percentages
    expected_pct_series = pd.DataFrame(
        {
            "accuratePassesPercentage": [80.0, 0.0],
            "goalConversionPercentage": [25.0, 0.0],
            "penaltyConversionPercentage": [80.0, 0.0],
        }
    )
    assert_frame_equal(
        df_result[
            [
                "accuratePassesPercentage",
                "goalConversionPercentage",
                "penaltyConversionPercentage",
            ]
        ],
        expected_pct_series,
        check_dtype=False,
    )

    # 2. Assert selected per-90 metrics (Player 1: / 20.0; Player 2: fallback 0.0)
    expected_p90 = pd.DataFrame(
        {
            "goals_p90": [0.5, 0.0],
            "assists_p90": [0.2, 0.0],
            "keyPasses_p90": [1.0, 0.0],
            "fouls_p90": [0.7, 0.0],
            "accurateCrosses_p90": [0.3, 0.0],
        }
    )
    assert_frame_equal(
        df_result[
            [
                "goals_p90",
                "assists_p90",
                "keyPasses_p90",
                "fouls_p90",
                "accurateCrosses_p90",
            ]
        ],
        expected_p90,
        check_dtype=False,
    )

    # 3. Assert no infinite or NaN values exist in any numeric column
    num_cols = df_result.select_dtypes(include=np.number).columns
    assert not np.isinf(df_result[num_cols].values).any(), "Found inf values in numerical features"
    assert not df_result[num_cols].isna().any().any(), "Found NaN values in numerical features"


def test_crear_features_contextuales_dummies_and_tiers():
    """
    Validates:
    1. Position grouping via POS_MAP and reference drop (pos_Defensa dropped; pos_Delantero & pos_Mediocampista kept).
    2. Nationality grouping and reference drop (nac_Otras dropped).
    3. contexto_equipo_top correctly identifies Top 4 Liga 1 destination clubs.
    """
    mock_input = pd.DataFrame(
        [
            {
                # Row 0: Forward from Peru going to Top 4 club
                "posicion": "Extremo izquierdo",
                "nacionalidad_str": "Perú",
                "club_destino": "Club Alianza Lima",
                "club_origen": "FBC Melgar",
            },
            {
                # Row 1: Midfielder from Argentina going to non-top club
                "posicion": "Pivote",
                "nacionalidad_str": "Argentina",
                "club_destino": "Sport Huancayo",
                "club_origen": "Rosario Central",
            },
            {
                # Row 2: Defender from non-top nationality (reference baseline)
                "posicion": "Defensa central",
                "nacionalidad_str": "Brasil",
                "club_destino": "Cusco FC",
                "club_origen": "Deportivo Pereira",
            },
        ]
    )

    df_context = crear_features_contextuales(mock_input)

    # Validate positional dummies: pos_Defensa must NOT exist; Delantero and Mediocampista must be binary
    assert "pos_Defensa" not in df_context.columns
    expected_pos = pd.DataFrame(
        {
            "pos_Delantero": [1, 0, 0],
            "pos_Mediocampista": [0, 1, 0],
        }
    )
    assert_frame_equal(df_context[["pos_Delantero", "pos_Mediocampista"]], expected_pos, check_dtype=False)

    # Validate nationality dummies: nac_Otras must NOT exist; Brasil falls into reference (all 0s)
    assert "nac_Otras" not in df_context.columns
    expected_nac = pd.DataFrame(
        {
            "nac_Perú": [1, 0, 0],
            "nac_Argentina": [0, 1, 0],
            "nac_Colombia": [0, 0, 0],
            "nac_Uruguay": [0, 0, 0],
        }
    )
    assert_frame_equal(
        df_context[["nac_Perú", "nac_Argentina", "nac_Colombia", "nac_Uruguay"]],
        expected_nac,
        check_dtype=False,
    )

    # Validate destination tier context: Alianza Lima is Top 4 (1), Huancayo & Cusco are not (0)
    expected_top = pd.Series([1, 0, 0], name="contexto_equipo_top")
    assert_series_equal(df_context["contexto_equipo_top"], expected_top, check_dtype=False)


def test_es_club_grande_and_origen_extranjero():
    """
    Validates:
    1. es_club_grande identifies clubs with keywords (e.g. Boca, River, Flamengo, Peñarol).
    2. proviene_liga_extranjera flags clubs not in Liga 1 catalog.
    3. Handles null, NaN, and unexpected formats safely.
    """
    mock_input = pd.DataFrame(
        [
            # Case 1: Big foreign club
            {
                "posicion": "Delantero",
                "nacionalidad_str": "Argentina",
                "club_destino": "Universitario de Deportes",
                "club_origen": "C.A. Boca Juniors",
            },
            # Case 2: Mid-tier domestic Liga 1 club
            {
                "posicion": "Mediocampista",
                "nacionalidad_str": "Perú",
                "club_destino": "Club Sporting Cristal",
                "club_origen": "Universidad Técnica de Cajamarca",
            },
            # Case 3: Foreign club but not a historical powerhouse
            {
                "posicion": "Defensa",
                "nacionalidad_str": "Uruguay",
                "club_destino": "FBC Melgar",
                "club_origen": "Danubio FC",
            },
            # Case 4: Missing or None origin
            {
                "posicion": "Delantero",
                "nacionalidad_str": "Colombia",
                "club_destino": "Cienciano",
                "club_origen": None,
            },
        ]
    )

    df_context = crear_features_contextuales(mock_input)

    # 1. proviene_club_grande (Boca Juniors=1, UTC=0, Danubio=0, None=0)
    expected_club_grande = pd.Series([1, 0, 0, 0], name="proviene_club_grande")
    assert_series_equal(df_context["proviene_club_grande"], expected_club_grande, check_dtype=False)

    # 2. proviene_liga_extranjera (Boca=1, UTC (in Liga 1)=0, Danubio=1, None=1)
    expected_extranjera = pd.Series([1, 0, 1, 1], name="proviene_liga_extranjera")
    assert_series_equal(df_context["proviene_liga_extranjera"], expected_extranjera, check_dtype=False)
