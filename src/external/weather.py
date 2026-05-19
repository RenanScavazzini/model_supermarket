"""
Descrição:
    Módulo responsável pela coleta, processamento e carregamento
    de dados climáticos históricos utilizando a API Open-Meteo.

Autor:
    Renan Douglas Floriano Scavazzini

Versão:
    1.0 - 18/05/2026 - Estrutura inicial de dados climáticos.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import requests
import pandas as pd

from pathlib import Path


def download_weather_data(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    output_path: str
) -> pd.DataFrame:
    """
    Descrição:
        Realiza download de dados climáticos históricos
        utilizando a API Open-Meteo.

    Parâmetros:
        start_date (str): Data inicial YYYY-MM-DD.
        end_date (str): Data final YYYY-MM-DD.
        latitude (float): Latitude da localização.
        longitude (float): Longitude da localização.
        output_path (str): Caminho parquet de saída.

    Retorno:
        pd.DataFrame
    """

    print("🌦 Baixando dados climáticos...")

    url = (

        "https://archive-api.open-meteo.com/v1/archive"

        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"

        "&daily="
        "temperature_2m_max,"
        "temperature_2m_min,"
        "temperature_2m_mean,"
        "precipitation_sum"

        "&timezone=America%2FSao_Paulo"
    )

    response = requests.get(url)

    response.raise_for_status()

    data = response.json()

    daily = data["daily"]

    weather_df = pd.DataFrame({

        "DATA": daily["time"],

        "TEMPERATURA_MAX":
        daily["temperature_2m_max"],

        "TEMPERATURA_MIN":
        daily["temperature_2m_min"],

        "TEMPERATURA_MEDIA":
        daily["temperature_2m_mean"],

        "CHUVA_MM":
        daily["precipitation_sum"],
    })

    weather_df["DATA"] = pd.to_datetime(
        weather_df["DATA"]
    )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    weather_df = process_weather_features(
        weather_df
    )

    # =====================================================
    # EXPORTAÇÃO
    # =====================================================

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    weather_df.to_parquet(
        output_path,
        index=False
    )

    print(
        f"✅ Dados climáticos salvos em: {output_path}"
    )

    return weather_df


def process_weather_features(
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Descrição:
        Cria features derivadas climáticas.

    Parâmetros:
        weather_df (pd.DataFrame): Base climática.

    Retorno:
        pd.DataFrame
    """

    # =====================================================
    # DIA CHUVOSO
    # =====================================================

    weather_df["DIA_CHUVOSO"] = (

        weather_df["CHUVA_MM"]

        > 0
    )

    # =====================================================
    # CATEGORIA TEMPERATURA
    # =====================================================

    def classificar_temperatura(temp):

        if pd.isna(temp):

            return "DESCONHECIDO"

        elif temp <= 10:

            return "MUITO_FRIO"

        elif temp <= 18:

            return "FRIO"

        elif temp <= 25:

            return "AMENO"

        elif temp <= 30:

            return "QUENTE"

        else:

            return "MUITO_QUENTE"

    weather_df["CAT_TEMPERATURA"] = (

        weather_df["TEMPERATURA_MEDIA"]

        .apply(classificar_temperatura)
    )

    # =====================================================
    # ORDENAÇÃO CATEGORIA
    # =====================================================

    ordem_temperatura = [

        "MUITO_FRIO",

        "FRIO",

        "AMENO",

        "QUENTE",

        "MUITO_QUENTE"
    ]

    weather_df["CAT_TEMPERATURA"] = pd.Categorical(

        weather_df["CAT_TEMPERATURA"],

        categories=ordem_temperatura,

        ordered=True
    )

    return weather_df


def load_weather_data(
    weather_path: str
) -> pd.DataFrame:
    """
    Descrição:
        Carrega base climática parquet.

    Parâmetros:
        weather_path (str): Caminho parquet.

    Retorno:
        pd.DataFrame
    """

    weather_path = Path(
        weather_path
    )

    if not weather_path.exists():

        raise FileNotFoundError(

            f"Arquivo climático não encontrado: "
            f"{weather_path}"
        )

    print(
        f"🌦 Carregando dados climáticos: "
        f"{weather_path}"
    )

    weather_df = pd.read_parquet(
        weather_path
    )

    weather_df["DATA"] = pd.to_datetime(
        weather_df["DATA"]
    )

    return weather_df