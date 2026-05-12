"""
Descrição:
    Módulo responsável pela criação de filtros interativos utilizados
    no dashboard Streamlit para segmentação dos dados analíticos.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd


def market_filter(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Descrição:
        Cria filtro interativo de supermercados utilizando componente
        multiselect da sidebar do Streamlit.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados analíticos.

    Referências:
        - Streamlit Inc. (2024). Streamlit Documentation.
    """

    markets = st.sidebar.multiselect(
        "Supermercado",
        options=df["SUPERMERCADO"].unique(),
        default=df["SUPERMERCADO"].unique()
    )

    return df[
        df["SUPERMERCADO"].isin(markets)
    ]