"""
Descrição:
    Página do dashboard responsável pelas análises comparativas
    entre supermercados.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis.summary_analyzer import SummaryAnalyzer

from src.core.logger import setup_logger


logger = setup_logger(__name__)


def render(
    df: pd.DataFrame
) -> None:
    """
    Descrição:
        Renderiza página de comparação entre supermercados.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados analíticos.

    Referências:
        - Streamlit Inc. (2024). Streamlit Documentation.
    """

    logger.info(
        'Renderizando página Supermarkets'
    )

    st.title(
        '🏪 Supermercados'
    )

    spending = (
        SummaryAnalyzer
        .total_por_supermercado(df)
    )

    st.subheader(
        '💰 Gastos por Supermercado'
    )

    fig = px.bar(
        spending,
        x='supermercado',
        y='preco_total',
        color='supermercado',
        title='Gastos Totais'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        '📊 Ticket Médio por Supermercado'
    )

    ticket = (
        SummaryAnalyzer
        .ticket_medio_por_supermercado(df)
    )

    fig_ticket = px.bar(
        ticket,
        x='supermercado',
        y='ticket_medio',
        color='supermercado',
        title='Ticket Médio'
    )

    st.plotly_chart(
        fig_ticket,
        use_container_width=True
    )

    logger.info(
        'Página Supermarkets renderizada com sucesso'
    )