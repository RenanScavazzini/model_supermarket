"""
Descrição:
    Página do dashboard responsável pelas análises temporais,
    incluindo evolução de gastos ao longo do tempo e visualizações
    interativas de séries temporais.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 11/05/2026
    1.1 - 12/05/2026 - Refatoração e melhorias de código

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import plotly.express as px
import pandas as pd

from src.analysis.temporal_analyzer import TemporalAnalyzer

from src.core.logger import setup_logger


logger = setup_logger(__name__)


def render(
    df: pd.DataFrame
) -> None:
    """
    Descrição:
        Renderiza página de análises temporais.

    Parâmetros:
        df (pd.DataFrame): DataFrame analítico.

    Referências:
        - Streamlit Documentation.
        - Plotly Documentation.
    """

    logger.info(
        'Renderizando página Temporal'
    )

    st.title(
        "📅 Análises Temporais"
    )

    analyzer = TemporalAnalyzer(df)

    st.subheader(
        "📈 Evolução Mensal dos Gastos"
    )

    monthly = (
        analyzer
        .monthly_spending()
    )

    fig_monthly = px.line(

        monthly,

        x='mes',

        y='preco_total',

        title='Gastos Mensais'
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "📊 Evolução Anual dos Gastos"
    )

    yearly = (
        analyzer
        .yearly_spending()
    )

    fig_yearly = px.bar(

        yearly,

        x='ano',

        y='preco_total',

        title='Gastos Anuais'
    )

    st.plotly_chart(
        fig_yearly,
        use_container_width=True
    )

    logger.info(
        'Página Temporal renderizada com sucesso'
    )