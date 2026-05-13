"""
Descrição:
    Página Temporal do dashboard responsável pelas análises
    temporais de gastos e evolução do comportamento de consumo.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 11/05/2026
    1.1 - 12/05/2026 - Refatoração e melhorias de código
    2.0 - 12/05/2026 - Adição de novas visualizações e melhorias na interatividade dos gráficos.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from src.analysis.temporal_analyzer import TemporalAnalyzer

from src.dashboard.components.filters import apply_filters

from src.dashboard.components.charts import (
    line_chart,
    bar_chart
)

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
    """

    logger.info(
        'Renderizando página Temporal'
    )

    df = apply_filters(df)

    analyzer = TemporalAnalyzer(df)

    st.title(
        '📅 Análises Temporais'
    )

    st.subheader(
        '📈 Evolução Mensal dos Gastos'
    )

    monthly = (
        analyzer
        .monthly_spending()
    )

    fig_monthly = line_chart(
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
        '📊 Evolução Anual dos Gastos'
    )

    yearly = (
        analyzer
        .yearly_spending()
    )

    fig_yearly = bar_chart(
        yearly,
        x='ano',
        y='preco_total',
        title='Gastos Anuais'
    )

    st.plotly_chart(
        fig_yearly,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        '📆 Evolução Mensal dos Gastos Diários'
    )

    daily_temp = df.copy()

    daily_temp['dia'] = (
        daily_temp['data_hora']
        .dt.date
    )

    daily_temp['mes'] = (
        daily_temp['data_hora']
        .dt.to_period('M')
        .astype(str)
    )

    daily_monthly = (

        daily_temp

        .groupby(['mes', 'dia'])['preco_total']

        .sum()

        .reset_index()
    )

    fig_daily = line_chart(
        daily_monthly,
        x='dia',
        y='preco_total',
        title='Evolução Mensal dos Gastos Diários',
        color='mes'
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )

    logger.info(
        'Página Temporal renderizada com sucesso'
    )