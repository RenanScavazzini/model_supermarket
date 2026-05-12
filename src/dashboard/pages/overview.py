"""
Descrição:
    Página principal do dashboard responsável pela exibição de indicadores
    gerais, métricas consolidadas e análises agregadas de gastos.

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

from src.analysis.summary_analyzer import SummaryAnalyzer

from src.dashboard.components.metrics import metric_card

from src.core.logger import setup_logger


logger = setup_logger(__name__)


def render(
    df: pd.DataFrame
) -> None:
    """
    Descrição:
        Renderiza a página de visão geral do dashboard contendo
        métricas principais e gráfico agregado por supermercado.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados analíticos.

    Referências:
        - Streamlit Inc. (2024). Streamlit Documentation.
        - Plotly Technologies Inc. (2024). Plotly Documentation.
    """

    logger.info(
        'Renderizando página Overview'
    )

    st.title("📊 Visão Geral")

    report = (
        SummaryAnalyzer
        .build_summary_report(df)
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric_card(
            "Total Gasto",
            f"R$ {report.total_gasto:,.2f}"
        )

    with col2:

        metric_card(
            "Notas",
            report.total_notas
        )

    with col3:

        metric_card(
            "Ticket Médio",
            f"R$ {report.gasto_medio:,.2f}"
        )

    with col4:

        metric_card(
            "Maior Compra",
            f"R$ {report.maior_gasto:,.2f}"
        )

    st.divider()

    st.subheader(
        "🏪 Gastos por Supermercado"
    )

    spending = (
        report
        .agrupamentos['por_supermercado']
    )

    st.bar_chart(
        spending,
        x='supermercado',
        y='preco_total'
    )

    st.divider()

    st.subheader(
        "📅 Evolução Mensal dos Gastos"
    )

    monthly = (
        report
        .agrupamentos['por_mes']
    )

    st.line_chart(
        monthly,
        x='mes_ano',
        y='preco_total'
    )

    logger.info(
        'Página Overview renderizada com sucesso'
    )