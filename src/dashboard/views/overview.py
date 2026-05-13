"""
Descrição:
    Página Overview do dashboard responsável pela apresentação
    de indicadores gerais, métricas consolidadas e gráficos
    analíticos relacionados ao comportamento de consumo.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026
    2.0 - 12/05/2026 - Adição de novos indicadores, gráficos e melhorias na apresentação visual dos dados.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from src.analysis.summary_analyzer import SummaryAnalyzer

from src.dashboard.components.metrics import metric_card
from src.dashboard.components.filters import apply_filters

from src.dashboard.components.charts import (
    bar_chart
)

from src.core.logger import setup_logger

from src.utils.formatters import (
    format_currency,
    format_number
)


logger = setup_logger(__name__)


def render(
    df: pd.DataFrame
) -> None:
    """
    Descrição:
        Renderiza página de visão geral do dashboard.

    Parâmetros:
        df (pd.DataFrame): DataFrame analítico.
    """

    logger.info(
        'Renderizando página Overview'
    )

    df = apply_filters(df)

    analyzer = SummaryAnalyzer(df)

    st.title(
        '📊 Visão Geral'
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        metric_card(
            'Total Gasto',
            format_currency(
                analyzer.total_spent()
            )
        )

    with col2:

        metric_card(
            'Notas',
            format_number(
                analyzer.total_invoices(),
                0
            )
        )

    with col3:

        metric_card(
            'Ticket Médio',
            format_currency(
                analyzer.avg_ticket()
            )
        )

    with col4:

        metric_card(
            'Qtde Produtos',
            format_number(
                df['produto'].nunique(),
                0
            )
        )

    with col5:

        metric_card(
            'Valor Máximo Gasto',
            format_currency(
                df['valor_total_nota'].max()
            )
        )

    st.divider()

    col_market, col_period = st.columns(
        2,
        gap="large"
    )

    with col_market:

        st.subheader(
            "🏪 Gastos por Supermercado"
        )

        spending_market = (

            df

            .groupby("supermercado")[
                "preco_total"
            ]

            .sum()

            .reset_index()
        )

        fig_market = bar_chart(
            spending_market,
            x="supermercado",
            y="preco_total",
            title="Gastos por Supermercado"
        )

        st.plotly_chart(
            fig_market,
            use_container_width=True,
            key="overview_market_chart"
        )

    with col_period:

        st.subheader(
            "🕒 Gastos por Período"
        )

        spending_period = (

            df

            .groupby("periodo_dia")[
                "preco_total"
            ]

            .sum()

            .reset_index()
        )

        fig_period = bar_chart(
            spending_period,
            x="periodo_dia",
            y="preco_total",
            title="Gastos por Período"
        )

        st.plotly_chart(
            fig_period,
            use_container_width=True,
            key="overview_period_chart"
        )

    spending_market = (

        df

        .groupby('supermercado')['preco_total']

        .sum()

        .reset_index()
    )

    st.divider()

    st.subheader(
        '📦 Gastos por Categoria'
    )

    spending_category = (

        df

        .groupby('categoria_produto')['preco_total']

        .sum()

        .reset_index()

        .sort_values(
            'preco_total',
            ascending=False
        )

        .head(15)
    )

    fig_category = bar_chart(
        spending_category,
        x='categoria_produto',
        y='preco_total',
        title='Gastos por Categoria'
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    logger.info(
        'Página Overview renderizada com sucesso'
    )