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
    3.0 - 13/05/2026 - Adição de slimes temáticos na visão geral.
    4.0 - 13/05/2026 - Adição das personagens Shion e Shuna na seção gráfica.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from src.analysis.summary_analyzer import SummaryAnalyzer

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

    # =====================================================
    # MÉTRICAS
    # =====================================================

    total_spent = analyzer.total_spent()

    total_invoices = analyzer.total_invoices()

    avg_ticket = analyzer.avg_ticket()

    distinct_products = df['produto'].nunique()

    max_invoice_value = df['valor_total_nota'].max()

    metric_cols = st.columns(5)

    metrics = [

        {
            "image": "image/ui/slime1.png",

            "label": "💰 Total Gasto",

            "value":
            format_currency(
                total_spent
            )
        },

        {
            "image": "image/ui/slime2.png",

            "label": "🧾 Notas",

            "value":
            format_number(
                total_invoices,
                0
            )
        },

        {
            "image": "image/ui/slime3.png",

            "label": "🎟️ Ticket Médio",

            "value":
            format_currency(
                avg_ticket
            )
        },

        {
            "image": "image/ui/slime4.png",

            "label": "🛒 Qtde Produtos",

            "value":
            format_number(
                distinct_products,
                0
            )
        },

        {
            "image": "image/ui/slime5.png",

            "label": "🔥 Valor Máximo",

            "value":
            format_currency(
                max_invoice_value
            )
        }
    ]

    for col, metric in zip(
        metric_cols,
        metrics
    ):

        with col:

            container = st.container(
                border=False
            )

            with container:

                img_col, text_col = st.columns(
                    [0.32, 0.68]
                )

                with img_col:

                    st.image(
                        metric["image"],
                        width=72
                    )

                with text_col:

                    st.html(

                        f"""
                        <div style="padding-top:10px;">

                            <div style="
                                font-size:20px;
                                color:#DADADA;
                                font-weight:600;
                                margin-bottom:4px;
                            ">
                                {metric["label"]}
                            </div>

                            <div style="
                                font-size:32px;
                                font-weight:700;
                                color:white;
                                line-height:1.1;
                            ">
                                {metric["value"]}
                            </div>

                        </div>
                        """
                    )

    st.divider()

    # =====================================================
    # GRÁFICOS
    # =====================================================

    col_market, col_period = st.columns(
        2,
        gap="large"
    )

    # =====================================================
    # GASTOS POR SUPERMERCADO
    # =====================================================

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

    # =====================================================
    # GASTOS POR PERÍODO
    # =====================================================

    with col_period:

        st.subheader(
            "🕒 Gastos por Período"
        )

        chart_col, image_col = st.columns(
            [0.82, 0.18]
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

        with chart_col:

            st.plotly_chart(

                fig_period,

                use_container_width=True,

                key="overview_period_chart"
            )

        with image_col:

            st.image(
                'image/ui/shion.png',
                width=240
            )

    st.divider()

    # =====================================================
    # GASTOS POR CATEGORIA
    # =====================================================

    st.subheader(
        '📦 Gastos por Categoria'
    )

    chart_col, image_col = st.columns(
        [0.18, 0.82]
    )

    spending_category = (

        df

        .groupby('categoria_produto')[

            'preco_total'
        ]

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

    with image_col:

        st.plotly_chart(

            fig_category,

            use_container_width=True,

            key='overview_category_chart'
        )

    with chart_col:

        st.image(
            'image/ui/shuna.png',
            width=140
        )

    logger.info(
        'Página Overview renderizada com sucesso'
    )