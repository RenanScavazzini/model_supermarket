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
    5.0 - 19/05/2026 - Refatoração completa para métricas dinâmicas nos gráficos.
    6.0 - 19/05/2026 - Inclusão de ordenações customizadas nos gráficos categóricos.

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

    # =====================================================
    # FILTROS
    # =====================================================

    df, selected_metric = apply_filters(df)

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

    max_invoice_value = (

        df[
            'valor_total_nota'
        ]

        .drop_duplicates()

        .max()
    )

    metric_cols = st.columns(
        [1.12, 0.2, 1, 1, 1, 1],
        gap="medium"
    )

    metrics = [

        {
            "image": "image/ui/slime3.png",

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
            "image": "image/ui/slime1.png",

            "label": "🎟️ Ticket Médio",

            "value":
            format_currency(
                avg_ticket
            )
        },

        {
            "image": "image/ui/slime4.png",

            "label": "🛒 Produtos Distintos",

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

    metric_render_cols = [

        metric_cols[0],

        metric_cols[2],

        metric_cols[3],

        metric_cols[4],

        metric_cols[5]
    ]

    for col, metric in zip(
        metric_render_cols,
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
    # ÁREA PRINCIPAL + SHION
    # =====================================================

    main_area, shion_area = st.columns(
        [0.80, 0.20],
        gap="medium"
    )

    # =====================================================
    # ÁREA PRINCIPAL
    # =====================================================

    with main_area:

        top_left, top_right = st.columns(
            [1, 1],
            gap="large"
        )

        # =================================================
        # SUPERMERCADO
        # =================================================

        with top_left:

            st.subheader(
                "🏪 Gastos por Supermercado"
            )

            fig_market = bar_chart(

                data=df,

                x="supermercado",

                metric=selected_metric
            )

            st.plotly_chart(

                fig_market,

                use_container_width=True,

                key="overview_market_chart"
            )

        # =================================================
        # PERÍODO
        # =================================================

        with top_right:

            st.subheader(
                "🕒 Gastos por Período"
            )

            fig_period = bar_chart(

                data=df,

                x="periodo_dia",

                metric=selected_metric,

                category_orders={

                    'periodo_dia': [

                        'MADRUGADA',

                        'MANHA',

                        'TARDE',

                        'NOITE'
                    ]
                }
            )

            st.plotly_chart(

                fig_period,

                use_container_width=True,

                key="overview_period_chart"
            )

        st.divider()

        # =================================================
        # CATEGORIA
        # =================================================

        st.subheader(
            '📦 Gastos por Categoria'
        )

        category_left, category_right = st.columns(
            [0.18, 0.82],
            gap="medium"
        )

        with category_left:

            st.markdown(
                "<div style='height:120px'></div>",
                unsafe_allow_html=True
            )

            st.image(
                'image/ui/shuna.png',
                width=200
            )

        with category_right:

            fig_category = bar_chart(

                data=df,

                x='categoria_produto',

                metric=selected_metric,

                top_n=20
            )

            st.plotly_chart(

                fig_category,

                use_container_width=True,

                key='overview_category_chart'
            )

    # =====================================================
    # SHION
    # =====================================================

    with shion_area:

        st.markdown(
            "<div style='height:180px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/shion.png',
            width=340
        )

    # =====================================================
    # FERIADO + DIA SEMANA
    # =====================================================

    st.divider()

    holiday_col, weekday_col, hakurou_col = st.columns(
        [0.25, 0.5, 0.20],
        gap="medium"
    )

    # =====================================================
    # FERIADO
    # =====================================================

    with holiday_col:

        st.subheader(
            '🎉 Gastos em Feriados'
        )

        fig_holiday = bar_chart(

            data=df,

            x='feriado',

            metric=selected_metric
        )

        st.plotly_chart(

            fig_holiday,

            use_container_width=True,

            key='overview_holiday_chart'
        )

    # =====================================================
    # DIA SEMANA
    # =====================================================

    with weekday_col:

        st.subheader(
            '📅 Gastos por Dia da Semana'
        )

        fig_weekday = bar_chart(

            data=df,

            x='dia_semana',

            metric=selected_metric,

            category_orders={

                'dia_semana': [

                    'DOMINGO',

                    'SEGUNDA',

                    'TERCA',

                    'QUARTA',

                    'QUINTA',

                    'SEXTA',

                    'SABADO'
                ]
            }
        )

        st.plotly_chart(

            fig_weekday,

            use_container_width=True,

            key='overview_weekday_chart'
        )

    # =====================================================
    # HAKUROU
    # =====================================================

    with hakurou_col:

        st.markdown(
            "<div style='height:20px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/hakurou.png',
            width=200
        )

    # =====================================================
    # ESTAÇÃO + CHUVA + TEMPERATURA
    # =====================================================

    st.divider()

    season_col, rain_col, temp_col, kurobee_col = st.columns(
        [0.28, 0.28, 0.28, 0.16],
        gap="medium"
    )

    # =====================================================
    # ESTAÇÃO
    # =====================================================

    with season_col:

        st.subheader(
            '🍂 Gastos por Estação'
        )

        fig_season = bar_chart(

            data=df,

            x='estacao_ano',

            metric=selected_metric,

            category_orders={

                'estacao_ano': [

                    'PRIMAVERA',

                    'VERAO',

                    'OUTONO',

                    'INVERNO'
                ]
            }
        )

        st.plotly_chart(

            fig_season,

            use_container_width=True,

            key='overview_season_chart'
        )

    # =====================================================
    # CHUVA
    # =====================================================

    with rain_col:

        st.subheader(
            '🌧️ Gastos em Dias Chuvosos'
        )

        fig_rain = bar_chart(

            data=df,

            x='dia_chuvoso',

            metric=selected_metric
        )

        st.plotly_chart(

            fig_rain,

            use_container_width=True,

            key='overview_rain_chart'
        )

    # =====================================================
    # TEMPERATURA
    # =====================================================

    with temp_col:

        st.subheader(
            '🌡️ Gastos por Temperatura'
        )

        fig_temp = bar_chart(

            data=df,

            x='cat_temperatura',

            metric=selected_metric,

            category_orders={

                'cat_temperatura': [

                    'MUITO_FRIO',

                    'FRIO',

                    'AMENO',

                    'QUENTE',

                    'MUITO_QUENTE'
                ]
            }
        )

        st.plotly_chart(

            fig_temp,

            use_container_width=True,

            key='overview_temp_chart'
        )

    # =====================================================
    # KUROBEE
    # =====================================================

    with kurobee_col:

        st.markdown(
            "<div style='height:140px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/kurobee.png',
            width=200
        )

    logger.info(
        'Página Overview renderizada com sucesso'
    )