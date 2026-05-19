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
    # ÁREA PRINCIPAL + SHION FIXA
    # =====================================================

    main_area, shion_area = st.columns(
        [0.80, 0.20],
        gap="medium"
    )

    # =====================================================
    # ÁREA PRINCIPAL
    # =====================================================

    with main_area:

        # =================================================
        # GRÁFICOS SUPERIORES
        # =================================================

        top_left, top_right = st.columns(
            [1, 1],
            gap="large"
        )

        # ================================================
        # GASTOS POR SUPERMERCADO
        # ================================================

        with top_left:

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

            )

            st.plotly_chart(

                fig_market,

                use_container_width=True,

                key="overview_market_chart"
            )

        # ================================================
        # GASTOS POR PERÍODO
        # ================================================

        with top_right:

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

            order_period = [
                'MANHA',
                'TARDE',
                'NOITE'
            ]

            spending_period['periodo_dia'] = pd.Categorical(

                spending_period['periodo_dia'],

                categories=order_period,

                ordered=True
            )

            spending_period = (

                spending_period

                .sort_values(
                    'periodo_dia'
                )
            )

            fig_period = bar_chart(

                spending_period,

                x="periodo_dia",

                y="preco_total",

            )

            st.plotly_chart(

                fig_period,

                use_container_width=True,

                key="overview_period_chart"
            )

        # =================================================
        # DIVIDER
        # =================================================

        st.divider()

        # =================================================
        # TÍTULO CATEGORIA
        # =================================================

        st.subheader(
            '📦 Gastos por Categoria'
        )

        # =================================================
        # GASTOS POR CATEGORIA
        # =================================================

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

            )

            st.plotly_chart(

                fig_category,

                use_container_width=True,

                key='overview_category_chart'
            )

    # =====================================================
    # SHION FIXA ATRAVESSANDO TUDO
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
    # FERIADOS
    # =====================================================

    with holiday_col:

        st.subheader(
            '🎉 Gastos em Feriados'
        )

        spending_holiday = (

            df

            .groupby('feriado')[
                'preco_total'
            ]

            .sum()

            .reset_index()
        )

        fig_holiday = bar_chart(

            spending_holiday,

            x='feriado',

            y='preco_total'
        )

        st.plotly_chart(

            fig_holiday,

            use_container_width=True,

            key='overview_holiday_chart'
        )

    # =====================================================
    # DIA DA SEMANA
    # =====================================================

    with weekday_col:

        st.subheader(
            '📅 Gastos por Dia da Semana'
        )

        spending_weekday = (

            df

            .groupby('dia_semana')[
                'preco_total'
            ]

            .sum()

            .reset_index()
        )

        ordem_semana = [

            'DOMINGO',

            'SEGUNDA',

            'TERCA',

            'QUARTA',

            'QUINTA',

            'SEXTA',

            'SABADO'
        ]

        spending_weekday['dia_semana'] = pd.Categorical(

            spending_weekday['dia_semana'],

            categories=ordem_semana,

            ordered=True
        )

        spending_weekday = (

            spending_weekday

            .sort_values(
                'dia_semana'
            )
        )

        fig_weekday = bar_chart(

            spending_weekday,

            x='dia_semana',

            y='preco_total'
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
    # ESTAÇÃO DO ANO
    # =====================================================

    with season_col:

        st.subheader(
            '🍂 Gastos por Estação'
        )

        spending_season = (

            df

            .groupby('estacao_ano')[
                'preco_total'
            ]

            .sum()

            .reset_index()
        )

        ordem_estacao = [

            'VERAO',

            'OUTONO',

            'INVERNO',

            'PRIMAVERA'
        ]

        spending_season['estacao_ano'] = pd.Categorical(

            spending_season['estacao_ano'],

            categories=ordem_estacao,

            ordered=True
        )

        spending_season = (

            spending_season

            .sort_values(
                'estacao_ano'
            )
        )

        fig_season = bar_chart(

            spending_season,

            x='estacao_ano',

            y='preco_total'
        )

        st.plotly_chart(

            fig_season,

            use_container_width=True,

            key='overview_season_chart'
        )

    # =====================================================
    # DIA CHUVOSO
    # =====================================================

    with rain_col:

        st.subheader(
            '🌧️ Gastos em Dias Chuvosos'
        )

        spending_rain = (

            df

            .groupby('dia_chuvoso')[
                'preco_total'
            ]

            .sum()

            .reset_index()
        )

        spending_rain['dia_chuvoso'] = (

            spending_rain['dia_chuvoso']

            .map({

                True: 'SIM',

                False: 'NAO'
            })
        )

        fig_rain = bar_chart(

            spending_rain,

            x='dia_chuvoso',

            y='preco_total'
        )

        st.plotly_chart(

            fig_rain,

            use_container_width=True,

            key='overview_rain_chart'
        )

    # =====================================================
    # CATEGORIA TEMPERATURA
    # =====================================================

    with temp_col:

        st.subheader(
            '🌡️ Gastos por Temperatura'
        )

        spending_temp = (

            df

            .groupby('cat_temperatura')[
                'preco_total'
            ]

            .sum()

            .reset_index()
        )

        ordem_temp = [

            'MUITO_FRIO',

            'FRIO',

            'AMENO',

            'QUENTE',

            'MUITO_QUENTE'
        ]

        spending_temp['cat_temperatura'] = pd.Categorical(

            spending_temp['cat_temperatura'],

            categories=ordem_temp,

            ordered=True
        )

        spending_temp = (

            spending_temp

            .sort_values(
                'cat_temperatura'
            )
        )

        fig_temp = bar_chart(

            spending_temp,

            x='cat_temperatura',

            y='preco_total'
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