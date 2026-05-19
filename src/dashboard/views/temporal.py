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
    3.0 - 13/05/2026 - Adição dos personagens Souei, Benimaru e Ranga.
    4.0 - 13/05/2026 - Reorganização da ordem dos gráficos e melhoria da evolução diária.
    5.0 - 19/05/2026 - Refatoração completa para métricas dinâmicas.
    6.0 - 19/05/2026 - Adição da evolução diária completa com dias sem compras.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.dashboard.components.filters import (
    apply_filters
)

from src.dashboard.components.charts import (
    line_chart,
    bar_chart,
    apply_brazilian_format,
    remove_plotly_title
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

    # =====================================================
    # FILTROS
    # =====================================================

    df, metric = apply_filters(df)

    st.title(
        '📅 Análises Temporais'
    )

    # =====================================================
    # EVOLUÇÃO ANUAL
    # =====================================================

    st.subheader(
        '📊 Evolução Anual'
    )

    benimaru_col, yearly_chart_col = st.columns(
        [0.16, 0.84],
        gap="medium"
    )

    with benimaru_col:

        st.markdown(
            "<div style='height:120px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/benimaru.png',
            width=240
        )

    with yearly_chart_col:

        fig_yearly = bar_chart(

            data=df,

            x='ano',

            metric=metric
        )

        st.plotly_chart(

            fig_yearly,

            use_container_width=True,

            key='temporal_year_chart'
        )

    st.divider()

    # =====================================================
    # EVOLUÇÃO MENSAL
    # =====================================================

    monthly_chart_col, ranga_col = st.columns(
        [0.70, 0.30],
        gap="medium"
    )

    with monthly_chart_col:

        st.subheader(
            '📈 Evolução Mensal'
        )

        monthly_temp = df.copy()

        monthly_temp['mes_ano'] = (

            monthly_temp['mes_ano']

            .astype(str)
        )

        fig_monthly = line_chart(

            data=monthly_temp,

            x='mes_ano',

            metric=metric
        )

        st.plotly_chart(

            fig_monthly,

            use_container_width=True,

            key='temporal_month_chart'
        )

    with ranga_col:

        st.markdown(
            "<div style='height:180px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/ranga.png',
            width=300
        )

    st.divider()

    # =====================================================
    # EVOLUÇÃO DIÁRIA
    # =====================================================

    st.subheader(
        '📆 Evolução Diária'
    )

    souei_col, daily_chart_col = st.columns(
        [0.16, 0.84],
        gap="medium"
    )

    daily_temp = df.copy()

    daily_temp['dia_ordem'] = (

        daily_temp['data_hora']

        .dt.normalize()
    )

    daily_temp['dia'] = (

        daily_temp['dia_ordem']

        .dt.strftime('%d/%m/%Y')
    )

    with daily_chart_col:

        fig_daily = line_chart(

            data=daily_temp,

            x='dia',

            metric=metric
        )

        st.plotly_chart(

            fig_daily,

            use_container_width=True,

            key='temporal_daily_chart'
        )

    with souei_col:

        st.markdown(
            "<div style='height:120px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/souei.png',
            width=240
        )

    st.divider()

    # =====================================================
    # EVOLUÇÃO DIÁRIA COMPLETA
    # =====================================================

    st.subheader(
        '📉 Evolução Diária Completa'
    )

    full_daily = df.copy()

    full_daily['dia_ordem'] = (

        full_daily['data_hora']

        .dt.normalize()
    )

    # =====================================================
    # AGREGAÇÃO
    # =====================================================

    if metric == 'preco_total':

        grouped = (

            full_daily

            .groupby('dia_ordem')[
                'preco_total'
            ]

            .sum()
        )

    elif metric == 'quantidade':

        grouped = (

            full_daily

            .groupby('dia_ordem')

            .size()
        )

    elif metric == 'qtd_notas':

        grouped = (

            full_daily

            .groupby('dia_ordem')[
                'chave_anonimizada'
            ]

            .nunique()
        )

    elif metric == 'valor_total_tributos':

        dedup = (

            full_daily[[
                'dia_ordem',
                'chave_anonimizada',
                'valor_total_tributos'
            ]]

            .drop_duplicates(
                subset=['chave_anonimizada']
            )
        )

        grouped = (

            dedup

            .groupby('dia_ordem')[
                'valor_total_tributos'
            ]

            .sum()
        )

    # =====================================================
    # RANGE COMPLETO
    # =====================================================

    full_range = pd.date_range(

        start=grouped.index.min(),

        end=grouped.index.max(),

        freq='D'
    )

    grouped = (

        grouped

        .reindex(
            full_range,
            fill_value=0
        )

        .reset_index()
    )

    grouped.columns = [

        'dia_ordem',

        'valor'
    ]

    grouped['dia'] = (

        grouped['dia_ordem']

        .dt.strftime('%d/%m/%Y')
    )

    # =====================================================
    # GRÁFICO COMPLETO
    # =====================================================

    fig_complete = px.line(

        grouped,

        x='dia_ordem',

        y='valor',

        markers=False
    )

    # =====================================================
    # LABELS DO EIXO
    # =====================================================

    fig_complete.update_xaxes(

        tickformat='%d/%m/%Y'
    )

    # =====================================================
    # REMOVE TÍTULO
    # =====================================================

    fig_complete = remove_plotly_title(
        fig_complete
    )

    # =====================================================
    # LAYOUT
    # =====================================================

    fig_complete.update_layout(

        plot_bgcolor='rgba(0,0,0,0)',

        paper_bgcolor='rgba(0,0,0,0)',

        hovermode='x unified'
    )

    # =====================================================
    # LINHA
    # =====================================================

    fig_complete.update_traces(

        line=dict(
            width=3
        )
    )

    # =====================================================
    # FORMATAÇÃO
    # =====================================================

    fig_complete = apply_brazilian_format(

        fig_complete,

        metric
    )

    st.plotly_chart(

        fig_complete,

        use_container_width=True,

        key='temporal_complete_daily_chart'
    )

    st.divider()

    logger.info(
        'Página Temporal renderizada com sucesso'
    )