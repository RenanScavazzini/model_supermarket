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

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from src.dashboard.components.filters import (
    apply_filters
)

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

        fig_monthly = line_chart(

            data=df,

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

    # =====================================================
    # PREPARAÇÃO DATAFRAME DIÁRIO
    # =====================================================

    daily_temp = df.copy()

    daily_temp['dia'] = (

        daily_temp[
            'data_hora'
        ]

        .dt.date
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

    logger.info(
        'Página Temporal renderizada com sucesso'
    )