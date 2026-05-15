"""
Descrição:
    Página Products do dashboard responsável pelas análises
    detalhadas de produtos, histórico de preços e resumos
    analíticos de consumo.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 11/05/2026
    1.1 - 12/05/2026 - Refatoração e melhorias de código
    2.0 - 12/05/2026 - Adição de novas funcionalidades e melhorias na lógica de análise dos produtos.
    3.0 - 13/05/2026 - Adição do mascote Veldora na seção de resumo.
    4.0 - 13/05/2026 - Integração com filtros globais do dashboard.
    5.0 - 13/05/2026 - Adição dos personagens Gobta, Ramiris e Milim.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from src.dashboard.components.charts import (
    line_chart
)

from src.dashboard.components.filters import (
    apply_filters
)

from src.analysis.product_analyzer import ProductAnalyzer

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
        Renderiza página de produtos.

    Parâmetros:
        df (pd.DataFrame): DataFrame analítico.
    """

    logger.info(
        'Renderizando página Products'
    )

    # =====================================================
    # FILTROS GLOBAIS
    # =====================================================

    filtered_df = apply_filters(df)

    st.title(
        '🛒 Produtos'
    )

    analyzer = ProductAnalyzer(
        filtered_df
    )

    # =====================================================
    # ÁREA SUPERIOR
    # =====================================================

    top_left, top_right = st.columns(
        [0.12, 0.88],
        gap="small"
    )

    # =====================================================
    # GOBTA
    # =====================================================

    with top_left:

        st.markdown(
            "<div style='height:6px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/gobta.png',
            width=90
        )

    # =====================================================
    # PESQUISA
    # =====================================================

    with top_right:

        product_name = st.text_input(
            'Pesquisar produto'
        )

        product_code = st.text_input(
            'Pesquisar código do produto'
        )

    result = pd.DataFrame()

    if product_name:

        result = analyzer.search_product(
            product_name
        )

    elif product_code:

        result = analyzer.search_product_code(
            int(product_code)
        )

    # =====================================================
    # RESULTADOS
    # =====================================================

    if not result.empty:

        produtos_disponiveis = sorted(

            result[
                'produto'
            ]

            .dropna()

            .unique()
        )

        produtos_selecionados = st.multiselect(
            'Filtrar produtos encontrados',
            produtos_disponiveis,
            default=produtos_disponiveis
        )

        result = result[

            result[
                'produto'
            ]

            .isin(
                produtos_selecionados
            )
        ]

        summary = analyzer.product_summary(
            result
        )

        if not summary:

            return

        # =====================================================
        # RESUMO
        # =====================================================

        st.subheader(
            '📋 Resumo do Produto'
        )

        summary_col, image_col = st.columns(
            [0.85, 0.30],
            gap="large"
        )

        # =====================================================
        # MÉTRICAS
        # =====================================================

        with summary_col:

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    '🛒 Produto(s)',
                    summary['Nome']
                )

                st.metric(
                    '🔢 Código(s)',
                    summary['Código']
                )

                st.metric(
                    '📦 Categoria(s)',
                    summary['Categoria']
                )

                st.metric(
                    '📏 Unidade(s)',
                    summary['Unidade']
                )

            with col2:

                st.metric(
                    '💰 Total Gasto',
                    format_currency(
                        summary['Total gasto']
                    )
                )

                st.metric(
                    '📊 Quantidade Comprada',
                    format_number(
                        summary['Quantidade comprada']
                    )
                )

                st.metric(
                    '📉 Valor Mínimo',
                    format_currency(
                        summary['Valor mínimo']
                    )
                )

                st.metric(
                    '📈 Valor Máximo',
                    format_currency(
                        summary['Valor máximo']
                    )
                )

            with col3:

                st.metric(
                    '🧮 Valor Médio',
                    format_currency(
                        summary['Valor médio']
                    )
                )

                st.metric(
                    '🕒 Período Mais Comprado',
                    summary['Período mais comprado']
                )

                st.metric(
                    '🏪 Supermercado(s)',
                    summary['Supermercados']
                )

        # =====================================================
        # VELDORA
        # =====================================================

        with image_col:

            st.image(
                'image/ui/veldora.png',
                width=220
            )

        st.divider()

        # =====================================================
        # HISTÓRICO DE COMPRAS
        # =====================================================

        history_title_col, history_image_col = st.columns(
            [0.3, 0.8]
        )

        with history_title_col:

            st.subheader(
                '📄 Histórico de Compras'
            )

        with history_image_col:

            st.image(
                'image/ui/ramiris.png',
                width=85
            )

        display_columns = [

            'produto',

            'codigo_produto',

            'preco_unitario',

            'quantidade',

            'preco_total',

            'data_hora',

            'supermercado'
        ]

        styled_df = (

            result[
                display_columns
            ]

            .style

            .format({

                'preco_unitario': '{:,.2f}',

                'quantidade': '{:,.2f}',

                'preco_total': '{:,.2f}'
            })
        )

        st.dataframe(
            styled_df,
            use_container_width=True
        )

        st.divider()

        # =====================================================
        # HISTÓRICO DE PREÇOS
        # =====================================================

        st.subheader(
            '📈 Histórico de Preços'
        )
        
        milim_col, price_chart_col = st.columns(
            [0.2, 0.8],
            gap="medium"
        )

        # =================================================
        # MILIM
        # =================================================

        with milim_col:

            st.markdown(
                "<div style='height:120px'></div>",
                unsafe_allow_html=True
            )

            st.image(
                'image/ui/milim.png',
                width=245
            )

        # =================================================
        # GRÁFICO
        # =================================================

        with price_chart_col:

            fig = line_chart(
                result,
                x='data_hora',
                y='preco_unitario',
                title='Histórico de preços',
                color='produto'
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key='products_price_history_chart'
            )

    else:

        st.info(
            'Digite um produto ou código para iniciar a análise.'
        )

    logger.info(
        'Página Products renderizada com sucesso'
    )