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

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from src.dashboard.components.charts import (
    line_chart
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

    st.title(
        '🛒 Produtos'
    )

    analyzer = ProductAnalyzer(df)

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

        st.subheader(
            '📋 Resumo do Produto'
        )

        summary_col, image_col = st.columns(
            [0.70, 0.30],
            gap="large"
        )

        # =====================================================
        # RESUMOS
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
                width=180
            )

        st.divider()

        # =====================================================
        # HISTÓRICO
        # =====================================================

        st.subheader(
            '📄 Histórico de Compras'
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

        fig = line_chart(
            result,
            x='data_hora',
            y='preco_unitario',
            title='Histórico de preços',
            color='produto'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            'Digite um produto ou código para iniciar a análise.'
        )

    logger.info(
        'Página Products renderizada com sucesso'
    )