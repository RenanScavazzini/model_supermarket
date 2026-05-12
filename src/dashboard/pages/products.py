"""
Descrição:
    Página do dashboard responsável pelas análises de produtos,
    incluindo busca textual, histórico de preços e visualizações
    interativas relacionadas aos itens das notas fiscais.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 11/05/2026
    1.1 - 12/05/2026 - Refatoração e melhorias de código

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import plotly.express as px
import pandas as pd

from src.analysis.product_analyzer import ProductAnalyzer

from src.core.logger import setup_logger


logger = setup_logger(__name__)


def render(
    df: pd.DataFrame
) -> None:
    """
    Descrição:
        Renderiza página de análises de produtos.

    Parâmetros:
        df (pd.DataFrame): DataFrame analítico.

    Referências:
        - Streamlit Documentation.
        - Plotly Documentation.
    """

    logger.info(
        'Renderizando página Products'
    )

    st.title(
        "🛒 Produtos"
    )

    analyzer = ProductAnalyzer(df)

    product = st.text_input(
        "Pesquisar produto"
    )

    if product:

        logger.info(
            f'Busca realizada: {product}'
        )

        result = analyzer.search_product(
            product
        )

        if result.empty:

            st.warning(
                "Nenhum produto encontrado."
            )

            logger.warning(
                f'Nenhum resultado para: {product}'
            )

            return

        st.subheader(
            "📋 Resultados da Busca"
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "📈 Histórico de Preços"
        )

        fig = px.line(

            result,

            x='data_hora',

            y='preco_unitario',

            color='produto',

            title='Histórico de preços'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        logger.info(
            f'Histórico renderizado para: {product}'
        )

    else:

        st.info(
            "Digite um produto para iniciar a análise."
        )

    logger.info(
        'Página Products renderizada com sucesso'
    )