"""
Descrição:
    Página Database do dashboard responsável pela visualização
    completa da base de dados com filtros interativos por coluna.

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

from src.core.logger import setup_logger


logger = setup_logger(__name__)


def render(
    df: pd.DataFrame
) -> None:
    """
    Descrição:
        Renderiza página de visualização completa
        da base de dados com filtros dinâmicos.

    Parâmetros:
        df (pd.DataFrame): DataFrame analítico.

    Referências:
        - Streamlit Documentation.
        - Pandas Documentation.
    """

    logger.info(
        'Renderizando página Database'
    )

    st.title(
        '🗄️ Database'
    )

    temp = df.copy()

    st.subheader(
        '🔎 Filtros da Base'
    )

    selected_columns = st.multiselect(

        'Selecionar colunas para filtrar',

        options=temp.columns.tolist(),

        default=[]
    )

    for column in selected_columns:

        unique_values = (

            temp[column]

            .dropna()

            .unique()
        )

        unique_values = sorted(
            unique_values
        )

        selected_values = st.multiselect(

            f'Filtrar {column}',

            options=unique_values,

            default=unique_values
        )

        temp = temp[
            temp[column]
            .isin(selected_values)
        ]

    st.divider()

    st.subheader(
        '📄 Base de Dados'
    )

    formatted_rows = (

        f"{len(temp):,}"

        .replace(',', 'X')

        .replace('.', ',')

        .replace('X', '.')
    )

    st.write(
        f'Registros encontrados: {formatted_rows}'
    )

    styled_df = temp.style.format({

        'preco_unitario': '{:,.2f}',

        'preco_total': '{:,.2f}',

        'quantidade': '{:,.2f}',

        'valor_total_nota': '{:,.2f}'
    })

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=700
    )

    logger.info(
        'Página Database renderizada com sucesso'
    )