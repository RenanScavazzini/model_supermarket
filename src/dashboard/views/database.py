"""
Descrição:
    Página Database do dashboard responsável pela visualização
    completa da base de dados com filtros interativos por coluna.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026
    2.0 - 13/05/2026 - Adição de filtros dinâmicos e estilização visual.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from pathlib import Path

from src.core.logger import setup_logger

from src.dashboard.components.filters import (
    apply_filters
)

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

    filtered_df = apply_filters(df)

    logger.info(
        'Renderizando página Database'
    )

    st.title(
        '🗄️ Database'
    )

    temp = filtered_df.copy()

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

    styled_df = temp.style.format({

        'preco_unitario':

        lambda x:
        f'{x:,.2f}'
        .replace(',', 'X')
        .replace('.', ',')
        .replace('X', '.'),

        'preco_total':

        lambda x:
        f'{x:,.2f}'
        .replace(',', 'X')
        .replace('.', ',')
        .replace('X', '.'),

        'quantidade':

        lambda x:
        f'{x:,.2f}'
        .replace(',', 'X')
        .replace('.', ',')
        .replace('X', '.'),

        'valor_total_nota':

        lambda x:
        f'{x:,.2f}'
        .replace(',', 'X')
        .replace('.', ',')
        .replace('X', '.')
    })

    col_image, col_table = st.columns(
        [0.14, 0.86],
        gap="large"
    )

    with col_image:

        diablo_path = (
            Path(
                "image/ui/diablo.png"
            )
        )

        st.image(
            str(diablo_path),
            width=180
        )

    with col_table:

        st.markdown(

            f"""
            <p style="
                font-size:18px;
                font-weight:600;
                color:white;
                margin-top:10px;
            ">
                📦 Registros encontrados:
                {len(temp):,.0f}
            </p>
            """.replace(",", "."),

            unsafe_allow_html=True
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=650
        )

    logger.info(
        'Página Database renderizada com sucesso'
    )