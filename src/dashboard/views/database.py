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
    3.0 - 19/05/2026 - Melhorias de performance, formatação e compatibilidade com filtros globais.

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


# ==========================================================
# FORMATADOR BRASILEIRO
# ==========================================================

def br_number(
    value
):
    """
    Descrição:
        Formata valores numéricos no padrão brasileiro.
    """

    try:

        return (

            f'{value:,.2f}'

            .replace(',', 'X')

            .replace('.', ',')

            .replace('X', '.')
        )

    except Exception:

        return value


# ==========================================================
# RENDER
# ==========================================================

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

    # ======================================================
    # FILTROS GLOBAIS
    # ======================================================

    filtered_df, _ = apply_filters(df)

    st.title(
        '🗄️ Database'
    )

    temp = filtered_df.copy()

    # ======================================================
    # FILTROS DINÂMICOS
    # ======================================================

    st.subheader(
        '🔎 Filtros da Base'
    )

    selected_columns = st.multiselect(

        'Selecionar colunas para filtrar',

        options=temp.columns.tolist(),

        default=[]
    )

    for column in selected_columns:

        try:

            unique_values = (

                temp[column]

                .dropna()

                .astype(str)

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

                temp[
                    column
                ]

                .astype(str)

                .isin(
                    selected_values
                )
            ]

        except Exception as error:

            logger.warning(
                f'Erro ao aplicar filtro na coluna {column}: {error}'
            )

    st.divider()

    # ======================================================
    # BASE DE DADOS
    # ======================================================

    st.subheader(
        '📄 Base de Dados'
    )

    styled_df = temp.style.format({

        # ==================================================
        # VALORES MONETÁRIOS
        # ==================================================

        'preco_unitario': br_number,

        'preco_total': br_number,

        'valor_total_nota': br_number,

        'valor_total_tributos': br_number,

        # ==================================================
        # QUANTIDADE
        # ==================================================

        'quantidade': br_number,

        'qtd_total_nota': br_number,

        # ==================================================
        # TEMPERATURA
        # ==================================================

        'temperatura_max': br_number,

        'temperatura_min': br_number,

        'temperatura_media': br_number,

        # ==================================================
        # CHUVA
        # ==================================================

        'chuva_mm': br_number
    })

    # ======================================================
    # LAYOUT
    # ======================================================

    col_image, col_table = st.columns(
        [0.14, 0.86],
        gap="large"
    )

    # ======================================================
    # IMAGEM
    # ======================================================

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

    # ======================================================
    # TABELA
    # ======================================================

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