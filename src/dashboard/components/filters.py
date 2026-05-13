"""
Descrição:
    Componentes de filtros globais do dashboard.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026
    2.0 - 12/05/2026 - Adição de novos filtros e melhorias na lógica de aplicação dos filtros.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import streamlit as st


def apply_filters(
    df: pd.DataFrame
) -> pd.DataFrame:

    temp = df.copy()

    st.sidebar.header(
        'Filtros'
    )

    anos = sorted(
        temp['ano'].dropna().unique()
    )

    meses = sorted(
        temp['mes'].dropna().unique()
    )

    periodos = sorted(
        temp['periodo_dia'].dropna().unique()
    )

    supermercados = sorted(
        temp['supermercado'].dropna().unique()
    )

    categorias = sorted(
        temp['categoria_produto'].dropna().unique()
    )

    ano_filter = st.sidebar.multiselect(
        'Ano',
        anos
    )

    mes_filter = st.sidebar.multiselect(
        'Mês',
        meses
    )

    periodo_filter = st.sidebar.multiselect(
        'Período',
        periodos
    )

    supermercado_filter = st.sidebar.multiselect(
        'Supermercado',
        supermercados
    )

    categoria_filter = st.sidebar.multiselect(
        'Categoria',
        categorias
    )

    if ano_filter:

        temp = temp[
            temp['ano'].isin(
                ano_filter
            )
        ]

    if mes_filter:

        temp = temp[
            temp['mes'].isin(
                mes_filter
            )
        ]

    if periodo_filter:

        temp = temp[
            temp['periodo_dia'].isin(
                periodo_filter
            )
        ]

    if supermercado_filter:

        temp = temp[
            temp['supermercado'].isin(
                supermercado_filter
            )
        ]

    if categoria_filter:

        temp = temp[
            temp['categoria_produto'].isin(
                categoria_filter
            )
        ]

    return temp