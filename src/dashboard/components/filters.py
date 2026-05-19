"""
Descrição:
    Componentes de filtros globais do dashboard.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026
    2.0 - 12/05/2026 - Adição de novos filtros e melhorias na lógica de aplicação dos filtros.
    3.0 - 19/05/2026 - Inclusão de filtros climáticos e temporais.
    4.0 - 19/05/2026 - Inclusão de métricas globais de visualização: Valor Total Tributos, Correção das métricas, Melhor arquitetura de filtros

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import streamlit as st


# ==========================================================
# CONFIGURAÇÃO DAS MÉTRICAS
# ==========================================================

METRIC_OPTIONS = {

    'Preço Total': 'preco_total',

    'Quantidade de Produtos': 'quantidade',

    'Quantidade de Notas': 'qtd_notas',

    'Valor Total Tributos': 'valor_total_tributos'
}


# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def apply_filters(
    df: pd.DataFrame
) -> tuple:

    temp = df.copy()

    st.sidebar.header(
        'Filtros'
    )

    # ======================================================
    # LISTAS DE FILTROS
    # ======================================================

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

    dias_semana = sorted(
        temp['dia_semana'].dropna().unique()
    )

    feriados = sorted(
        temp['feriado'].dropna().unique()
    )

    estacoes = sorted(
        temp['estacao_ano'].dropna().unique()
    )

    categorias_temperatura = sorted(
        temp['cat_temperatura'].dropna().unique()
    )

    dias_chuvosos = sorted(
        temp['dia_chuvoso'].dropna().unique()
    )

    # ======================================================
    # MÉTRICA GLOBAL
    # ======================================================

    metric_filter = st.sidebar.selectbox(

        'Métrica do Gráfico',

        list(
            METRIC_OPTIONS.keys()
        )
    )

    selected_metric = METRIC_OPTIONS[
        metric_filter
    ]

    # ======================================================
    # COMPONENTES SIDEBAR
    # ======================================================

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

    dia_semana_filter = st.sidebar.multiselect(
        'Dia da Semana',
        dias_semana
    )

    feriado_filter = st.sidebar.multiselect(
        'Feriado',
        feriados
    )

    estacao_filter = st.sidebar.multiselect(
        'Estação do Ano',
        estacoes
    )

    cat_temperatura_filter = st.sidebar.multiselect(
        'Categoria Temperatura',
        categorias_temperatura
    )

    dia_chuvoso_filter = st.sidebar.multiselect(
        'Dia Chuvoso',
        dias_chuvosos
    )

    # ======================================================
    # APLICAÇÃO DOS FILTROS
    # ======================================================

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

    if dia_semana_filter:

        temp = temp[
            temp['dia_semana'].isin(
                dia_semana_filter
            )
        ]

    if feriado_filter:

        temp = temp[
            temp['feriado'].isin(
                feriado_filter
            )
        ]

    if estacao_filter:

        temp = temp[
            temp['estacao_ano'].isin(
                estacao_filter
            )
        ]

    if cat_temperatura_filter:

        temp = temp[
            temp['cat_temperatura'].isin(
                cat_temperatura_filter
            )
        ]

    if dia_chuvoso_filter:

        temp = temp[
            temp['dia_chuvoso'].isin(
                dia_chuvoso_filter
            )
        ]

    # ======================================================
    # RETORNO
    # ======================================================

    return (
        temp,
        selected_metric
    )