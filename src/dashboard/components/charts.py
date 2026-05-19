"""
Descrição:
    Módulo responsável pela criação de gráficos interativos
    utilizando Plotly, fornecendo componentes reutilizáveis
    para visualização de dados no dashboard.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026
    2.0 - 13/05/2026 - Adição de valores acima das barras.
    3.0 - 13/05/2026 - Remoção completa dos títulos internos.
    4.0 - 13/05/2026 - Compatibilidade mobile.
    5.0 - 19/05/2026 - Refatoração completa para métricas dinâmicas.
    6.0 - 19/05/2026 - Correção de ordenação temporal.
    7.0 - 19/05/2026 - Adição de Top N em gráficos categóricos.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import plotly.express as px

from src.utils.formatters import (
    format_currency,
    format_number
)

# ==========================================================
# CONFIGURAÇÃO MÉTRICAS
# ==========================================================

METRIC_CONFIG = {

    'preco_total': {

        'label': 'Preço Total',

        'is_currency': True
    },

    'preco_unitario': {

        'label': 'Preço Unitário',

        'is_currency': True
    },

    'quantidade': {

        'label': 'Quantidade de Registros',

        'is_currency': False
    },

    'qtd_notas': {

        'label': 'Quantidade de Notas',

        'is_currency': False
    },

    'valor_total_tributos': {

        'label': 'Valor Total Tributos',

        'is_currency': True
    }
}


# ==========================================================
# REMOVE TÍTULO
# ==========================================================

def remove_plotly_title(
    fig
):
    """
    Descrição:
        Remove completamente o título interno do Plotly.
    """

    fig.update_layout(

        title_text='',

        title=None,

        margin=dict(
            t=30
        )
    )

    return fig


# ==========================================================
# FORMATAÇÃO BRASILEIRA
# ==========================================================

def apply_brazilian_format(
    fig,
    metric: str
):
    """
    Descrição:
        Aplica formatação brasileira aos gráficos.
    """

    is_currency = METRIC_CONFIG[
        metric
    ][
        'is_currency'
    ]

    fig.update_layout(

        separators=',.'
    )

    if is_currency:

        fig.update_layout(

            yaxis=dict(

                tickprefix='R$ ',

                tickformat=',.2f'
            )
        )

        fig.update_traces(

            hovertemplate=
            'R$ %{y:,.2f}<extra></extra>'
        )

    else:

        fig.update_layout(

            yaxis=dict(

                tickformat=',.0f'
            )
        )

        fig.update_traces(

            hovertemplate=
            '%{y:,.0f}<extra></extra>'
        )

    return fig


# ==========================================================
# PREPARAÇÃO MÉTRICAS
# ==========================================================

def prepare_metric_data(
    data: pd.DataFrame,
    group_col: str,
    metric: str
) -> pd.DataFrame:
    """
    Descrição:
        Prepara dados agregados conforme métrica.
    """

    group_cols = [group_col]

    # ======================================================
    # ORDENAÇÃO TEMPORAL DIÁRIA
    # ======================================================

    if (
        group_col == 'dia'
        and
        'dia_ordem' in data.columns
    ):

        group_cols.append(
            'dia_ordem'
        )

    # ======================================================
    # PREÇO TOTAL
    # ======================================================

    if metric == 'preco_total':

        result = (

            data

            .groupby(group_cols)[
                'preco_total'
            ]

            .sum()

            .reset_index(
                name='valor'
            )
        )

    # ======================================================
    # PREÇO UNITÁRIO
    # ======================================================

    elif metric == 'preco_unitario':

        result = (

            data

            .groupby(group_cols)[
                'preco_unitario'
            ]

            .mean()

            .reset_index(
                name='valor'
            )
        )

    # ======================================================
    # QUANTIDADE
    # ======================================================

    elif metric == 'quantidade':

        result = (

            data

            .groupby(group_cols)

            .size()

            .reset_index(
                name='valor'
            )
        )

    # ======================================================
    # QUANTIDADE NOTAS
    # ======================================================

    elif metric == 'qtd_notas':

        result = (

            data

            .groupby(group_cols)[
                'chave_anonimizada'
            ]

            .nunique()

            .reset_index(
                name='valor'
            )
        )

    # ======================================================
    # TRIBUTOS
    # ======================================================

    elif metric == 'valor_total_tributos':

        dedup = (

            data[[
                *group_cols,
                'chave_anonimizada',
                'valor_total_tributos'
            ]]

            .drop_duplicates(
                subset=['chave_anonimizada']
            )
        )

        result = (

            dedup

            .groupby(group_cols)[
                'valor_total_tributos'
            ]

            .sum()

            .reset_index(
                name='valor'
            )
        )

    else:

        raise ValueError(
            f'Métrica inválida: {metric}'
        )

    return result


# ==========================================================
# GRÁFICO DE BARRAS
# ==========================================================

def bar_chart(
    data: pd.DataFrame,
    x: str,
    metric: str,
    title: str = None,
    category_orders: dict = None,
    top_n: int = 15
):
    """
    Descrição:
        Cria gráfico de barras padronizado.
    """

    plot_data = prepare_metric_data(

        data=data,

        group_col=x,

        metric=metric
    )

    # =====================================================
    # TOP N
    # =====================================================

    if (

        top_n

        and

        not category_orders

    ):

        plot_data = (

            plot_data

            .sort_values(

                by='valor',

                ascending=False
            )

            .head(top_n)
        )

    # =====================================================
    # LABELS FORMATADAS
    # =====================================================

    if METRIC_CONFIG[metric]['is_currency']:

        plot_data['text_label'] = (

            plot_data['valor']

            .apply(format_currency)
        )

    else:

        plot_data['text_label'] = (

            plot_data['valor']

            .apply(
                lambda x: format_number(
                    x,
                    0
                )
            )
        )

    fig = px.bar(

        plot_data,

        x=x,

        y='valor',

        text='text_label',

        category_orders=category_orders
    )

    # =====================================================
    # REMOVE TÍTULO
    # =====================================================

    fig = remove_plotly_title(
        fig
    )

    # =====================================================
    # LAYOUT
    # =====================================================

    fig.update_layout(

        plot_bgcolor='rgba(0,0,0,0)',

        paper_bgcolor='rgba(0,0,0,0)'
    )

    # =====================================================
    # TEXTO DAS BARRAS
    # =====================================================

    fig.update_traces(

        textposition='outside',

        textfont=dict(
            size=13,
            color='white'
        ),

        cliponaxis=False
    )

    # =====================================================
    # FORMATAÇÃO FINAL
    # =====================================================

    fig = apply_brazilian_format(
        fig,
        metric
    )

    # =====================================================
    # ORDENAÇÃO
    # =====================================================

    if category_orders:

        fig.update_xaxes(

            categoryorder='array',

            categoryarray=
            category_orders.get(x, [])
        )

    else:

        fig.update_xaxes(

            categoryorder='total descending'
        )

    return fig


# ==========================================================
# GRÁFICO DE LINHA
# ==========================================================

def line_chart(
    data: pd.DataFrame,
    x: str,
    metric: str,
    color: str = None,
    title: str = None
):
    """
    Descrição:
        Cria gráfico de linha padronizado.
    """

    # ======================================================
    # QUANDO EXISTE COLOR
    # ======================================================

    if color:

        plot_data_list = []

        for color_value in data[color].dropna().unique():

            subset = (

                data[
                    data[color] == color_value
                ]
            )

            prepared = prepare_metric_data(

                data=subset,

                group_col=x,

                metric=metric
            )

            prepared[color] = color_value

            plot_data_list.append(
                prepared
            )

        plot_data = pd.concat(

            plot_data_list,

            ignore_index=True
        )

    # ======================================================
    # SEM COLOR
    # ======================================================

    else:

        plot_data = prepare_metric_data(

            data=data,

            group_col=x,

            metric=metric
        )

    # ======================================================
    # ORDENAÇÃO TEMPORAL
    # ======================================================

    if 'dia_ordem' in plot_data.columns:

        plot_data = plot_data.sort_values(
            by='dia_ordem'
        )

    else:

        plot_data = plot_data.sort_values(
            by=x
        )

    # ======================================================
    # FIGURA
    # ======================================================

    fig = px.line(

        plot_data,

        x=x,

        y='valor',

        color=color,

        markers=False
    )

    # ======================================================
    # REMOVE TÍTULO
    # ======================================================

    fig = remove_plotly_title(
        fig
    )

    # ======================================================
    # LAYOUT
    # ======================================================

    fig.update_layout(

        plot_bgcolor='rgba(0,0,0,0)',

        paper_bgcolor='rgba(0,0,0,0)',

        hovermode='x unified'
    )

    # ======================================================
    # LINHAS
    # ======================================================

    fig.update_traces(

        line=dict(
            width=3
        )
    )

    # ======================================================
    # FORMATAÇÃO
    # ======================================================

    fig = apply_brazilian_format(
        fig,
        metric
    )

    return fig