"""
Descrição:
    Módulo responsável pela criação de gráficos interativos utilizando Plotly,
    fornecendo componentes reutilizáveis para visualização de dados no dashboard.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026
    2.0 - 13/05/2026 - Adição de valores acima das barras.
    3.0 - 13/05/2026 - Remoção completa dos títulos internos dos gráficos.
    4.0 - 13/05/2026 - Compatibilidade mobile e Plotly antigo.
    5.0 - 19/05/2026 - Inclusão de métricas dinâmicas e agregações inteligentes.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import plotly.express as px

from src.utils.formatters import format_currency, format_number


# ==========================================================
# CONFIGURAÇÃO DAS MÉTRICAS
# ==========================================================

METRIC_CONFIG = {

    'preco_total': {

        'label': 'Preço Total',

        'is_currency': True
    },

    'valor_total_tributos': {

        'label': 'Valor Total Tributos',

        'is_currency': True
    },

    'quantidade': {

        'label': 'Quantidade de Registros',

        'is_currency': False
    },

    'qtd_notas': {

        'label': 'Quantidade de Notas',

        'is_currency': False
    }
}


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

    # ======================================================
    # MONETÁRIO
    # ======================================================

    if is_currency:

        fig.update_layout(

            yaxis=dict(

                tickprefix='R$ ',

                tickformat=',.2f',

                separatethousands=True
            )
        )

        fig.update_traces(

            hovertemplate=
            'R$ %{y:,.2f}<extra></extra>'
        )

    # ======================================================
    # QUANTITATIVO
    # ======================================================

    else:

        fig.update_layout(

            yaxis=dict(

                tickformat=',.0f',

                separatethousands=True
            )
        )

        fig.update_traces(

            hovertemplate=
            '%{y:,.0f}<extra></extra>'
        )

    return fig


# ==========================================================
# REMOVE TÍTULO
# ==========================================================

def remove_plotly_title(
    fig
):
    """
    Descrição:
        Remove completamente título interno do Plotly.
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
# PREPARAÇÃO DOS DADOS
# ==========================================================

def prepare_metric_data(
    data: pd.DataFrame,
    group_col: str,
    metric: str,
    category_orders: dict = None
) -> pd.DataFrame:
    """
    Descrição:
        Prepara dados agregados conforme métrica selecionada.
    """

    # ======================================================
    # PREÇO TOTAL
    # ======================================================

    if metric == 'preco_total':

        result = (

            data

            .groupby(group_col)[
                'preco_total'
            ]

            .sum()

            .reset_index()
        )

    # ======================================================
    # QUANTIDADE DE REGISTROS
    # ======================================================

    elif metric == 'quantidade':

        result = (

            data

            .groupby(group_col)

            .size()

            .reset_index(
                name='valor'
            )
        )

        return result

    # ======================================================
    # QUANTIDADE DE NOTAS
    # ======================================================

    elif metric == 'qtd_notas':

        result = (

            data

            .groupby(group_col)[
                'chave_anonimizada'
            ]

            .nunique()

            .reset_index()
        )

    # ======================================================
    # VALOR TOTAL TRIBUTOS
    # ======================================================

    elif metric == 'valor_total_tributos':

        dedup = (

            data[[
                group_col,
                'chave_anonimizada',
                'valor_total_tributos'
            ]]

            .drop_duplicates(
                subset=['chave_anonimizada']
            )
        )

        result = (

            dedup

            .groupby(group_col)[
                'valor_total_tributos'
            ]

            .sum()

            .reset_index()
        )

    # ======================================================
    # ERRO
    # ======================================================

    else:

        raise ValueError(
            f'Métrica inválida: {metric}'
        )

    # ======================================================
    # PADRONIZAÇÃO
    # ======================================================

    result.columns = [
        group_col,
        'valor'
    ]
    
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

        metric=metric,

        category_orders=category_orders
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

    if METRIC_CONFIG[metric]['is_currency']:

        plot_data['text_label'] = (

            plot_data['valor']

            .apply(format_currency)
        )

    else:

        text_template = '%{text:,.0f}'

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
    # ORDENAÇÃO DAS CATEGORIAS
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
    # AGRUPAMENTO
    # ======================================================

    group_cols = [x]

    if color:

        group_cols.append(
            color
        )

    # ======================================================
    # PREÇO TOTAL
    # ======================================================

    if metric == 'preco_total':

        plot_data = (

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
    # QUANTIDADE
    # ======================================================

    elif metric == 'quantidade':

        plot_data = (

            data

            .groupby(group_cols)

            .size()

            .reset_index(
                name='valor'
            )
        )

    # ======================================================
    # QUANTIDADE DE NOTAS
    # ======================================================

    elif metric == 'qtd_notas':

        plot_data = (

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
    # VALOR TOTAL TRIBUTOS
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

        plot_data = (

            dedup

            .groupby(group_cols)[
                'valor_total_tributos'
            ]

            .sum()

            .reset_index(
                name='valor'
            )
        )

    # ======================================================
    # ERRO
    # ======================================================

    else:

        raise ValueError(
            f'Métrica inválida: {metric}'
        )

    # ======================================================
    # ORDENAÇÃO TEMPORAL
    # ======================================================

    plot_data = plot_data.sort_values(
        by=x
    )

    # ======================================================
    # CRIA FIGURA
    # ======================================================

    fig = px.line(

        plot_data,

        x=x,

        y='valor',

        color=color,

        markers=True
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
    # ESPESSURA DAS LINHAS
    # ======================================================

    fig.update_traces(

        line=dict(
            width=3
        ),

        marker=dict(
            size=7
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