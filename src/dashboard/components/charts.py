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

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import plotly.express as px


def apply_brazilian_format(
    fig
):
    """
    Descrição:
        Aplica formatação brasileira aos gráficos.

    Parâmetros:
        fig: Figura Plotly.

    Retorno:
        fig formatada.
    """

    fig.update_layout(

        yaxis=dict(

            tickformat=',.2f',

            separatethousands=True
        )
    )

    fig.update_traces(

        hovertemplate=
        '%{y:,.2f}<extra></extra>'
    )

    return fig


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


def bar_chart(
    data,
    x,
    y,
    title=None
):
    """
    Descrição:
        Cria gráfico de barras padronizado.
    """

    fig = px.bar(

        data,

        x=x,

        y=y,

        text_auto='.2s'
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

    fig = apply_brazilian_format(
        fig
    )

    return fig


def line_chart(
    data,
    x,
    y,
    title=None,
    color=None
):
    """
    Descrição:
        Cria gráfico de linha padronizado.
    """

    fig = px.line(

        data,

        x=x,

        y=y,

        color=color
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

    fig = apply_brazilian_format(
        fig
    )

    return fig