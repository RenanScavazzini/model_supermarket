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
    4.0 - 13/05/2026 - Compatibilidade com versões antigas do Plotly.

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


def bar_chart(
    data,
    x,
    y,
    title=None
):
    """
    Descrição:
        Cria gráfico de barras padronizado.

    Parâmetros:
        data: DataFrame.
        x: Coluna eixo X.
        y: Coluna eixo Y.

    Retorno:
        Figura Plotly.
    """

    fig = px.bar(

        data,

        x=x,

        y=y,

        text_auto='.2s'
    )

    # =====================================================
    # REMOVE TÍTULO COMPLETAMENTE
    # =====================================================

    fig.layout.title = None

    # =====================================================
    # LAYOUT
    # =====================================================

    fig.update_layout(

        plot_bgcolor='rgba(0,0,0,0)',

        paper_bgcolor='rgba(0,0,0,0)',

        margin=dict(
            t=60
        )
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

    Parâmetros:
        data: DataFrame.
        x: Coluna eixo X.
        y: Coluna eixo Y.
        color: Agrupamento opcional.

    Retorno:
        Figura Plotly.
    """

    fig = px.line(

        data,

        x=x,

        y=y,

        color=color
    )

    # =====================================================
    # REMOVE TÍTULO COMPLETAMENTE
    # =====================================================

    fig.layout.title = None

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