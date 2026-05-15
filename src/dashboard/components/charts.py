"""
Descrição:
    Módulo responsável pela criação de gráficos interativos utilizando Plotly,
    fornecendo componentes reutilizáveis para visualização de dados no dashboard.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026
    2.0 - 13/05/2026 - Refatoração e melhorias visuais.

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
    title
):
    """
    Descrição:
        Cria gráfico de barras padronizado.

    Parâmetros:
        data: DataFrame.
        x: Coluna eixo X.
        y: Coluna eixo Y.
        title: Título gráfico.

    Retorno:
        Figura Plotly.
    """

    fig = px.bar(

        data,

        x=x,

        y=y,

        title=title
    )

    fig.update_layout(

        plot_bgcolor='rgba(0,0,0,0)',

        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig = apply_brazilian_format(
        fig
    )

    return fig


def line_chart(
    data,
    x,
    y,
    title,
    color=None
):
    """
    Descrição:
        Cria gráfico de linha padronizado.

    Parâmetros:
        data: DataFrame.
        x: Coluna eixo X.
        y: Coluna eixo Y.
        title: Título gráfico.
        color: Agrupamento opcional.

    Retorno:
        Figura Plotly.
    """

    fig = px.line(

        data,

        x=x,

        y=y,

        color=color,

        title=title
    )

    fig.update_layout(

        plot_bgcolor='rgba(0,0,0,0)',

        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig = apply_brazilian_format(
        fig
    )

    return fig