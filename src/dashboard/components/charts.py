"""
Descrição:
    Módulo responsável pela criação de gráficos interativos utilizando Plotly,
    fornecendo componentes reutilizáveis para visualização de dados no dashboard.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import plotly.express as px
import pandas as pd


def bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str
):
    """
    Descrição:
        Cria gráfico de barras interativo utilizando Plotly Express.

    Parâmetros:
        data (pd.DataFrame): DataFrame contendo os dados do gráfico.
        x (str): Nome da coluna utilizada no eixo X.
        y (str): Nome da coluna utilizada no eixo Y.
        title (str): Título do gráfico.

    Referências:
        - Plotly Technologies Inc. (2024). Plotly Express Documentation.
    """

    fig = px.bar(
        data,
        x=x,
        y=y,
        title=title
    )

    return fig


def line_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str
):
    """
    Descrição:
        Cria gráfico de linha interativo utilizando Plotly Express.

    Parâmetros:
        data (pd.DataFrame): DataFrame contendo os dados do gráfico.
        x (str): Nome da coluna utilizada no eixo X.
        y (str): Nome da coluna utilizada no eixo Y.
        title (str): Título do gráfico.

    Referências:
        - Plotly Technologies Inc. (2024). Plotly Express Documentation.
    """

    fig = px.line(
        data,
        x=x,
        y=y,
        title=title
    )

    return fig