"""
Descrição:
    Módulo responsável pela criação de componentes de métricas para exibição
    de indicadores no dashboard Streamlit.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st


def metric_card(
    title: str,
    value
) -> None:
    """
    Descrição:
        Renderiza um card de métrica utilizando componente nativo do Streamlit.

    Parâmetros:
        title (str): Título da métrica exibida.
        value: Valor principal apresentado no card.

    Referências:
        - Streamlit Inc. (2024). Streamlit Documentation.
    """

    st.metric(
        label=title,
        value=value
    )