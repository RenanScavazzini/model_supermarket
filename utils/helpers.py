"""
Descrição:
    Módulo contendo funções utilitárias auxiliares utilizadas
    em diferentes componentes do projeto.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd


def format_currency(
    value: float
) -> str:
    """
    Descrição:
        Formata valor monetário em padrão brasileiro.

    Parâmetros:
        value (float): Valor monetário.

    Referências:
        ---
    """

    return f'R$ {value:,.2f}'


def safe_division(
    numerator: float,
    denominator: float
) -> float:
    """
    Descrição:
        Realiza divisão segura evitando divisão por zero.

    Parâmetros:
        numerator (float): Numerador.
        denominator (float): Denominador.

    Referências:
        ---
    """

    if denominator == 0:

        return 0.0

    return numerator / denominator


def dataframe_memory_mb(
    df: pd.DataFrame
) -> float:
    """
    Descrição:
        Calcula uso de memória do DataFrame em megabytes.

    Parâmetros:
        df (pd.DataFrame): DataFrame analisado.

    Referências:
        ---
    """

    return round(
        (
            df.memory_usage(
                deep=True
            ).sum()
            / 1024**2
        ),
        2
    )