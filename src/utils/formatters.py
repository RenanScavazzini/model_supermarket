"""
Descrição:
    Módulo responsável pela padronização de formatação
    numérica e monetária da aplicação.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

def format_currency(
    value: float
) -> str:
    """
    Descrição:
        Formata valor monetário no padrão brasileiro:
        milhar com ponto e decimal com vírgula.

    Parâmetros:
        value (float): Valor monetário.

    Retorno:
        str: Valor formatado.
    """

    formatted = f"{value:,.2f}"

    formatted = (
        formatted
        .replace(',', 'X')
        .replace('.', ',')
        .replace('X', '.')
    )

    return f"R$ {formatted}"


def format_number(
    value: float,
    decimals: int = 2
) -> str:
    """
    Descrição:
        Formata números gerais no padrão brasileiro:
        milhar com ponto e decimal com vírgula.

    Parâmetros:
        value (float): Valor numérico.
        decimals (int): Quantidade de casas decimais.

    Retorno:
        str: Valor formatado.
    """

    formatted = f"{value:,.{decimals}f}"

    formatted = (
        formatted
        .replace(',', 'X')
        .replace('.', ',')
        .replace('X', '.')
    )

    return formatted