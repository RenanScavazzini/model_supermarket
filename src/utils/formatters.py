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
        Formata valor monetário no padrão brasileiro.
    """

    if value is None:

        return 'R$ 0,00'

    try:

        formatted = f"{float(value):,.2f}"

        formatted = (

            formatted

            .replace(',', 'X')

            .replace('.', ',')

            .replace('X', '.')
        )

        return f"R$ {formatted}"

    except Exception:

        return 'R$ 0,00'


def format_number(
    value: float,
    decimals: int = 2
) -> str:
    """
    Descrição:
        Formata números gerais no padrão brasileiro.
    """

    if value is None:

        value = 0

    try:

        formatted = f"{float(value):,.{decimals}f}"

        formatted = (

            formatted

            .replace(',', 'X')

            .replace('.', ',')

            .replace('X', '.')
        )

        return formatted

    except Exception:

        return '0'