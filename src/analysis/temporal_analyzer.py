"""
Descrição:
    Módulo responsável pelas análises temporais relacionadas
    às notas fiscais e comportamento de consumo ao longo do tempo.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 11/05/2026
    1.1 - 12/05/2026 - Refatoração e melhorias de código

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd

from src.core.logger import setup_logger


class TemporalAnalyzer:
    """
    Descrição:
        Classe responsável pelas análises temporais da base
        de notas fiscais.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        """
        Descrição:
            Inicializa a classe de análise temporal.

        Parâmetros:
            df (pd.DataFrame): DataFrame analítico.

        Referências:
            ---
        """

        self.logger = setup_logger(
            self.__class__.__name__
        )

        self.df = df

    def monthly_spending(
        self
    ) -> pd.DataFrame:
        """
        Descrição:
            Calcula gastos mensais agregados.

        Parâmetros:
            ---

        Referências:
            ---
        """

        temp = self.df.copy()

        temp['mes'] = (

            temp['data_hora']

            .dt.to_period('M')

            .astype(str)
        )

        result = (

            temp

            .groupby('mes')['preco_total']

            .sum()

            .reset_index()
        )

        self.logger.info(
            'Análise mensal gerada'
        )

        return result

    def yearly_spending(
        self
    ) -> pd.DataFrame:
        """
        Descrição:
            Calcula gastos anuais agregados.

        Parâmetros:
            ---

        Referências:
            ---
        """

        temp = self.df.copy()

        temp['ano'] = (
            temp['data_hora']
            .dt.year
        )

        result = (

            temp

            .groupby('ano')['preco_total']

            .sum()

            .reset_index()
        )

        self.logger.info(
            'Análise anual gerada'
        )

        return result