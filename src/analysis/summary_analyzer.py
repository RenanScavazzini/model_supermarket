"""
Descrição:
    Módulo responsável pelo cálculo de métricas descritivas, agregações e geração
    de relatórios analíticos sobre dados de notas fiscais de supermercado.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 05/05/2026
    1.1 - 11/05/2026 - Adição de métricas por supermercado e geração de relatório consolidado
    2.0 - 12/05/2026 - Refatoração e melhorias de código, adição de novas métricas e melhorias na lógica de análise.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd

from src.core.logger import setup_logger


class SummaryAnalyzer:
    """
    Descrição:
        Classe responsável pela geração de métricas
        consolidadas e análises descritivas gerais.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        """
        Descrição:
            Inicializa classe de análise de resumo.

        Parâmetros:
            df (pd.DataFrame): DataFrame analítico.

        Referências:
            ---
        """

        self.logger = setup_logger(
            self.__class__.__name__
        )

        self.df = df

    def total_spent(
        self
    ) -> float:
        """
        Descrição:
            Calcula valor total gasto.

        Parâmetros:
            ---

        Referências:
            ---
        """

        result = round(

            self.df[
                'preco_total'
            ].sum(),

            2
        )

        self.logger.info(
            'Total gasto calculado'
        )

        return result

    def total_invoices(
        self
    ) -> int:
        """
        Descrição:
            Calcula quantidade de notas fiscais distintas
            utilizando chave_anonimizada.

        Parâmetros:
            ---

        Referências:
            ---
        """

        if 'chave_anonimizada' in self.df.columns:

            result = (

                self.df[
                    'chave_anonimizada'
                ]

                .dropna()

                .nunique()
            )

        elif 'nota_fiscal_id' in self.df.columns:

            result = (

                self.df[
                    'nota_fiscal_id'
                ]

                .dropna()

                .nunique()
            )

        else:

            result = len(self.df)

        self.logger.info(
            f'Quantidade de notas calculada: {result}'
        )

        return int(result)


    def avg_ticket(
        self
    ) -> float:
        """
        Descrição:
            Calcula ticket médio das compras.

        Parâmetros:
            ---

        Referências:
            ---
        """

        invoices = self.total_invoices()

        if invoices == 0:

            return 0.0

        result = round(

            self.total_spent()

            / invoices,

            2
        )

        self.logger.info(
            'Ticket médio calculado'
        )

        return result

    def total_products(
        self
    ) -> int:
        """
        Descrição:
            Calcula quantidade de produtos distintos.

        Parâmetros:
            ---

        Referências:
            ---
        """

        result = (

            self.df[
                'produto'
            ]

            .nunique()
        )

        self.logger.info(
            'Quantidade de produtos calculada'
        )

        return result

    def spending_by_market(
        self
    ) -> pd.DataFrame:
        """
        Descrição:
            Calcula gastos agregados por supermercado.

        Parâmetros:
            ---

        Referências:
            ---
        """

        result = (

            self.df

            .groupby(
                'supermercado'
            )['preco_total']

            .sum()

            .reset_index()

            .sort_values(
                'preco_total',
                ascending=False
            )
        )

        self.logger.info(
            'Resumo por supermercado calculado'
        )

        return result

    def spending_by_period(
        self
    ) -> pd.DataFrame:
        """
        Descrição:
            Calcula gastos agregados por período do dia.

        Parâmetros:
            ---

        Referências:
            ---
        """

        result = (

            self.df

            .groupby(
                'periodo_dia'
            )['preco_total']

            .sum()

            .reset_index()

            .sort_values(
                'preco_total',
                ascending=False
            )
        )

        self.logger.info(
            'Resumo por período calculado'
        )

        return result

    def spending_by_category(
        self
    ) -> pd.DataFrame:
        """
        Descrição:
            Calcula gastos agregados por categoria de produto.

        Parâmetros:
            ---

        Referências:
            ---
        """

        result = (

            self.df

            .groupby(
                'categoria_produto'
            )['preco_total']

            .sum()

            .reset_index()

            .sort_values(
                'preco_total',
                ascending=False
            )
        )

        self.logger.info(
            'Resumo por categoria calculado'
        )

        return result