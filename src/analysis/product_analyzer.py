"""
Descrição:
    Módulo responsável pelas análises relacionadas a produtos,
    incluindo busca textual, ranking de produtos mais comprados
    e histórico temporal de preços.

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


class ProductAnalyzer:
    """
    Descrição:
        Classe responsável pelas análises de produtos presentes
        nas notas fiscais de supermercado.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        """
        Descrição:
            Inicializa a classe de análise de produtos.

        Parâmetros:
            df (pd.DataFrame): DataFrame analítico.

        Referências:
            ---
        """

        self.logger = setup_logger(
            self.__class__.__name__
        )

        self.df = df

    def most_purchased_products(
        self,
        top_n: int = 10
    ) -> pd.Series:
        """
        Descrição:
            Retorna ranking dos produtos mais comprados.

        Parâmetros:
            top_n (int): Quantidade de produtos retornados.

        Referências:
            ---
        """

        result = (

            self.df

            .groupby('produto')

            .size()

            .sort_values(
                ascending=False
            )

            .head(top_n)
        )

        self.logger.info(
            f'Top {top_n} produtos gerado'
        )

        return result

    def search_product(
        self,
        product_name: str
    ) -> pd.DataFrame:
        """
        Descrição:
            Realiza busca textual de produtos.

        Parâmetros:
            product_name (str): Nome ou trecho do produto.

        Referências:
            ---
        """

        result = self.df[

            self.df['produto']

            .str.contains(
                product_name,
                case=False,
                na=False
            )

        ][[
            'produto',
            'preco_unitario',
            'preco_total',
            'data_hora',
            'supermercado'
        ]]

        result = result.sort_values(
            'data_hora'
        )

        self.logger.info(
            f'Busca realizada para: {product_name}'
        )

        return result

    def product_price_history(
        self,
        product_name: str
    ) -> pd.DataFrame:
        """
        Descrição:
            Retorna histórico temporal de preços do produto.

        Parâmetros:
            product_name (str): Nome do produto.

        Referências:
            ---
        """

        result = self.search_product(
            product_name
        )

        self.logger.info(
            f'Histórico de preços gerado para: {product_name}'
        )

        return result