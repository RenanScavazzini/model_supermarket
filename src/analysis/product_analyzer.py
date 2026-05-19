"""
Descrição:
    Módulo responsável pelas análises relacionadas a produtos,
    incluindo buscas, histórico de preços e resumos analíticos.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 11/05/2026
    1.1 - 12/05/2026 - Refatoração e melhorias de código
    2.0 - 12/05/2026 - Adição de novas funcionalidades e melhorias na lógica de análise dos produtos.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd

from src.core.logger import setup_logger


class ProductAnalyzer:
    """
    Descrição:
        Classe responsável pelas análises relacionadas
        aos produtos das notas fiscais.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        """
        Descrição:
            Inicializa classe de análise de produtos.

        Parâmetros:
            df (pd.DataFrame): DataFrame analítico.

        Referências:
            ---
        """

        self.logger = setup_logger(
            self.__class__.__name__
        )

        self.df = df

    def search_product(
        self,
        product_name: str
    ) -> pd.DataFrame:
        """
        Descrição:
            Realiza busca textual por nome de produto.

        Parâmetros:
            product_name (str): Nome pesquisado.

        Referências:
            ---
        """

        return self.df[

            self.df['produto']

            .str.contains(
                product_name,
                case=False,
                na=False
            )
        ]

    def search_product_code(
        self,
        code: int
    ) -> pd.DataFrame:
        """
        Descrição:
            Realiza busca por código do produto.

        Parâmetros:
            code (int): Código do produto.

        Referências:
            ---
        """

        return self.df[

            self.df[
                'cod_produto'
            ] == code
        ]

    def product_summary(
        self,
        df_product: pd.DataFrame
    ) -> dict:
        """
        Descrição:
            Retorna resumo analítico do(s) produto(s).

        Parâmetros:
            df_product (pd.DataFrame): Base filtrada.

        Referências:
            ---
        """

        if df_product.empty:

            return {}

        produtos = sorted(

            df_product[
                'produto'
            ]

            .astype(str)

            .unique()
        )

        codigos = sorted(

            df_product[
                'cod_produto'
            ]

            .astype(str)

            .unique()
        )

        categorias = sorted(

            df_product[
                'categoria_produto'
            ]

            .astype(str)

            .unique()
        )

        unidades = sorted(

            df_product[
                'unidade'
            ]

            .astype(str)

            .unique()
        )

        supermercados = sorted(

            df_product[
                'supermercado'
            ]

            .astype(str)

            .unique()
        )

        return {

            'Nome':

                ', '.join(
                    produtos
                ),

            'Código':

                ', '.join(
                    codigos
                ),

            'Categoria':

                ', '.join(
                    categorias
                ),

            'Unidade':

                ', '.join(
                    unidades
                ),

            'Total gasto':

                round(

                    float(

                        df_product[
                            'preco_total'
                        ].sum()
                    ),

                    2
                ),

            'Quantidade comprada':

                round(

                    float(

                        df_product[
                            'quantidade'
                        ].sum()
                    ),

                    2
                ),

            'Valor mínimo':

                round(

                    float(

                        df_product[
                            'preco_unitario'
                        ].min()
                    ),

                    2
                ),

            'Valor máximo':

                round(

                    float(

                        df_product[
                            'preco_unitario'
                        ].max()
                    ),

                    2
                ),

            'Valor médio':

                round(

                    float(

                        df_product[
                            'preco_total'
                        ].sum()

                        /

                        df_product[
                            'quantidade'
                        ].sum()
                    ),

                    2
                ),

            'Supermercados':

                ', '.join(
                    supermercados
                ),

            'Período mais comprado':

                str(

                    df_product[
                        'periodo_dia'
                    ].mode()[0]
                )
        }