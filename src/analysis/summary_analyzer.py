"""
Descrição:
    Módulo responsável pelo cálculo de métricas descritivas, agregações e geração
    de relatórios analíticos sobre dados de notas fiscais de supermercado.

    Implementa:
    - indicadores gerais de consumo
    - agregações por dimensão
    - análises temporais
    - rankings de produtos
    - métricas por supermercado
    - geração consolidada de relatório analítico

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 05/05/2026
    1.1 - 11/05/2026 - Adição de métricas por supermercado e geração de relatório consolidado

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd

from dataclasses import dataclass

from typing import Dict
from typing import Any

from src.core.logger import setup_logger


@dataclass
class SummaryReport:
    """
    Descrição:
        Estrutura de dados responsável por armazenar métricas consolidadas
        e agrupamentos analíticos derivados dos dados de notas fiscais.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        2.0 - 11/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    total_gasto: float
    total_notas: int
    gasto_medio: float
    menor_gasto: float
    maior_gasto: float
    agrupamentos: Dict[str, pd.DataFrame]


class SummaryAnalyzer:
    """
    Descrição:
        Classe responsável pelo cálculo de métricas descritivas,
        agregações estatísticas e geração de relatórios analíticos.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        2.0 - 11/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    logger = setup_logger(__name__)

    @staticmethod
    def _validate_columns(
        df: pd.DataFrame,
        required_columns: list[str]
    ) -> None:
        """
        Descrição:
            Valida a existência de colunas obrigatórias no DataFrame.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.
            required_columns (list[str]): Lista de colunas obrigatórias.

        Referências:
            ---
        """

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f'Colunas ausentes: {missing}'
            )

    @staticmethod
    def _aggregate_sum(
        df: pd.DataFrame,
        group_col: str,
        value_col: str = 'preco_total'
    ) -> pd.DataFrame:
        """
        Descrição:
            Realiza agregação padronizada utilizando soma.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.
            group_col (str): Coluna de agrupamento.
            value_col (str): Coluna de valor.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """

        SummaryAnalyzer._validate_columns(
            df,
            [group_col, value_col]
        )

        return (
            df
            .groupby(group_col, as_index=False)[value_col]
            .sum()
            .sort_values(
                value_col,
                ascending=False
            )
        )

    @staticmethod
    def total_geral(
        df: pd.DataFrame
    ) -> float:
        """
        Descrição:
            Calcula o valor total gasto.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        SummaryAnalyzer._validate_columns(
            df,
            ['preco_total']
        )

        return float(
            df['preco_total'].sum()
        )

    @staticmethod
    def total_por_periodo(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Agrega gastos por período do dia.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """

        return SummaryAnalyzer._aggregate_sum(
            df,
            'periodo_dia'
        )

    @staticmethod
    def total_por_supermercado(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Agrega gastos por supermercado.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """

        return SummaryAnalyzer._aggregate_sum(
            df,
            'supermercado'
        )

    @staticmethod
    def total_por_produto(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Agrega gastos por produto.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """

        return SummaryAnalyzer._aggregate_sum(
            df,
            'produto'
        )

    @staticmethod
    def total_por_codigo(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Agrega gastos por código de produto.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """

        return SummaryAnalyzer._aggregate_sum(
            df,
            'codigo_produto'
        )

    @staticmethod
    def total_por_categoria(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Agrega gastos por categoria de produto.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        return SummaryAnalyzer._aggregate_sum(
            df,
            'categoria_produto'
        )

    @staticmethod
    def total_por_mes(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Agrega gastos por mês.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        SummaryAnalyzer._validate_columns(
            df,
            ['data_hora', 'preco_total']
        )

        temp = df.copy()

        temp['mes_ano'] = (
            temp['data_hora']
            .dt.to_period('M')
            .astype(str)
        )

        return SummaryAnalyzer._aggregate_sum(
            temp,
            'mes_ano'
        )

    @staticmethod
    def total_por_ano(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Agrega gastos por ano.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        SummaryAnalyzer._validate_columns(
            df,
            ['data_hora', 'preco_total']
        )

        temp = df.copy()

        temp['ano'] = (
            temp['data_hora']
            .dt.year
        )

        return SummaryAnalyzer._aggregate_sum(
            temp,
            'ano'
        )

    @staticmethod
    def gasto_medio_por_nota(
        df: pd.DataFrame
    ) -> float:
        """
        Descrição:
            Calcula gasto médio por nota fiscal.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            - Freedman, D. et al. (2007). Statistics.
        """

        SummaryAnalyzer._validate_columns(
            df,
            ['nota_fiscal_id', 'preco_total']
        )

        invoice_totals = (
            df
            .groupby('nota_fiscal_id')['preco_total']
            .sum()
        )

        if invoice_totals.empty:

            return 0.0

        return float(
            invoice_totals.mean()
        )

    @staticmethod
    def nota_mais_cara(
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Descrição:
            Identifica a nota fiscal com maior gasto.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        SummaryAnalyzer._validate_columns(
            df,
            ['nota_fiscal_id', 'preco_total']
        )

        invoice_totals = (
            df
            .groupby('nota_fiscal_id')['preco_total']
            .sum()
        )

        if invoice_totals.empty:

            return {
                'nota_fiscal_id': None,
                'preco_total': 0.0
            }

        nota_id = invoice_totals.idxmax()

        return {
            'nota_fiscal_id': nota_id,
            'preco_total': float(
                invoice_totals.max()
            )
        }

    @staticmethod
    def nota_mais_barata(
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Descrição:
            Identifica a nota fiscal com menor gasto.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        SummaryAnalyzer._validate_columns(
            df,
            ['nota_fiscal_id', 'preco_total']
        )

        invoice_totals = (
            df
            .groupby('nota_fiscal_id')['preco_total']
            .sum()
        )

        if invoice_totals.empty:

            return {
                'nota_fiscal_id': None,
                'preco_total': 0.0
            }

        nota_id = invoice_totals.idxmin()

        return {
            'nota_fiscal_id': nota_id,
            'preco_total': float(
                invoice_totals.min()
            )
        }

    @staticmethod
    def top_produtos(
        df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Descrição:
            Retorna os produtos com maior valor acumulado.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.
            top_n (int): Quantidade de produtos retornados.

        Referências:
            ---
        """

        result = SummaryAnalyzer.total_por_produto(df)

        return result.head(top_n)

    @staticmethod
    def ticket_medio_por_supermercado(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Calcula ticket médio por supermercado.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        SummaryAnalyzer._validate_columns(
            df,
            [
                'supermercado',
                'nota_fiscal_id',
                'preco_total'
            ]
        )

        grouped = (
            df
            .groupby(
                ['supermercado', 'nota_fiscal_id']
            )['preco_total']
            .sum()
            .reset_index()
        )

        result = (
            grouped
            .groupby('supermercado', as_index=False)['preco_total']
            .mean()
            .rename(
                columns={
                    'preco_total': 'ticket_medio'
                }
            )
            .sort_values(
                'ticket_medio',
                ascending=False
            )
        )

        return result

    @staticmethod
    def build_summary_report(
        df: pd.DataFrame
    ) -> SummaryReport:
        """
        Descrição:
            Gera relatório consolidado contendo métricas
            principais e agrupamentos analíticos.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """

        SummaryAnalyzer.logger.info(
            'Gerando relatório consolidado'
        )

        total = SummaryAnalyzer.total_geral(df)

        total_notas = (
            df['nota_fiscal_id']
            .nunique()
        )

        gasto_medio = (
            SummaryAnalyzer
            .gasto_medio_por_nota(df)
        )

        nota_barata = (
            SummaryAnalyzer
            .nota_mais_barata(df)['preco_total']
        )

        nota_cara = (
            SummaryAnalyzer
            .nota_mais_cara(df)['preco_total']
        )

        agrupamentos = {

            'por_periodo':
                SummaryAnalyzer.total_por_periodo(df),

            'por_supermercado':
                SummaryAnalyzer.total_por_supermercado(df),

            'por_produto':
                SummaryAnalyzer.total_por_produto(df),

            'por_codigo':
                SummaryAnalyzer.total_por_codigo(df),

            'por_categoria':
                SummaryAnalyzer.total_por_categoria(df),

            'por_mes':
                SummaryAnalyzer.total_por_mes(df),

            'por_ano':
                SummaryAnalyzer.total_por_ano(df),

            'ticket_medio_supermercado':
                SummaryAnalyzer.ticket_medio_por_supermercado(df),

            'top_produtos':
                SummaryAnalyzer.top_produtos(df)
        }

        return SummaryReport(
            total_gasto=total,
            total_notas=total_notas,
            gasto_medio=gasto_medio,
            menor_gasto=nota_barata,
            maior_gasto=nota_cara,
            agrupamentos=agrupamentos
        )