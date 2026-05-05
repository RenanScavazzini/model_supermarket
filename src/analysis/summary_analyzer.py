"""
Descrição:
    Módulo responsável pelo cálculo de métricas descritivas e agregações sobre dados
    de notas fiscais de supermercado. Inclui indicadores de gasto, análises por
    dimensão e geração de relatório consolidado.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 29/04/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SummaryReport:
    """
    Descrição:
        Estrutura de dados que representa um relatório consolidado contendo métricas
        descritivas e agrupamentos derivados dos dados de notas fiscais.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    total_gasto: float
    total_notas: int
    gasto_medio: float
    menos_gasto: float
    maior_gasto: float
    agrupamento: Dict[str, pd.DataFrame]


class SummaryAnalyzer:
    """
    Descrição:
        Classe responsável pelo cálculo de indicadores descritivos e agregações
        estatísticas sobre dados de supermercado.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    @staticmethod
    def total_geral(df: pd.DataFrame) -> float:
        """
        Descrição:
            Calcula o valor total gasto considerando toda a base de dados.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados de notas fiscais.

        Referências:
            ---
        """
        return float(df['preco_total'].sum())

    @staticmethod
    def total_por_periodo(df: pd.DataFrame) -> pd.DataFrame:
        """
        Descrição:
            Realiza a agregação do valor total gasto por período do dia.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """
        return df.groupby('periodo_dia', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_supermercado(df: pd.DataFrame) -> pd.DataFrame:
        """
        Descrição:
            Realiza a agregação do valor total gasto por supermercado.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """
        return df.groupby('supermercado', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_produto(df: pd.DataFrame) -> pd.DataFrame:
        """
        Descrição:
            Realiza a agregação do valor total gasto por produto.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """
        return df.groupby('produto', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_codigo(df: pd.DataFrame) -> pd.DataFrame:
        """
        Descrição:
            Realiza a agregação do valor total gasto por código de produto.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """
        return df.groupby('codigo_produto', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_periodo_temporal(df: pd.DataFrame, periodo: str) -> float:
        """
        Descrição:
            Calcula o valor total gasto filtrando por um período específico do dia.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.
            periodo (str): Período do dia a ser filtrado.

        Referências:
            ---
        """
        return float(df.loc[df['periodo_dia'] == periodo, 'preco_total'].sum())

    @staticmethod
    def gasto_medio_por_nota(df: pd.DataFrame) -> float:
        """
        Descrição:
            Calcula o gasto médio por nota fiscal a partir da agregação por identificador.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            - Freedman, D. et al. (2007). Statistics.
        """
        invoice_totals = df.groupby('nota_fiscal_id')['preco_total'].sum()
        return float(invoice_totals.mean())

    @staticmethod
    def nota_mais_cara(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Descrição:
            Identifica a nota fiscal com maior valor de gasto.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            ---
        """
        invoice_totals = df.groupby('nota_fiscal_id')['preco_total'].sum()
        nota_id = invoice_totals.idxmax()
        return {'nota_fiscal_id': nota_id, 'preco_total': float(invoice_totals.max())}

    @staticmethod
    def nota_mais_barata(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Descrição:
            Identifica a nota fiscal com menor valor de gasto.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            ---
        """
        invoice_totals = df.groupby('nota_fiscal_id')['preco_total'].sum()
        nota_id = invoice_totals.idxmin()
        return {'nota_fiscal_id': nota_id, 'preco_total': float(invoice_totals.min())}

    @staticmethod
    def gerenciar_resumo(df: pd.DataFrame) -> SummaryReport:
        """
        Descrição:
            Gera um relatório consolidado contendo métricas principais e agrupamentos.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.

        Referências:
            - Tukey, J. (1977). Exploratory Data Analysis.
        """
        total = SummaryAnalyzer.total_geral(df)
        total_notas = df['nota_fiscal_id'].nunique()
        gasto_medio = SummaryAnalyzer.gasto_medio_por_nota(df)
        nota_barata = SummaryAnalyzer.nota_mais_barata(df)['preco_total']
        nota_cara = SummaryAnalyzer.nota_mais_cara(df)['preco_total']

        agrupamento = {
            'por_periodo': SummaryAnalyzer.total_por_periodo(df),
            'por_supermercado': SummaryAnalyzer.total_por_supermercado(df),
            'por_produto': SummaryAnalyzer.total_por_produto(df),
            'por_codigo': SummaryAnalyzer.total_por_codigo(df),
        }

        return SummaryReport(
            total_gasto=total,
            total_notas=total_notas,
            gasto_medio=gasto_medio,
            menos_gasto=nota_barata,
            maior_gasto=nota_cara,
            agrupamento=agrupamento,
        )