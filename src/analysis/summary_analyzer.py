import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SummaryReport:
    total_gasto: float
    total_notas: int
    gasto_medio: float
    menos_gasto: float
    maior_gasto: float
    agrupamento: Dict[str, pd.DataFrame]


class SummaryAnalyzer:
    """Calcula os principais indicadores e agregações do supermercado."""

    @staticmethod
    def total_geral(df: pd.DataFrame) -> float:
        """Retorna o gasto total da base."""
        return float(df['preco_total'].sum())

    @staticmethod
    def total_por_periodo(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega o total por período do dia."""
        return df.groupby('periodo_dia', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_supermercado(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega o total por supermercado."""
        return df.groupby('supermercado', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_produto(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega o total por produto."""
        return df.groupby('produto', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_codigo(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega o total por código de produto."""
        return df.groupby('codigo_produto', as_index=False)['preco_total'].sum().sort_values('preco_total', ascending=False)

    @staticmethod
    def total_por_periodo_temporal(df: pd.DataFrame, periodo: str) -> float:
        """Retorna o gasto total em um período específico."""
        return float(df.loc[df['periodo_dia'] == periodo, 'preco_total'].sum())

    @staticmethod
    def gasto_medio_por_nota(df: pd.DataFrame) -> float:
        """Calcula o gasto médio por nota fiscal."""
        invoice_totals = df.groupby('nota_fiscal_id')['preco_total'].sum()
        return float(invoice_totals.mean())

    @staticmethod
    def nota_mais_cara(df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra a nota fiscal com maior gasto."""
        invoice_totals = df.groupby('nota_fiscal_id')['preco_total'].sum()
        nota_id = invoice_totals.idxmax()
        return {'nota_fiscal_id': nota_id, 'preco_total': float(invoice_totals.max())}

    @staticmethod
    def nota_mais_barata(df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra a nota fiscal com menor gasto."""
        invoice_totals = df.groupby('nota_fiscal_id')['preco_total'].sum()
        nota_id = invoice_totals.idxmin()
        return {'nota_fiscal_id': nota_id, 'preco_total': float(invoice_totals.min())}

    @staticmethod
    def gerenciar_resumo(df: pd.DataFrame) -> SummaryReport:
        """Gera um relatório completo com indicadores principais."""
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
