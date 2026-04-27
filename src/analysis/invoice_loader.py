import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class InvoiceLoader:
    """Carrega e normaliza dados de notas fiscais de supermercado."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def load(self, path: str) -> pd.DataFrame:
        """Carrega os dados a partir de CSV, Excel ou JSON."""
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        if suffix in ['.csv']:
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif suffix in ['.json']:
            df = pd.read_json(path, orient='records', encoding='utf-8')
        else:
            raise ValueError(f'Formato de arquivo não suportado: {suffix}')

        return self._normalize(df)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza colunas e adiciona informações de tempo."""
        df = df.copy()

        column_map = {
            self.config.get('data.invoice_id_column', 'nota_fiscal_id'): 'nota_fiscal_id',
            self.config.get('data.store_column', 'supermercado'): 'supermercado',
            self.config.get('data.product_code_column', 'codigo_produto'): 'codigo_produto',
            self.config.get('data.product_name_column', 'produto'): 'produto',
            self.config.get('data.quantity_column', 'quantidade'): 'quantidade',
            self.config.get('data.unit_price_column', 'preco_unitario'): 'preco_unitario',
            self.config.get('data.total_price_column', 'preco_total'): 'preco_total',
            self.config.get('data.date_column', 'data_hora'): 'data_hora',
        }

        for source, target in column_map.items():
            if source in df.columns:
                df = df.rename(columns={source: target})

        if 'preco_total' not in df.columns and 'quantidade' in df.columns and 'preco_unitario' in df.columns:
            df['preco_total'] = df['quantidade'] * df['preco_unitario']

        if 'data_hora' in df.columns:
            df['data_hora'] = pd.to_datetime(df['data_hora'], errors='coerce')
        else:
            df['data_hora'] = pd.NaT

        df['data'] = df['data_hora'].dt.date
        df['ano'] = df['data_hora'].dt.year
        df['mes'] = df['data_hora'].dt.month
        df['dia'] = df['data_hora'].dt.day
        df['hora'] = df['data_hora'].dt.hour
        df['periodo_dia'] = df['hora'].apply(self._classify_periodo)
        df['preco_total'] = pd.to_numeric(df['preco_total'], errors='coerce').fillna(0.0)
        df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0).astype(int)

        return df

    @staticmethod
    def _classify_periodo(hour: Optional[int]) -> str:
        """Classifica o período do dia a partir da hora."""
        if hour is None or np.isnan(hour):
            return 'Desconhecido'
        if 0 <= hour < 12:
            return 'Manhã'
        if 12 <= hour < 18:
            return 'Tarde'
        return 'Noite'

    def load_and_prepare(self, path: str) -> pd.DataFrame:
        """Carrega e prepara os dados para análise."""
        df = self.load(path)
        return df
