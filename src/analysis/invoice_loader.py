"""
Descrição:
    Módulo responsável pelo carregamento, normalização e enriquecimento de dados de notas fiscais
    de supermercado. Suporta múltiplos formatos de entrada (CSV, Excel, JSON), padroniza nomes de
    colunas, trata inconsistências, realiza feature engineering temporal e integra dados de QR Codes
    para enriquecimento analítico.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 29/04/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs


class InvoiceLoader:
    """
    Descrição:
        Classe responsável pelo carregamento e normalização de dados de notas fiscais.
        Implementa leitura de múltiplos formatos, padronização de schema e enriquecimento
        temporal, garantindo consistência para análises posteriores.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    def __init__(self, config: dict = None):
        """
        Descrição:
            Inicializa a classe com configurações opcionais para controle do comportamento
            de leitura e normalização.

        Parâmetros:
            config (dict): Configurações customizadas para o processamento dos dados.

        Referências:
            ---
        """
        self.config = config or {}

    def _get_config_value(self, key: str, default=None):
        """
        Descrição:
            Recupera valores de configuração utilizando notação hierárquica (dot notation),
            permitindo acesso a estruturas aninhadas.

        Parâmetros:
            key (str): Chave de acesso no formato hierárquico.
            default: Valor padrão caso a chave não exista.

        Referências:
            - Fowler, M. (2018). Patterns of Enterprise Application Architecture.
        """
        value = self.config
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part, default)
            else:
                return default
        return value if value is not None else default

    def load(self, path: str) -> pd.DataFrame:
        """
        Descrição:
            Realiza o carregamento de dados a partir de diferentes formatos de arquivo e
            encaminha para normalização.

        Parâmetros:
            path (str): Caminho do arquivo de entrada.

        Referências:
            - McKinney, W. (2018). Python for Data Analysis.
        """
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
        """
        Descrição:
            Padroniza colunas, trata inconsistências, cria variáveis derivadas e aplica
            enriquecimento temporal para análise.

        Parâmetros:
            df (pd.DataFrame): DataFrame bruto carregado.

        Referências:
            - Wickham, H. (2014). Tidy Data.
            - Han, J. et al. (2011). Data Mining: Concepts and Techniques.
        """
        df = df.copy()

        column_map = {
            self._get_config_value('data.invoice_id_column', 'nota_fiscal_id'): 'nota_fiscal_id',
            self._get_config_value('data.store_column', 'supermercado'): 'supermercado',
            self._get_config_value('data.product_code_column', 'codigo_produto'): 'codigo_produto',
            self._get_config_value('data.product_name_column', 'produto'): 'produto',
            self._get_config_value('data.quantity_column', 'quantidade'): 'quantidade',
            self._get_config_value('data.unit_price_column', 'preco_unitario'): 'preco_unitario',
            self._get_config_value('data.total_price_column', 'preco_total'): 'preco_total',
            self._get_config_value('data.date_column', 'data_hora'): 'data_hora',
            self._get_config_value('data.period_column', 'periodo'): 'periodo_dia',
        }

        for source, target in column_map.items():
            if source in df.columns:
                df = df.rename(columns={source: target})

        if 'preco_total' not in df.columns:
            if 'quantidade' in df.columns and 'preco_unitario' in df.columns:
                df['preco_total'] = pd.to_numeric(df['quantidade'], errors='coerce') * pd.to_numeric(df['preco_unitario'], errors='coerce')
            else:
                df['preco_total'] = 0.0

        if 'data_hora' in df.columns:
            df['data_hora'] = pd.to_datetime(df['data_hora'], format='%d/%m/%Y', errors='coerce')
        else:
            df['data_hora'] = pd.NaT

        df['data'] = df['data_hora'].dt.date
        df['ano'] = df['data_hora'].dt.year
        df['mes'] = df['data_hora'].dt.month
        df['dia'] = df['data_hora'].dt.day

        if 'periodo_dia' in df.columns:
            df['periodo_dia'] = df['periodo_dia'].fillna('Tarde')
            df['hora'] = df['periodo_dia'].apply(self._get_hour_from_periodo)
        else:
            df['periodo_dia'] = 'Tarde'
            df['hora'] = 15

        df['data_hora'] = df['data_hora'] + pd.to_timedelta(df['hora'].astype(int), unit='h')

        nat_mask = df['data_hora'].isna()
        if nat_mask.any():
            df.loc[nat_mask, 'data_hora'] = pd.Timestamp('1900-01-01 00:00:00')

        if 'periodo_dia' not in df.columns:
            df['periodo_dia'] = df['hora'].apply(self._classify_periodo)
        else:
            df['periodo_dia'] = df['periodo_dia'].fillna('Desconhecido')

        if 'quantidade' not in df.columns:
            df['quantidade'] = 0
        if 'preco_unitario' not in df.columns:
            df['preco_unitario'] = 0.0

        df['preco_total'] = pd.to_numeric(df['preco_total'], errors='coerce').fillna(0.0)
        df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0).astype(int)

        return df

    @staticmethod
    def _get_hour_from_periodo(periodo: str) -> int:
        """
        Descrição:
            Converte o período do dia em um valor representativo de hora.

        Parâmetros:
            periodo (str): Período textual (manhã, tarde, noite).

        Referências:
            ---
        """
        if isinstance(periodo, str):
            periodo = periodo.strip().lower()
            if 'manh' in periodo:
                return 9
            elif 'tard' in periodo:
                return 15
            elif 'noit' in periodo:
                return 20
        return 15

    @staticmethod
    def _classify_periodo(hour: Optional[int]) -> str:
        """
        Descrição:
            Classifica o período do dia com base na hora.

        Parâmetros:
            hour (int): Hora do dia.

        Referências:
            ---
        """
        if hour is None or np.isnan(hour):
            return 'Desconhecido'
        if 0 <= hour < 12:
            return 'Manhã'
        if 12 <= hour < 18:
            return 'Tarde'
        return 'Noite'

    def _parse_qrcode_url(self, url: str) -> Optional[str]:
        """
        Descrição:
            Extrai a chave da nota fiscal (NFC-e) a partir de uma URL de QR Code.

        Parâmetros:
            url (str): URL do QR Code.

        Referências:
            - Fielding, R. (2000). REST Architectural Style.
        """
        if not isinstance(url, str) or not url.strip():
            return None
        parsed = urlparse(url.strip())
        query = parse_qs(parsed.query)
        p_values = query.get('p') or query.get('P')
        if not p_values:
            return None
        p_value = p_values[0]
        return p_value.split('|', 1)[0] if '|' in p_value else p_value

    def _load_qrcodes(self, path: str) -> pd.DataFrame:
        """
        Descrição:
            Carrega URLs de QR Codes e extrai suas respectivas chaves.

        Parâmetros:
            path (str): Caminho do arquivo contendo URLs.

        Referências:
            ---
        """
        path_obj = Path(path)
        if not path_obj.exists():
            return pd.DataFrame(columns=['nfce_chave', 'qrcode_url'])

        lines = [line.strip() for line in path_obj.read_text(encoding='utf-8').splitlines() if line.strip()]
        data = []
        for line in lines:
            nfce_chave = self._parse_qrcode_url(line)
            if nfce_chave:
                data.append({'nfce_chave': nfce_chave, 'qrcode_url': line})

        return pd.DataFrame(data)

    def _load_qrcode_mapping(self, path: str) -> pd.DataFrame:
        """
        Descrição:
            Carrega arquivo de mapeamento entre chaves reais e anonimizada.

        Parâmetros:
            path (str): Caminho do arquivo de mapeamento.

        Referências:
            - Elmasri, R., & Navathe, S. (2015). Fundamentals of Database Systems.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            return pd.DataFrame()

        suffix = path_obj.suffix.lower()
        if suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        if suffix == '.csv':
            return pd.read_csv(path, encoding='utf-8')
        raise ValueError(f'Formato de arquivo de mapeamento não suportado: {suffix}')

    def merge_qrcodes(self, df: pd.DataFrame, qrcodes_path: str, mapping_path: Optional[str] = None) -> pd.DataFrame:
        """
        Descrição:
            Realiza o merge entre dados de notas fiscais e QR Codes, com suporte a anonimização.

        Parâmetros:
            df (pd.DataFrame): DataFrame principal.
            qrcodes_path (str): Caminho do arquivo de QR Codes.
            mapping_path (str): Caminho do arquivo de mapeamento (opcional).

        Referências:
            - Kimball, R. (2013). Data Warehouse Toolkit.
        """
        df_out = df.copy()
        qrcodes = self._load_qrcodes(qrcodes_path)
        if qrcodes.empty:
            return df_out

        if mapping_path:
            mapping = self._load_qrcode_mapping(mapping_path)
            if {'nfce_chave', 'nfce_chave_anon'}.issubset(mapping.columns):
                qrcodes = qrcodes.merge(mapping[['nfce_chave', 'nfce_chave_anon']], on='nfce_chave', how='left')
                qrcodes = qrcodes.dropna(subset=['nfce_chave_anon'])
                qrcodes = qrcodes.groupby('nfce_chave_anon', as_index=False).agg(
                    qrcode_url=('qrcode_url', lambda urls: ';'.join(urls)),
                    qrcode_count=('qrcode_url', 'size')
                ).rename(columns={'nfce_chave_anon': 'nota_fiscal_id'})
            else:
                return df_out
        else:
            qrcodes = qrcodes.groupby('nfce_chave', as_index=False).agg(
                qrcode_url=('qrcode_url', lambda urls: ';'.join(urls)),
                qrcode_count=('qrcode_url', 'size')
            ).rename(columns={'nfce_chave': 'nota_fiscal_id'})

        if 'nota_fiscal_id' not in df_out.columns:
            return df_out

        return df_out.merge(qrcodes, on='nota_fiscal_id', how='left')

    def load_and_prepare(self, path: str) -> pd.DataFrame:
        """
        Descrição:
            Pipeline simplificado para carregamento e preparação dos dados.

        Parâmetros:
            path (str): Caminho do arquivo de entrada.

        Referências:
            ---
        """
        df = self.load(path)
        return df