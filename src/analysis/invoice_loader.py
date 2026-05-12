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
    1.0 - 05/05/2026
    1.1 - 11/05/2026 - Adição de suporte a múltiplos formatos e configuração dinâmica
    1.2 - 12/05/2026 - Padronização automática de schema interno do dashboard

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import numpy as np

from pathlib import Path

from typing import Optional
from typing import Dict
from typing import Any

from src.core.logger import setup_logger


class InvoiceLoader:
    """
    Descrição:
        Classe responsável pelo carregamento, normalização e enriquecimento
        de dados de notas fiscais.
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None
    ):
        """
        Descrição:
            Inicializa o loader.

        Parâmetros:
            config (Dict[str, Any] | None): Configurações opcionais.

        Referências:
            ---
        """

        self.logger = setup_logger(
            self.__class__.__name__
        )

        self.config = config or {}

    def _get_config_value(
        self,
        key: str,
        default=None
    ):
        """
        Descrição:
            Recupera configuração hierárquica.

        Parâmetros:
            key (str): Chave hierárquica.
            default: Valor padrão.

        Referências:
            ---
        """

        value = self.config

        for part in key.split('.'):

            if isinstance(value, dict):

                value = value.get(
                    part,
                    default
                )

            else:

                return default

        return value if value is not None else default

    def _normalize_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Padroniza nomes de colunas para utilização interna
            da aplicação e dashboard.

        Parâmetros:
            df (pd.DataFrame): DataFrame original.

        Referências:
            ---
        """

        column_mapping = {

            'CHAVE_ANONIMIZADA': 'nota_fiscal_id',

            'DATA': 'data_hora',

            'MES_ANO': 'mes_ano',

            'PERIODO': 'periodo_dia',

            'CNPJ': 'cnpj',

            'SUPERMERCADO': 'supermercado',

            'QTDE_TOTAL_NOTA': 'qtde_total_nota',

            'VALOR_TOTAL_NOTA': 'valor_total_nota',

            'VALOR_TOTAL_TRIBUTOS': 'valor_total_tributos',

            'COD_PRODUTO': 'codigo_produto',

            'CAT_PRODUTO': 'categoria_produto',

            'PRODUTO': 'produto',

            'UNIDADE': 'unidade',

            'QTDE': 'quantidade',

            'VALOR_PRODUTO': 'preco_unitario',

            'VALOR_TOTAL_PRODUTO': 'preco_total'
        }

        existing_mapping = {

            source: target

            for source, target in column_mapping.items()

            if source in df.columns
        }

        df = df.rename(
            columns=existing_mapping
        )

        return df

    def _validate_schema(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        Descrição:
            Valida presença das colunas obrigatórias.

        Parâmetros:
            df (pd.DataFrame): DataFrame validado.

        Referências:
            ---
        """

        required_columns = [

            'produto',

            'preco_total',

            'supermercado',

            'data_hora'
        ]

        missing = [

            col

            for col in required_columns

            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f'Colunas obrigatórias ausentes: {missing}'
            )

    @staticmethod
    def _cached_read(
        path: str,
        suffix: str
    ) -> pd.DataFrame:
        """
        Descrição:
            Realiza leitura do arquivo de dados.

        Parâmetros:
            path (str): Caminho do arquivo.
            suffix (str): Extensão do arquivo.

        Referências:
            ---
        """

        if suffix == '.csv':

            return pd.read_csv(
                path,
                encoding='utf-8',
                low_memory=False
            )

        if suffix in ['.xlsx', '.xls']:

            return pd.read_excel(path)

        if suffix == '.json':

            return pd.read_json(
                path,
                orient='records'
            )

        raise ValueError(
            f'Formato não suportado: {suffix}'
        )

    def load(
        self,
        path: str
    ) -> pd.DataFrame:
        """
        Descrição:
            Carrega, valida e normaliza os dados.

        Parâmetros:
            path (str): Caminho do arquivo.

        Referências:
            ---
        """

        path_obj = Path(path)

        suffix = path_obj.suffix.lower()

        self.logger.info(
            f'Carregando arquivo: {path}'
        )

        df = self._cached_read(
            path,
            suffix
        )

        self.logger.info(
            f'DataFrame carregado com '
            f'{df.shape[0]} linhas e '
            f'{df.shape[1]} colunas'
        )

        df = self._normalize_columns(df)

        self._validate_schema(df)

        df = self._normalize(df)

        self.logger.info(
            'Normalização concluída com sucesso'
        )

        return df

    def _normalize(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Descrição:
            Realiza tratamento e enriquecimento dos dados.

        Parâmetros:
            df (pd.DataFrame): DataFrame normalizado.

        Referências:
            ---
        """

        df = df.copy()

        if 'preco_total' not in df.columns:

            if (
                'quantidade' in df.columns
                and
                'preco_unitario' in df.columns
            ):

                df['preco_total'] = (

                    pd.to_numeric(
                        df['quantidade'],
                        errors='coerce'
                    )

                    *

                    pd.to_numeric(
                        df['preco_unitario'],
                        errors='coerce'
                    )
                )

            else:

                df['preco_total'] = 0.0

        if 'data_hora' in df.columns:

            df['data_hora'] = pd.to_datetime(
                df['data_hora'],
                errors='coerce',
                dayfirst=True
            )

        else:

            df['data_hora'] = pd.NaT

        df['ano'] = df['data_hora'].dt.year

        df['mes'] = df['data_hora'].dt.month

        df['dia'] = df['data_hora'].dt.day

        df['hora'] = df['data_hora'].dt.hour

        if 'periodo_dia' not in df.columns:

            df['periodo_dia'] = (

                df['hora']

                .apply(
                    self._classify_periodo
                )
            )

        numeric_columns = [

            'quantidade',

            'preco_unitario',

            'preco_total',

            'valor_total_nota',

            'valor_total_tributos'
        ]

        for col in numeric_columns:

            if col in df.columns:

                df[col] = (

                    pd.to_numeric(
                        df[col],
                        errors='coerce'
                    )

                    .fillna(0)
                )

        categorical_columns = [

            'supermercado',

            'produto',

            'categoria_produto',

            'periodo_dia'
        ]

        for col in categorical_columns:

            if col in df.columns:

                df[col] = (

                    df[col]

                    .astype(str)

                    .astype('category')
                )

        return df

    @staticmethod
    def _classify_periodo(
        hour: Optional[int]
    ) -> str:
        """
        Descrição:
            Classifica período do dia baseado na hora.

        Parâmetros:
            hour (Optional[int]): Hora analisada.

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

    def summary(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Descrição:
            Retorna resumo estrutural do DataFrame.

        Parâmetros:
            df (pd.DataFrame): DataFrame analisado.

        Referências:
            ---
        """

        return {

            'rows': len(df),

            'columns': len(df.columns),

            'memory_mb': round(

                (
                    df.memory_usage(
                        deep=True
                    ).sum()
                    / 1024**2
                ),

                2
            ),

            'missing': (

                df.isna()

                .sum()

                .to_dict()
            )
        }

    def save(
        self,
        df: pd.DataFrame,
        path: str
    ) -> None:
        """
        Descrição:
            Exporta DataFrame para diferentes formatos.

        Parâmetros:
            df (pd.DataFrame): DataFrame exportado.
            path (str): Caminho de saída.

        Referências:
            ---
        """

        path_obj = Path(path)

        suffix = path_obj.suffix.lower()

        if suffix == '.csv':

            df.to_csv(
                path,
                index=False,
                encoding='utf-8'
            )

        elif suffix in ['.xlsx', '.xls']:

            df.to_excel(
                path,
                index=False
            )

        elif suffix == '.parquet':

            df.to_parquet(
                path,
                index=False
            )

        else:

            raise ValueError(
                f'Formato não suportado: {suffix}'
            )

        self.logger.info(
            f'DataFrame salvo em: {path}'
        )

    def load_and_prepare(
        self,
        path: str,
        qrcodes_path: Optional[str] = None,
        mapping_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Descrição:
            Executa pipeline completo de preparação dos dados.

        Parâmetros:
            path (str): Caminho principal da base.
            qrcodes_path (Optional[str]): Caminho opcional de QR Codes.
            mapping_path (Optional[str]): Caminho opcional de mapeamento.

        Referências:
            ---
        """

        df = self.load(path)

        return df