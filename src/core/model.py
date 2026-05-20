"""
Descrição:
    Módulo responsável pelas análises exploratórias,
    feature engineering, modelagem estatística,
    machine learning e inteligência artificial
    aplicadas à base de dados de supermercados.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 19/05/2026 - Estrutura inicial do módulo.
    2.0 - 19/05/2026 - Adição de feature engineering.
    3.0 - 19/05/2026 - Adição de Market Basket Analysis.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st

import pandas as pd
import numpy as np

from typing import Dict

from sklearn.preprocessing import (
    StandardScaler
)

from mlxtend.frequent_patterns import (
    fpgrowth,
    association_rules
)

from src.core.logger import setup_logger


logger = setup_logger(__name__)


# ==========================================================
# ANÁLISE EXPLORATÓRIA
# ==========================================================

def analyze_dataset(
    df: pd.DataFrame
) -> Dict:
    """
    Descrição:
        Realiza análise exploratória completa da base
        de dados, gerando estatísticas descritivas,
        diagnósticos de qualidade e informações
        estruturais para modelagem.

    Parâmetros:
        df (pd.DataFrame):
            Base analítica.

    Retorno:
        Dict:
            Dicionário contendo todas as análises.
    """

    logger.info(
        'Iniciando análise exploratória da base'
    )

    # ======================================================
    # SHAPE
    # ======================================================

    shape = {

        'linhas': df.shape[0],

        'colunas': df.shape[1]
    }

    # ======================================================
    # TIPOS
    # ======================================================

    dtypes = pd.DataFrame({

        'variavel': df.columns,

        'tipo': df.dtypes.astype(str)
    })

    # ======================================================
    # NULOS
    # ======================================================

    missing = pd.DataFrame({

        'total_nulos': df.isna().sum(),

        'percentual_nulos': (
            df
            .isna()
            .mean()
            .mul(100)
            .round(2)
        )
    })

    missing = (

        missing

        .sort_values(

            by='percentual_nulos',

            ascending=False
        )
    )

    # ======================================================
    # DUPLICADOS
    # ======================================================

    duplicates = {

        'total_duplicados':

        int(
            df.duplicated().sum()
        ),

        'percentual_duplicados':

        round(

            df.duplicated().mean() * 100,

            2
        )
    }

    # ======================================================
    # CARDINALIDADE
    # ======================================================

    cardinality = pd.DataFrame({

        'variavel': df.columns,

        'qtd_unicos': [

            df[col].nunique()

            for col in df.columns
        ]
    })

    cardinality = (

        cardinality

        .sort_values(

            by='qtd_unicos',

            ascending=False
        )
    )

    # ======================================================
    # ESTATÍSTICAS NUMÉRICAS
    # ======================================================

    numeric_description = (

        df

        .describe(

            include=[np.number]
        )

        .T
    )

    # ======================================================
    # ESTATÍSTICAS CATEGÓRICAS
    # ======================================================

    categorical_cols = (

        df

        .select_dtypes(

            include=['object', 'category']
        )

        .columns
    )

    categorical_summary = {}

    for col in categorical_cols:

        categorical_summary[col] = (

            df[col]

            .value_counts(

                dropna=False
            )

            .head(10)
        )

    # ======================================================
    # CORRELAÇÃO
    # ======================================================

    correlation = (

        df

        .corr(

            numeric_only=True
        )
    )

    # ======================================================
    # INTERVALO TEMPORAL
    # ======================================================

    date_range = None

    if 'data_hora' in df.columns:

        date_range = {

            'data_min':

            df['data_hora'].min(),

            'data_max':

            df['data_hora'].max(),

            'dias':

            (
                df['data_hora'].max()

                -

                df['data_hora'].min()
            ).days
        }

    # ======================================================
    # OUTLIERS (IQR)
    # ======================================================

    outliers = {}

    numeric_cols = (

        df

        .select_dtypes(

            include=np.number
        )

        .columns
    )

    for col in numeric_cols:

        q1 = df[col].quantile(0.25)

        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        total_outliers = (

            (
                df[col] < lower
            )

            |

            (
                df[col] > upper
            )

        ).sum()

        outliers[col] = {

            'total_outliers':

            int(total_outliers),

            'percentual_outliers':

            round(

                total_outliers
                /
                len(df)
                *
                100,

                2
            )
        }

    # ======================================================
    # MEMÓRIA
    # ======================================================

    memory_usage_mb = round(

        df.memory_usage(
            deep=True
        ).sum()
        /
        1024
        /
        1024,

        2
    )

    # ======================================================
    # RESULTADO FINAL
    # ======================================================

    results = {

        'shape': shape,

        'dtypes': dtypes,

        'missing': missing,

        'duplicates': duplicates,

        'cardinality': cardinality,

        'numeric_description': numeric_description,

        'categorical_summary': categorical_summary,

        'correlation': correlation,

        'date_range': date_range,

        'outliers': outliers,

        'memory_usage_mb': memory_usage_mb
    }

    logger.info(
        'Análise exploratória concluída'
    )

    return results


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def feature_engineering(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Descrição:
        Realiza feature engineering na base analítica,
        criando variáveis derivadas para análises,
        modelagem estatística, machine learning,
        IA generativa e sistemas de recomendação.

    Parâmetros:
        df (pd.DataFrame):
            Base analítica original.

    Retorno:
        pd.DataFrame:
            Base enriquecida com novas features.
    """

    logger.info(
        'Iniciando feature engineering'
    )

    temp = df.copy()

    # ======================================================
    # GARANTIA DATETIME
    # ======================================================

    temp['data_hora'] = pd.to_datetime(
        temp['data_hora']
    )

    # ======================================================
    # FEATURES TEMPORAIS
    # ======================================================

    logger.info(
        'Criando features temporais'
    )

    temp['ano'] = (
        temp['data_hora']
        .dt.year
    )

    temp['mes'] = (
        temp['data_hora']
        .dt.month
    )

    temp['dia'] = (
        temp['data_hora']
        .dt.day
    )

    temp['hora'] = (
        temp['data_hora']
        .dt.hour
    )

    temp['dia_ano'] = (
        temp['data_hora']
        .dt.dayofyear
    )

    temp['semana_ano'] = (
        temp['data_hora']
        .dt.isocalendar()
        .week
        .astype(int)
    )

    temp['fim_semana'] = np.where(

        temp['dia_semana'].isin([
            'SABADO',
            'DOMINGO'
        ]),

        1,

        0
    )

    # ======================================================
    # FEATURES FINANCEIRAS
    # ======================================================

    logger.info(
        'Criando features financeiras'
    )

    temp['ticket_medio_item'] = (

        temp['valor_total_nota']

        /

        temp['qtd_total_nota']
    )

    temp['percentual_tributo'] = (

        temp['valor_total_tributos']

        /

        temp['valor_total_nota']
    )

    temp['preco_por_quantidade'] = (

        temp['preco_total']

        /

        temp['quantidade']
    )

    # ======================================================
    # AJUSTES INFINITOS
    # ======================================================

    temp.replace(

        [np.inf, -np.inf],

        np.nan,

        inplace=True
    )

    # ======================================================
    # FEATURES DE FREQUÊNCIA
    # ======================================================

    logger.info(
        'Criando features de frequência'
    )

    freq_produto = (

        temp['produto']

        .value_counts()
    )

    temp['frequencia_produto'] = (

        temp['produto']

        .map(freq_produto)
    )

    freq_categoria = (

        temp['categoria_produto']

        .value_counts()
    )

    temp['frequencia_categoria'] = (

        temp['categoria_produto']

        .map(freq_categoria)
    )

    freq_supermercado = (

        temp['supermercado']

        .value_counts()
    )

    temp['frequencia_supermercado'] = (

        temp['supermercado']

        .map(freq_supermercado)
    )

    # ======================================================
    # FEATURES DE PREÇO
    # ======================================================

    logger.info(
        'Criando features de preço'
    )

    preco_medio_produto = (

        temp

        .groupby('produto')[
            'preco_unitario'
        ]

        .transform('mean')
    )

    temp['desvio_preco_produto'] = (

        temp['preco_unitario']

        -

        preco_medio_produto
    )

    temp['percentual_desvio_preco'] = (

        temp['desvio_preco_produto']

        /

        preco_medio_produto
    )

    # ======================================================
    # PRODUTO EM PROMOÇÃO
    # ======================================================

    temp['produto_promocao'] = np.where(

        temp['percentual_desvio_preco'] < -0.15,

        1,

        0
    )

    # ======================================================
    # FEATURES POR NOTA
    # ======================================================

    logger.info(
        'Criando features por nota fiscal'
    )

    qtd_itens_nota = (

        temp

        .groupby('chave_anonimizada')[
            'produto'
        ]

        .transform('count')
    )

    temp['qtd_itens_nota'] = (
        qtd_itens_nota
    )

    qtd_categorias_nota = (

        temp

        .groupby('chave_anonimizada')[
            'categoria_produto'
        ]

        .transform('nunique')
    )

    temp['qtd_categorias_nota'] = (
        qtd_categorias_nota
    )

    # ======================================================
    # FEATURES CLIMÁTICAS
    # ======================================================

    logger.info(
        'Criando features climáticas'
    )

    temp['amplitude_termica'] = (

        temp['temperatura_max']

        -

        temp['temperatura_min']
    )

    temp['chuva_intensa'] = np.where(

        temp['chuva_mm'] >= 10,

        1,

        0
    )

    # ======================================================
    # FEATURES NORMALIZADAS
    # ======================================================

    logger.info(
        'Criando variáveis normalizadas'
    )

    scaler = StandardScaler()

    numeric_cols = [

        'preco_unitario',

        'preco_total',

        'quantidade',

        'valor_total_nota',

        'temperatura_media',

        'chuva_mm'
    ]

    numeric_cols = [

        col for col in numeric_cols

        if col in temp.columns
    ]

    scaled_values = scaler.fit_transform(
        temp[numeric_cols]
    )

    scaled_df = pd.DataFrame(

        scaled_values,

        columns=[
            f'{col}_scaled'

            for col in numeric_cols
        ],

        index=temp.index
    )

    temp = pd.concat(

        [temp, scaled_df],

        axis=1
    )

    # ======================================================
    # TEXTO LIMPO PRODUTO
    # ======================================================

    logger.info(
        'Criando features textuais'
    )

    temp['produto_clean'] = (

        temp['produto']

        .astype(str)

        .str.upper()

        .str.strip()

        .str.replace(
            r'[^A-Z0-9 ]',
            '',
            regex=True
        )

        .str.replace(
            r'\s+',
            ' ',
            regex=True
        )
    )

    # ======================================================
    # TAMANHO TEXTO
    # ======================================================

    temp['produto_tamanho_texto'] = (

        temp['produto_clean']

        .str.len()
    )

    # ======================================================
    # FLAGS IMPORTANTES
    # ======================================================

    logger.info(
        'Criando flags analíticas'
    )

    temp['compra_grande'] = np.where(

        temp['valor_total_nota'] >=
        temp['valor_total_nota'].quantile(0.90),

        1,

        0
    )

    temp['produto_caro'] = np.where(

        temp['preco_unitario'] >=
        temp['preco_unitario'].quantile(0.90),

        1,

        0
    )

    temp['produto_barato'] = np.where(

        temp['preco_unitario'] <=
        temp['preco_unitario'].quantile(0.10),

        1,

        0
    )

    # ======================================================
    # FINALIZAÇÃO
    # ======================================================

    logger.info(
        f'Feature engineering concluído | '
        f'Novas colunas: {temp.shape[1] - df.shape[1]}'
    )

    return temp


# ==========================================================
# MARKET BASKET ANALYSIS
# ==========================================================

@st.cache_data(
    show_spinner=False
)

def run_market_basket_analysis(
    df: pd.DataFrame,
    min_support: float = 0.015,
    metric: str = 'lift',
    min_threshold: float = 1.0,
    max_len: int = 3,
    top_n_rules: int = 50,
    top_products_limit: int = 150
) -> Dict:
    """
    Descrição:
        Executa análise de cesta de compras
        (Market Basket Analysis) utilizando
        FP-Growth e regras de associação.

    Objetivos:
        - Identificar produtos frequentemente
          comprados juntos.
        - Descobrir padrões de consumo.
        - Encontrar relações fortes entre itens.
        - Criar base para recomendação de produtos.
        - Suportar IA generativa e sistemas de sugestão.

    Parâmetros:
        df (pd.DataFrame):
            Base analítica.

        min_support (float):
            Suporte mínimo do FP-Growth.

        metric (str):
            Métrica principal das regras.

        min_threshold (float):
            Valor mínimo da métrica.

        max_len (int):
            Quantidade máxima de itens por combinação.

        top_n_rules (int):
            Quantidade de regras retornadas.

    Retorno:
        Dict:
            Dicionário contendo:
                - basket_matrix
                - frequent_items
                - rules
                - top_rules
                - product_frequency
                - metrics_summary
    """

    logger.info(
        'Iniciando Market Basket Analysis'
    )

    temp = df.copy()

    # ======================================================
    # VALIDAÇÕES
    # ======================================================

    required_cols = [

        'chave_anonimizada',

        'produto'
    ]

    missing_cols = [

        col for col in required_cols

        if col not in temp.columns
    ]

    if missing_cols:

        raise ValueError(

            f'Colunas obrigatórias ausentes: '
            f'{missing_cols}'
        )

    # ======================================================
    # LIMPEZA
    # ======================================================

    logger.info(
        'Realizando limpeza dos dados'
    )

    temp = temp.dropna(

        subset=[
            'chave_anonimizada',
            'produto'
        ]
    )

    temp['produto'] = (

        temp['produto']

        .astype(str)

        .str.upper()

        .str.strip()
    )

    # ======================================================
    # FILTRO TOP PRODUTOS
    # ======================================================

    logger.info(
        'Selecionando produtos mais relevantes'
    )

    top_products = (

        temp['produto']

        .value_counts()

        .head(top_products_limit)

        .index
    )

    temp = temp[
        temp['produto'].isin(
            top_products
        )
    ]

    # ======================================================
    # MATRIZ BASKET
    # ======================================================

    logger.info(
        'Construindo basket matrix'
    )

    basket = (

        temp

        .groupby([
            'chave_anonimizada',
            'produto'
        ])['produto']

        .count()

        .unstack()

        .fillna(0)
    )

    basket = (

        basket > 0

    ).astype(int)

    # ======================================================
    # FREQUENT ITEMSETS
    # ======================================================

    logger.info(
        'Executando FP-Growth'
    )

    frequent_items = fpgrowth(

        basket,

        min_support=min_support,

        use_colnames=True,

        max_len=max_len
    )

    frequent_items = (

        frequent_items

        .sort_values(

            by='support',

            ascending=False
        )

        .reset_index(drop=True)
    )

    # ======================================================
    # REGRAS DE ASSOCIAÇÃO
    # ======================================================

    logger.info(
        'Gerando regras de associação'
    )

    rules = association_rules(

        frequent_items,

        metric=metric,

        min_threshold=min_threshold
    )

    # ======================================================
    # TRANSFORMAÇÃO TEXTO
    # ======================================================

    rules['antecedents'] = (

        rules['antecedents']

        .apply(
            lambda x:
            ', '.join(list(x))
        )
    )

    rules['consequents'] = (

        rules['consequents']

        .apply(
            lambda x:
            ', '.join(list(x))
        )
    )

    # ======================================================
    # ORDENAÇÃO
    # ======================================================

    rules = (

        rules

        .sort_values(

            by=[
                'lift',
                'confidence',
                'support'
            ],

            ascending=False
        )

        .reset_index(drop=True)
    )

    # ======================================================
    # TOP REGRAS
    # ======================================================

    top_rules = (

        rules

        .head(top_n_rules)
    )

    # ======================================================
    # FREQUÊNCIA PRODUTOS
    # ======================================================

    logger.info(
        'Calculando frequência dos produtos'
    )

    product_frequency = (

        temp['produto']

        .value_counts()

        .reset_index()
    )

    product_frequency.columns = [

        'produto',

        'frequencia'
    ]

    # ======================================================
    # MÉTRICAS RESUMIDAS
    # ======================================================

    logger.info(
        'Calculando métricas resumidas'
    )

    metrics_summary = {

        'total_notas':

        int(
            temp[
                'chave_anonimizada'
            ].nunique()
        ),

        'total_produtos':

        int(
            temp[
                'produto'
            ].nunique()
        ),

        'total_regras':

        int(
            len(rules)
        ),

        'media_lift':

        round(

            rules['lift'].mean(),

            4
        ) if len(rules) > 0 else None,

        'media_confidence':

        round(

            rules['confidence'].mean(),

            4
        ) if len(rules) > 0 else None,

        'media_support':

        round(

            rules['support'].mean(),

            4
        ) if len(rules) > 0 else None
    }

    # ======================================================
    # RESULTADO FINAL
    # ======================================================

    results = {

        'basket_matrix': basket,

        'frequent_items': frequent_items,

        'rules': rules,

        'top_rules': top_rules,

        'product_frequency': product_frequency,

        'metrics_summary': metrics_summary
    }

    logger.info(
        'Market Basket Analysis concluído'
    )

    return results