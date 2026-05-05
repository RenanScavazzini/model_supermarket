"""
Descrição:
    Módulo responsável pela construção de modelos estatísticos e de machine learning
    aplicados a dados de notas fiscais de supermercado. Inclui modelos de classificação,
    regressão, clusterização e análise de associação de produtos.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 29/04/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


@dataclass
class ModelResult:
    """
    Descrição:
        Estrutura de dados para armazenar o resultado de modelos estatísticos ou de
        machine learning, incluindo nome, descrição, métricas de avaliação e o
        objeto do modelo treinado.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    nome: str
    descricao: str
    metrics: Dict[str, float]
    modelo: Optional[object] = None


class StatisticalModels:
    """
    Descrição:
        Classe responsável pela construção e treinamento de modelos estatísticos e
        de machine learning voltados para análise de comportamento de consumo.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    @staticmethod
    def build_high_spend_flag(df: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
        """
        Descrição:
            Cria uma variável binária indicando se uma nota fiscal pertence ao grupo
            de alto gasto com base em um quantil da distribuição.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados de notas fiscais.
            quantile (float): Percentil utilizado como threshold para definir alto gasto.

        Referências:
            - Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning.
        """
        invoice_totals = df.groupby('nota_fiscal_id')['preco_total'].sum()
        threshold = invoice_totals.quantile(quantile)
        high_spend = invoice_totals >= threshold
        df_flag = df.drop(columns=['preco_total'], errors='ignore').merge(
            high_spend.rename('gasto_alto'),
            left_on='nota_fiscal_id',
            right_index=True,
            how='left'
        )
        df_flag['gasto_alto'] = df_flag['gasto_alto'].fillna(False).astype(int)
        return df_flag

    @staticmethod
    def train_naive_bayes(df: pd.DataFrame, target_column: str = 'gasto_alto') -> ModelResult:
        """
        Descrição:
            Treina um modelo Naive Bayes categórico para classificar notas fiscais
            com alto gasto com base em variáveis agregadas.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.
            target_column (str): Nome da variável alvo.

        Referências:
            - Murphy, K. (2012). Machine Learning: A Probabilistic Perspective.
        """
        invoice_features = df.groupby('nota_fiscal_id').agg(
            supermercado=('supermercado', 'first'),
            periodo_dia=('periodo_dia', 'first'),
            total_itens=('quantidade', 'sum'),
            gasto_total=('preco_total', 'sum'),
        )
        invoice_features[target_column] = df.groupby('nota_fiscal_id')[target_column].first()
        invoice_features = invoice_features.reset_index(drop=True)

        X = invoice_features[['supermercado', 'periodo_dia', 'total_itens']]
        y = invoice_features[target_column]

        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_encoded = encoder.fit_transform(X[['supermercado', 'periodo_dia']])
        X_final = np.concatenate([X_encoded, X[['total_itens']].values], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42, stratify=y
        )

        model = CategoricalNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return ModelResult(
            nome='Naive Bayes de Gasto Alto',
            descricao='Classifica notas fiscais de alto gasto com base em loja, período do dia e número de itens.',
            metrics={'acuracia': float(accuracy)},
            modelo=model,
        )

    @staticmethod
    def train_spend_regressor(df: pd.DataFrame) -> ModelResult:
        """
        Descrição:
            Treina um modelo de regressão para prever o valor total gasto em uma nota fiscal.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados agregados.

        Referências:
            - Breiman, L. (2001). Random Forests.
        """
        invoice_features = df.groupby('nota_fiscal_id').agg(
            supermercado=('supermercado', 'first'),
            periodo_dia=('periodo_dia', 'first'),
            total_itens=('quantidade', 'sum'),
            gasto_total=('preco_total', 'sum'),
        ).reset_index(drop=True)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_cat = encoder.fit_transform(invoice_features[['supermercado', 'periodo_dia']])
        X = np.concatenate([X_cat, invoice_features[['total_itens']].values], axis=1)
        y = invoice_features['gasto_total'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        return ModelResult(
            nome='Regressão de Gasto Total',
            descricao='Prevê o gasto total de uma nota com base em características agregadas.',
            metrics={'mse': float(mse)},
            modelo=model,
        )

    @staticmethod
    def fit_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> ModelResult:
        """
        Descrição:
            Aplica algoritmo KMeans para segmentar notas fiscais com base em comportamento de compra.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.
            n_clusters (int): Número de clusters.

        Referências:
            - MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
        """
        invoice_features = df.groupby('nota_fiscal_id').agg(
            total_itens=('quantidade', 'sum'),
            gasto_total=('preco_total', 'sum'),
        ).reset_index(drop=True)

        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(invoice_features[['total_itens', 'gasto_total']])
        invoice_features['cluster'] = model.labels_
        inertia = float(model.inertia_)

        return ModelResult(
            nome='Clusterização de Notas',
            descricao='Agrupa notas fiscais por volume de itens e valor total.',
            metrics={'inercia': inertia},
            modelo=model,
        )

    @staticmethod
    def product_association(df: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
        """
        Descrição:
            Calcula a coocorrência de produtos em notas fiscais utilizando suporte mínimo.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.
            min_support (float): Suporte mínimo para considerar associação.

        Referências:
            - Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules.
        """
        basket = df.groupby(['nota_fiscal_id', 'produto'])['quantidade'].sum().unstack(fill_value=0)
        support = (basket > 0).sum() / basket.shape[0]
        popular = support[support >= min_support].index.tolist()
        sub = basket[popular].astype(bool)

        pairs = []
        for i, prod_i in enumerate(popular):
            for prod_j in popular[i + 1:]:
                support_ij = float((sub[prod_i] & sub[prod_j]).mean())
                if support_ij >= min_support:
                    pairs.append({'produto_a': prod_i, 'produto_b': prod_j, 'suporte': support_ij})

        return pd.DataFrame(pairs).sort_values('suporte', ascending=False)