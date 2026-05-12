"""
Descrição:
    Módulo responsável pela implementação de modelos estatísticos
    e algoritmos de Machine Learning aplicados à análise de consumo
    em notas fiscais de supermercados.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

from src.core.logger import setup_logger


class StatisticalModels:
    """
    Descrição:
        Classe responsável pelo treinamento e execução de modelos
        estatísticos aplicados ao comportamento de consumo.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        """
        Descrição:
            Inicializa os modelos estatísticos.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados analíticos.

        Referências:
            ---
        """

        self.logger = setup_logger(
            self.__class__.__name__
        )

        self.df = df

    def train_regression(
        self,
        features: list[str],
        target: str
    ) -> LinearRegression:
        """
        Descrição:
            Treina modelo de regressão linear.

        Parâmetros:
            features (list[str]): Variáveis independentes.
            target (str): Variável alvo.

        Referências:
            - James, G. et al. (2021). An Introduction to Statistical Learning.
        """

        X = self.df[features]

        y = self.df[target]

        model = LinearRegression()

        model.fit(X, y)

        self.logger.info(
            'Modelo de regressão treinado'
        )

        return model

    def train_naive_bayes(
        self,
        features: list[str],
        target: str
    ) -> GaussianNB:
        """
        Descrição:
            Treina modelo Naive Bayes.

        Parâmetros:
            features (list[str]): Variáveis independentes.
            target (str): Variável alvo.

        Referências:
            - Hastie, T. et al. (2009). The Elements of Statistical Learning.
        """

        X = self.df[features]

        y = self.df[target]

        model = GaussianNB()

        model.fit(X, y)

        self.logger.info(
            'Modelo Naive Bayes treinado'
        )

        return model

    def train_kmeans(
        self,
        features: list[str],
        n_clusters: int = 3
    ) -> KMeans:
        """
        Descrição:
            Treina modelo KMeans para clusterização.

        Parâmetros:
            features (list[str]): Variáveis utilizadas na clusterização.
            n_clusters (int): Quantidade de clusters.

        Referências:
            - Hastie, T. et al. (2009). The Elements of Statistical Learning.
        """

        X = self.df[features]

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )

        model.fit(X)

        self.logger.info(
            f'Modelo KMeans treinado com {n_clusters} clusters'
        )

        return model