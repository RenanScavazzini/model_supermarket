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
    nome: str
    descricao: str
    metrics: Dict[str, float]
    modelo: Optional[object] = None


class StatisticalModels:
    """Constrói modelos estatísticos sobre dados de supermercado."""

    @staticmethod
    def build_high_spend_flag(df: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
        """Adiciona variável binária de gasto alto por nota fiscal."""
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
        """Treina um modelo Naive Bayes para classificar notas de alto gasto."""
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
        """Treina um modelo de regressão para prever gasto total."""
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
        """Agrupa notas fiscais em clusters com KMeans."""
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
        """Calcula coocorrência simples de produtos em notas fiscais."""
        basket = df.groupby(['nota_fiscal_id', 'produto'])['quantidade'].sum().unstack(fill_value=0)
        support = (basket > 0).sum() / basket.shape[0]
        popular = support[support >= min_support].index.tolist()
        sub = basket[popular].astype(bool)

        pairs = []
        for i, prod_i in enumerate(popular):
            for prod_j in popular[i + 1 :]:
                support_ij = float((sub[prod_i] & sub[prod_j]).mean())
                if support_ij >= min_support:
                    pairs.append({'produto_a': prod_i, 'produto_b': prod_j, 'suporte': support_ij})

        return pd.DataFrame(pairs).sort_values('suporte', ascending=False)
