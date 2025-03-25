import pandas as pd 
import numpy as np 
import seaborn as sns
import locale
from functools import reduce
from matplotlib import pyplot as plt 
import sys
from typing import List, Union 
from scipy.stats import shapiro, kendalltau, spaearmanr, pearsonr, normaltest
from itertools import combinations
from multiprocessing import Pool

# from bradac.utils.connectors.spark import SparkConnector
import pyspark.sql.dataframe as df 
from pyspark.sql import DataFrame, functions as F, Row, types as T
from pyspark.sql.window import Window 
from pyspark.ml.feature import Bucketizer
from pyspark.ml.stat import Correlation 

from utils.connect_spark import get_setup 

setup = get_setup()
ss = SparkConnector(conf=setup)
spark = ss.getSpark()


def tab_analise_bin(
    dataset: DataFrame, tipos_var: dict
) -> pd.DataFrame:
    binaries = tipos_var['binary']
    target = tipos_var['target']

    df_aux = pd.DataFrame(
        {
            'VARIAVEL': [],
            'CATEGORIA': [],
            'QUANTIDADE_TOTAL': [],
            'QUANTIDADE_BAD': [],
        }
    )

    for bin in binaries:
        dataset.withColumn(bin, F.col(bin).cast(IntegerType()))

        df_aux['VARIAVEL'] = ((bin + ' ') * 3).split()

        df_bad = (
            data.filter(F.col(target) == 1)
            .groupBy(bin)
            .agg(f.count('*').alias('QUANTIDADE_BAD'))
            .withColumnRenameds(bin, 'CATEGORIA')
        )
        df_bad = df_bad.withColumn(
            'CATEGORIA',
            F.when(F.col('CATEGORIA').isNull(), ' MISSING').otherwise(
                F.col('CATEGORIA')
            ),
        )
        df_bad = df_bad.withColumn(
            'CATEGORIA', F.col('CATEGORIA').cast('string')
        )

        df_total = (
            dataset.groupBy(bin)
            .agg(F.count('*').alias('QUANTIDADE_TOTAL'))
            .withColumnRenamed(bin, 'CATEGORIA')
        )
        df_total = df_total.withColumn(
            'CATEGORIA',
            F.when(F.col('CATEGORIA').isNull(), ' MISSING').otherwise(
                F.col('CATEGORIA')
            ),
        )
        df_total = df_total.withColumn(
            'CATEGORIA', F.col('CATEGORIA').cast('string')
        )

        df_bad = df_bad.toPandas()
        df_total = df_total.toPandas()

        df_csl = pd.merge(df_aux, df_bad, on='CATEGORIA', how='left')
        df_csl = pd.merge(df_csl, df_total, on='CATEGORIA', how='left')

        df_geral = pd.concat([df_geral, df_csl])
        df_geral.loc[
            (df_geral['QUANTIDADE_TOTAL'].isna()), 'QUANTIDADE_TOTAL'
        ] = 0
        df_geral['QUANTIDADE_TOTAL'] = df_geral['QUANTIDADE_TOTAL'].astype(
            'Int64'
        )
        df_geral.loc[(df_geral['QUANTIDADE_BAD'].isna()), 'QUANTIDADE_BAD'] = 0
        df_geral['QUANTIDADE_BAD'] = df_geral['QUANTIDADE_BAD'].astype('Int64')
        df_geral = df_geral.sort_values(by='VARIAVEL')

    return df_geral


def tab_analise_cat(
    dataset: DataFrame, tipos_var: list
) -> pd.DataFrame:
    categories = tipos_var['categorical']
    target = tipos_var['target']

    df_geral = pd.DataFrame(
        {
            'VARIAVEL': [],
            'CATEGORIA': [],
            'QUANTIDADE_TOTAL': [],
            'QUANTIDADE_BAD': [],
        }
    )

    for categ in categories:
        df_aux = pd.DataFrame(
            {
                'VARIAVEL': (
                    (str(categ) + ' ') * dataset.select(categ).distinct().count()
                ).split(),
                'CATEGORIA': dataset.select(categ)
                .distinct()
                .toPandas()[categ],
            }
        )
        df_aux.loc[(df_aux['CATEGORIA'].isna()), 'CATEGORIA'] = ' MISSING'

        df_bad = (
            dataset.filter(F.col(target) == 1)
            .groupBy(categ)
            .agg(F.count('*').alias('QUANTIDADE_BAD'))
            .withColumnRenamed(categ, 'CATEGORIA')
        )
        df_bad = df_bad.withColumn(
            'CATEGORIA', F.col('CATEGORIA').cast('string')
        )

        df_total = (
            dataset.groupBy(categ)
            .agg(F.count('*').alias('QUANTIDADE_TOTAL'))
            .withColumnRenamed(categ, 'CATEGORIA')
        )
        df_total = df_total.withColumn(
            'CATEGORIA',
            F.when(F.col('CATEGORIA').isNull(), ' MISSING').otherwise(
                F.col('CATEGORIA')
            ),
        )
        df_total = df_total.withColumn(
            'CATEGORIA', F.col('CATEGORIA').cast('string')
        )

        df_bad = df_bad.toPandas()
        df_total = df_total.toPandas()

        df_csl = pd.merge(df_aux, df_bad, on='CATEGORIA', how='left')
        df_csl = pd.merge(df_csl, df_total, on='CATEGORIA', how='left')

        df_geral = pd.concat([df_geral, df_csl])
        df_geral.loc[
            (df_geral['QUANTIDADE_TOTAL'].isna()), 'QUANTIDADE_TOTAL'
        ] = 0
        df_geral['QUANTIDADE_TOTAL'] = df_geral['QUANTIDADE_TOTAL'].astype(
            'Int64'
        )
        df_geral.loc[(df_geral['QUANTIDADE_BAD'].isna()), 'QUANTIDADE_BAD'] = 0
        df_geral['QUANTIDADE_BAD'] = df_geral['QUANTIDADE_BAD'].astype('Int64')
        df_geral = df_geral.sort_values(by='VARIAVEL')

    return df_geral


def tab_analise_num(
    dataset: DataFrame, tipos_var: list, imputado: bool = False
) -> pd.DataFrame:
    numerics = tipos_var['numeric']
    target = tipos_var['target']

    t1 = dataset.select(*numerics).describe().toPandas().transpose()
    t1.rename(
        columns={
            0: 'QUANTIDADE_TOTAL',
            1: 'MEDIA_TOTAL',
            2: 'DESVIO_TOTAL',
            3: 'MINIMO_TOTAL',
            4: 'MAXIMO_TOTAL',
        },
        inplace=True,
    )
    t1 = t1.drop(t1.index[0])

    t2 = (
        dataset.filter(F.col(target) == 1)
        .select(*numerics)
        .describe()
        .toPandas()
        .transpose()
    )
    t2.rename(
        columns={
            0: 'QUANTIDADE_BAD',
            1: 'MEDIA_BAD',
            2: 'DESVIO_BAD',
            3: 'MINIMO_BAD',
            4: 'MAXIMO_BAD',
        },
        inplace=True,
    )
    t2 = t2.drop(t2.index[0])

    tab_final = pd.concat([t1, t2], axis=1)

    return tab_final


def calcule_response_frequency(dataset, response_column='response'):
    total_count = dataset.count()
    
    response_frequency = dataset.groupBy(response_column).count().withColumn('PERCENTUAL', F.round((F.col('count')/total_count)*100, 2))
    response_frequency = response_frequency.withColumnRenamed('response', 'COMUNICADOS')
    response_frequency = response_frequency.withColumnRenamed('count', 'QUANTIDADE')

    newRow = spark.createDataFrame([('TOTAL', total_count, 100.00)], ['COMUNICADOS', 'QUANTIDADE', 'PERCENTUAL'])
    response_frequency = response_frequency.union(newRow)
    response_frequency = response_frequency \
        .withColumn('QUANTIDADE', F.format_number('QUANTIDADE', 0)) \
        .withColumn('QUANTIDADE', F.regexp_replace('QUANTIDADE', ',', '.')) \
        .withColumn('PERCENTUAL', F.regexp_replace('PERCENTUAL', '\\.', '.'))
    response_frequency = response_frequency.toPandas()

    return response_frequency