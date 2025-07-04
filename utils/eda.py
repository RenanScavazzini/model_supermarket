import pandas as pd 
import numpy as np 
import seaborn as sns
import locale
from functools import reduce
from matplotlib import pyplot as plt 
import sys
from typing import List, Union 
from scipy.stats import shapiro, kendalltau, spearmanr, pearsonr, normaltest
from itertools import combinations
from multiprocessing import Pool
import time
from nfceget import app
import openpyxl
import warnings

import pyspark.sql.dataframe as df 
from pyspark.sql import DataFrame, functions as F, Row, types as T
from pyspark.sql.window import Window 
from pyspark.ml.feature import Bucketizer
from pyspark.ml.stat import Correlation 


def tictoc(tic, toc):
    """
    Descrição:
        Calcula o tempo decorrido entre dois instantes e retorna uma string formatada com o tempo decorrido e os horários de início e fim.
    --------------------
    Argumentos:
        tic (float): Timestamp inicial (em segundos desde a época Unix).
        toc (float): Timestamp final (em segundos desde a época Unix).
    --------------------
    Retorno:
        str: String formatada com o tempo decorrido e os horários de início e fim.
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    p1 = time.strftime("%H:%M:%S", time.gmtime(toc - tic))
    p2 = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(tic - 10800))
    p3 = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(toc - 10800))
    return f'({p1}) [{p2} - [{p3}]'


def produtos_unicos(df):
    """
    Descrição:
        Realiza a análise e transformação dos dados de produtos, identificando códigos e nomes distintos,
        ordenando o DataFrame, e aplicando o maior código e nome mais recente para cada produto.
    --------------------
    Argumentos:
        df (DataFrame): DataFrame contendo os dados dos produtos.
    --------------------
    Retorno:
        DataFrame: DataFrame transformado com os produtos únicos.
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    print('Antes da transformação:')
    print(f"Há \033[93m{df['COD_PRODUTO'].nunique()}\033[0m código(s) de produto(s) distinto(s).")
    print(f"Há \033[93m{df['PRODUTO'].nunique()}\033[0m nome(s) de produto(s) distinto(s).")
    print('\n')
    df = df.sort_values(by=["PRODUTO", "COD_PRODUTO", "DATA"])
    maior_codigo_produto = df.groupby("PRODUTO")["COD_PRODUTO"].max()
    df["COD_PRODUTO"] = df["PRODUTO"].map(maior_codigo_produto)
    produto_mais_recente = df.drop_duplicates(subset="COD_PRODUTO", keep="last").set_index("COD_PRODUTO")["PRODUTO"]
    df["PRODUTO"] = df["COD_PRODUTO"].map(produto_mais_recente)
    print('Após a transformação:')
    print(f"Há \033[93m{df['COD_PRODUTO'].nunique()}\033[0m código(s) de produto(s) distinto(s).")
    print(f"Há \033[93m{df['PRODUTO'].nunique()}\033[0m nome(s) de produto(s) distinto(s).")
    df_aux1 = df.groupby("COD_PRODUTO").filter(lambda x: x["PRODUTO"].nunique() > 1)
    print(f"Há \033[93m{df_aux1['COD_PRODUTO'].nunique()}\033[0m código(s) de produto(s) com o mesmo código mas com nome(s) diferente(s)")
    print(f"Há \033[93m{df_aux1['PRODUTO'].nunique()}\033[0m nome(s) de produto(s) com o mesmo código mas com nome(s) diferente(s)")

    return df


def resumo_valores_gerais(df):
    """
    Descrição:
        Realiza a análise e resumo dos dados de gastos, identificando totais, médias e extremos por data, mês, ano, supermercado e período.
    --------------------
    Argumentos:
        df (DataFrame): DataFrame contendo os dados dos gastos.
    --------------------
    Retorno:
        None
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    print('\033[93mResumos gerais:\033[0m')
    total_gastos = df["VALOR_TOTAL"].sum()
    idas = df["CHAVE_ANON"].nunique()
    media_gastos = total_gastos / idas
    print(f"Total de gastos: R$ {round(total_gastos,2)}")
    print(f"Quantidade de idas ao mercado: {idas}")
    print(f"Média de gastos por nota fiscal: {round(media_gastos,2)}")

    print('\n\033[93mResumo por datas:\033[0m')
    total_por_data = df.groupby(pd.to_datetime(df["DATA"]))["VALOR_TOTAL"].sum()
    dia_mais_gasto = total_por_data.idxmax()
    dia_menos_gasto = total_por_data.idxmin()
    media_gasto_dia = total_por_data.mean()
    print(
        f"Dia que mais gastou: {dia_mais_gasto} com valor total de {total_por_data[dia_mais_gasto]}"
    )
    print(
        f"Dia que menos gastou: {dia_menos_gasto} com valor total de {total_por_data[dia_menos_gasto]}"
    )
    print(f"Gasto médio por dia: {round(media_gasto_dia,2)}")
    total_por_mes = df.groupby(pd.to_datetime(df["DATA"]).dt.to_period("M"))["VALOR_TOTAL"].sum()
    mes_mais_gasto = total_por_mes.idxmax()
    mes_menos_gasto = total_por_mes.idxmin()
    media_gasto_mes = total_por_mes.mean()
    print(
        f"\nMês que mais gastou: {mes_mais_gasto} com valor total de {total_por_mes[mes_mais_gasto]}"
    )
    print(
        f"Mês que menos gastou: {mes_menos_gasto} com valor total de {total_por_mes[mes_menos_gasto]}"
    )
    print(f"Gasto médio por mês: {round(media_gasto_mes,2)}")
    total_por_ano = df.groupby(pd.to_datetime(df["DATA"]).dt.year)["VALOR_TOTAL"].sum()
    ano_mais_gasto = total_por_ano.idxmax()
    ano_menos_gasto = total_por_ano.idxmin()
    media_gasto_ano = total_por_ano.mean()
    print(
        f"\nAno que mais gastou: {ano_mais_gasto} com valor total de {total_por_ano[ano_mais_gasto]}"
    )
    print(
        f"Ano que menos gastou: {ano_menos_gasto} com valor total de {total_por_ano[ano_menos_gasto]}"
    )
    print(f"Gasto médio por ano: {round(media_gasto_ano,2)}")


def resumo_valores_lugar_periodo(df):
    """
    Descrição:
        Realiza o resumo dos gastos agrupados por supermercado e por período.
    --------------------
    Argumentos:
        df (DataFrame): DataFrame contendo os dados dos gastos.
    --------------------
    Retorno:
        None
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    print('\n\033[93mResumo por supermercado:\033[0m')
    total_por_supermercado = df.groupby("SUPERMERCADO")["VALOR_TOTAL"].sum()
    print(f"Valor total gasto por supermercado:\n{total_por_supermercado}")

    print('\n\033[93mResumo por período:\033[0m')
    total_por_supermercado = df.groupby("PERIODO")["VALOR_TOTAL"].sum()
    print(f"Valor total gasto por período:\n{total_por_supermercado}")


def resumo_valores_ano_mes(df):
    """
    Descrição:
        Realiza o resumo dos gastos agrupados por ano e por mês.
    --------------------
    Argumentos:
        df (DataFrame): DataFrame contendo os dados dos gastos.
    --------------------
    Retorno:
        None
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    print('\n\033[93mResumo por ano:\033[0m')
    print(f"Valor total gasto por ano:")
    total_por_ano = df.groupby(pd.to_datetime(df["DATA"]).dt.year)["VALOR_TOTAL"].sum()
    display(total_por_ano)

    print('\n\033[93mResumo por mês:\033[0m')
    print(f"Valor total gasto por mês:")
    total_por_mes = df.groupby(pd.to_datetime(df["DATA"]).dt.to_period("M"))["VALOR_TOTAL"].sum()
    display(total_por_mes)


def resumo_produto_mais_comprado(df):
    """
    Descrição:
        Realiza o resumo do produto mais comprado, mostrando quantidade total, valor total gasto e agrupamento mensal.
    --------------------
    Argumentos:
        df (DataFrame): DataFrame contendo os dados dos gastos.
    --------------------
    Retorno:
        None
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    print('\n\033[93mResumo do produto mais comprado:\033[0m')
    soma_quantidade = df.groupby("PRODUTO")["QTDE"].sum()
    produto_mais_comprado = soma_quantidade.idxmax()
    df_produto_mais_comprado = df[df["PRODUTO"] == produto_mais_comprado].copy()
    df_produto_mais_comprado["DATA"] = pd.to_datetime(df_produto_mais_comprado["DATA"])
    total_quantidade_produto_mais_comprado = df_produto_mais_comprado["QTDE"].sum()
    total_gastos_produto_mais_comprado = df_produto_mais_comprado["VALOR_TOTAL"].sum()
    print(f"O produto que mais comprei foi: {produto_mais_comprado}")
    print(f"Quantidade comprada: {total_quantidade_produto_mais_comprado}")
    print(f"Total gasto: {round(total_gastos_produto_mais_comprado,2)}")
    soma_por_mes = df_produto_mais_comprado.groupby(
        df_produto_mais_comprado["DATA"].dt.to_period("M")
    ).agg({"QTDE": "sum", "VALOR_TOTAL": "sum"})
    print(
        f"\nSoma da quantidade e valor total do produto '{produto_mais_comprado}' por mês:\n{soma_por_mes}"
    )


def exibe_subtabela_num_produto(
    df,
    num_produto
):
    """
    Descrição:
        Exibe a subtabela de um produto específico, ordenada por data, a partir do DataFrame fornecido.
    --------------------
    Argumentos:
        df (DataFrame): DataFrame contendo os dados dos produtos.
        num_produto (int): Índice do produto na lista de produtos únicos.
    --------------------
    Retorno:
        None
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    df_unitario = df[["PRODUTO", "VALOR_UNIDADE", "DATA"]].drop_duplicates()
    df_unitario = df_unitario.sort_values(["PRODUTO", "DATA"])
    produtos = df_unitario["PRODUTO"].unique()
    subtabelas = {
        produto: df_unitario[df_unitario["PRODUTO"] == produto].sort_values("DATA")
        for produto in produtos
    }
    display(subtabelas[produtos[num_produto]])


def exibe_subtabela_nome_produto(
    df,
    nome_produto
):
    """
    Descrição:
        Exibe as subtabelas de produtos cujo nome contém o termo especificado, ordenadas por data.
    --------------------
    Argumentos:
        df (DataFrame): DataFrame contendo os dados dos produtos.
        nome_produto (str): Termo a ser buscado nos nomes dos produtos.
    --------------------
    Retorno:
        None
    --------------------
    Autor: 
        Renan Douglas Floriano Scavazzini <renanscavazzini@gmail.com>
    --------------------
    Data da última atualização: 
        04/07/2025
    """
    df_unitario = df[["PRODUTO", "VALOR_UNIDADE", "DATA"]].drop_duplicates()
    df_unitario = df_unitario.sort_values(["PRODUTO", "DATA"])
    produtos = df_unitario["PRODUTO"].unique()
    subtabelas = {
        produto: df_unitario[df_unitario["PRODUTO"] == produto].sort_values("DATA")
        for produto in produtos
    }
    for produto, subtabela in subtabelas.items():
        if nome_produto.lower() in produto.lower():
            display(subtabela)





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