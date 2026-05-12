import pandas as pd 


def produtos_unicos(df):
    """
    Descrição:
        Realiza a padronização de produtos no DataFrame, identificando nomes
        e códigos distintos, ordenando os registros e associando o maior
        código ao nome mais recente de cada produto.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados dos produtos.

    Retorno:
        pd.DataFrame: DataFrame transformado contendo produtos únicos e
        padronizados.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    print('Antes da transformação:')
    print(f"Há \033[93m{df['COD_PRODUTO'].nunique()}\033[0m código(s) de produto(s) distinto(s).")
    print(f"Há \033[93m{df['PRODUTO'].nunique()}\033[0m nome(s) de produto(s) distinto(s).")
    print('\n')

    df = df.sort_values(by=["PRODUTO", "COD_PRODUTO", "DATA"])

    maior_codigo_produto = df.groupby("PRODUTO")["COD_PRODUTO"].max()
    df["COD_PRODUTO"] = df["PRODUTO"].map(maior_codigo_produto)

    produto_mais_recente = (
        df.drop_duplicates(subset="COD_PRODUTO", keep="last")
        .set_index("COD_PRODUTO")["PRODUTO"]
    )

    df["PRODUTO"] = df["COD_PRODUTO"].map(produto_mais_recente)

    print('Após a transformação:')
    print(f"Há \033[93m{df['COD_PRODUTO'].nunique()}\033[0m código(s) de produto(s) distinto(s).")
    print(f"Há \033[93m{df['PRODUTO'].nunique()}\033[0m nome(s) de produto(s) distinto(s).")

    df_aux1 = df.groupby("COD_PRODUTO").filter(
        lambda x: x["PRODUTO"].nunique() > 1
    )

    print(
        f"Há \033[93m{df_aux1['COD_PRODUTO'].nunique()}\033[0m código(s) "
        "de produto(s) com nomes diferentes"
    )

    return df


def resumo_valores_gerais(df):
    """
    Descrição:
        Realiza análises estatísticas gerais dos gastos presentes no
        DataFrame, incluindo totais, médias e identificação de períodos
        com maior e menor valor gasto.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados financeiros e de
            compras.

    Retorno:
        pd.DataFrame: DataFrame com os resumos gerais dos gastos, incluindo totais e médias.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    print('\033[93mResumos gerais:\033[0m')

    total_gastos = df["VALOR_TOTAL_PRODUTO"].sum()
    idas = df["CHAVE_ANONIMIZADA"].nunique()
    media_gastos = total_gastos / idas

    print(f"Total de gastos: R$ {round(total_gastos, 2)}")
    print(f"Quantidade de idas ao mercado: {idas}")
    print(f"Média de gastos por nota fiscal: {round(media_gastos, 2)}")

    return pd.DataFrame({
        "Total de gastos": [total_gastos],
        "Quantidade de idas ao mercado": [idas],
        "Média de gastos por nota fiscal": [media_gastos]
    })


def resumo_valores_lugar_periodo(df):
    """
    Descrição:
        Realiza o resumo dos gastos agrupados por supermercado e período
        de compra.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados financeiros das
            compras.

    Retorno:
        tuple: Tupla contendo dois DataFrames, um com o resumo por
        supermercado e outro com o resumo por período.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    print('\n\033[93mResumo por supermercado:\033[0m')

    total_por_supermercado = (
        df.groupby("SUPERMERCADO")["VALOR_TOTAL_PRODUTO"].sum()
    )

    print(total_por_supermercado)

    print('\n\033[93mResumo por período:\033[0m')

    total_por_periodo = (
        df.groupby("PERIODO")["VALOR_TOTAL_PRODUTO"].sum()
    )

    print(total_por_periodo)

    return total_por_supermercado, total_por_periodo


def resumo_valores_ano_mes(df):
    """
    Descrição:
        Realiza o resumo dos gastos agrupados por ano e mês, exibindo
        valores totais para cada período temporal identificado.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados financeiros e
            datas das compras.

    Retorno:
        tuple: Tupla contendo dois DataFrames, um com o resumo por ano
        e outro com o resumo por mês.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    print('\n\033[93mResumo por ano:\033[0m')

    total_por_ano = (
        df.groupby(pd.to_datetime(df["DATA"]).dt.year)
        ["VALOR_TOTAL_PRODUTO"]
        .sum()
    )
    print(total_por_ano)

    print('\n\033[93mResumo por mês:\033[0m')

    total_por_mes = (
        df.groupby(pd.to_datetime(df["DATA"]).dt.to_period("M"))
        ["VALOR_TOTAL_PRODUTO"]
        .sum()
    )

    print(total_por_mes)

    return total_por_ano, total_por_mes


def resumo_produto_mais_comprado(df):
    """
    Descrição:
        Identifica o produto mais comprado considerando a quantidade
        total adquirida e apresenta estatísticas agregadas de consumo
        e gastos ao longo do tempo.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados dos produtos e
            compras realizadas.

    Retorno:
        str: Nome do produto mais comprado com base na quantidade total
        adquirida.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    print('\n\033[93mResumo do produto mais comprado:\033[0m')

    soma_quantidade = df.groupby("PRODUTO")["QTDE"].sum()

    produto_mais_comprado = soma_quantidade.idxmax()

    print(f"O produto que mais comprei foi: {produto_mais_comprado}")

    return produto_mais_comprado


def exibe_subtabela_cod_produto(df, cod_produto):
    """
    Descrição:
        Exibe a subtabela de um produto específico a partir
        do código do produto (COD_PRODUTO).

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados.
        cod_produto (int | str): Código do produto.

    Retorno:
        pd.DataFrame: Subtabela contendo o histórico do produto.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 10/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    subtabela = (
        df.loc[df["COD_PRODUTO"] == cod_produto,
               ["COD_PRODUTO", "PRODUTO", "VALOR_PRODUTO", "DATA"]]
        .drop_duplicates()
        .sort_values("DATA")
    )

    return subtabela


def exibe_subtabela_nome_produto(df, nome_produto):
    """
    Descrição:
        Exibe subtabelas de produtos cujo nome contenha o termo
        informado pelo usuário, permitindo consultas textuais parciais.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados dos produtos.
        nome_produto (str): Texto utilizado para busca nos nomes dos
            produtos.

    Retorno:
        list: Lista de DataFrames, cada uma correspondente a uma subtabela de produto encontrado 
              ou um DataFrame vazio caso nenhum produto corresponda ao critério de busca.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    df_unitario = (
        df[["PRODUTO", "COD_PRODUTO", "VALOR_PRODUTO", "DATA"]]
        .drop_duplicates()
    )

    df_unitario = df_unitario.sort_values(["PRODUTO", "DATA"])

    produtos = df_unitario["PRODUTO"].unique()

    subtabelas = {
        produto: df_unitario[
            df_unitario["PRODUTO"] == produto
        ].sort_values("DATA")
        for produto in produtos
    }

    for produto, subtabela in subtabelas.items():

        if nome_produto.lower() in produto.lower():
            return subtabela

    return pd.DataFrame()