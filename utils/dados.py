import time
import pandas as pd
from functools import reduce
from nfceget import app
import hashlib
import openpyxl
import warnings

import pyspark.sql.types as T


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


def aplicar_alter_table(tabela_dict):
    """
    Descrição:
        Altera propriedades e comentários de colunas de uma tabela Delta no Databricks, conforme especificado em um dicionário de configuração.
    --------------------
    Argumentos:
        tabela_dict (dict): Dicionário contendo:
            - "tabela" (str): Nome da tabela a ser alterada.
            - "descricao" (str, opcional): Descrição da tabela.
            - "colunas" (list, opcional): Lista de dicionários com as chaves:
                - "nome" (str): Nome da coluna.
                - "tipo" (str): Tipo da coluna.
                - "descricao" (str, opcional): Comentário da coluna.
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
    tabela = tabela_dict["tabela"]
    descricao = tabela_dict.get("descricao", "")
    colunas = tabela_dict.get("colunas", [])

    if descricao:
        spark.sql(f"ALTER TABLE {tabela} SET TBLPROPERTIES ('comment' = '{descricao}')")

    for col in colunas:
        nome = col["nome"]
        tipo = col["tipo"]
        comentario = col.get("descricao", "")
        spark.sql(f"""
            ALTER TABLE {tabela}
            CHANGE COLUMN {nome} {nome} {tipo} COMMENT '{comentario}'
        """)