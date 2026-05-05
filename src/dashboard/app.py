"""
Descrição:
    Módulo responsável pela interface interativa do projeto utilizando Streamlit,
    permitindo visualização de métricas, gráficos e resultados de modelos estatísticos
    aplicados a dados de notas fiscais de supermercado.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 29/04/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.core.config_loader import ConfigLoader
from src.analysis.invoice_loader import InvoiceLoader
from src.analysis.summary_analyzer import SummaryAnalyzer
from src.analysis.statistical_models import StatisticalModels


def main():
    """
    Descrição:
        Função principal responsável por inicializar o dashboard, carregar dados
        e orquestrar a exibição de métricas, gráficos e modelos.

    Parâmetros:
        ---

    Referências:
        ---
        
    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    config = ConfigLoader().config
    st.set_page_config(page_title=config.get('dashboard.title', 'Painel Supermercado'), layout='wide')

    st.title(config.get('dashboard.title', 'Painel Interativo de Supermercado'))
    st.markdown('Analisa notas fiscais para gerar métricas de gastos, comportamento de compra e modelos estatísticos.')

    input_path = st.sidebar.text_input('Caminho do arquivo de notas fiscais', config.get('data.raw_invoice_path', 'data/notas_fiscais.csv'))
    if st.sidebar.button('Carregar dados') or input_path:
        try:
            loader = InvoiceLoader(config)
            df = loader.load_and_prepare(input_path)

            st.sidebar.success('Dados carregados com sucesso!')
            display_metrics(df)
            display_charts(df)
            display_models(df)
        except Exception as error:
            st.sidebar.error(f'Erro ao carregar dados: {error}')


def display_metrics(df: pd.DataFrame) -> None:
    """
    Descrição:
        Exibe os principais indicadores agregados de gasto e volume de notas fiscais.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados processados.

    Referências:
        ---
        
    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    report = SummaryAnalyzer.gerenciar_resumo(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total de Gasto', f'R$ {report.total_gasto:,.2f}')
    col2.metric('Total de Notas', report.total_notas)
    col3.metric('Gasto Médio por Nota', f'R$ {report.gasto_medio:,.2f}')
    col4.metric('Maior Nota', f'R$ {report.maior_gasto:,.2f}')

    with st.expander('Resumo de agregações'):
        st.write('Total por período do dia')
        st.dataframe(report.agrupamento['por_periodo'].reset_index(drop=True))
        st.write('Total por supermercado')
        st.dataframe(report.agrupamento['por_supermercado'].reset_index(drop=True))
        st.write('Top produtos por gasto')
        st.dataframe(report.agrupamento['por_produto'].head(10).reset_index(drop=True))


def display_charts(df: pd.DataFrame) -> None:
    """
    Descrição:
        Gera e exibe gráficos interativos com base nas agregações dos dados.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados processados.

    Referências:
        - Tufte, E. (2001). The Visual Display of Quantitative Information.
        
    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    df_periodo = SummaryAnalyzer.total_por_periodo(df)
    df_supermercado = SummaryAnalyzer.total_por_supermercado(df)
    df_produto = SummaryAnalyzer.total_por_produto(df).head(15)

    st.subheader('Visualizações Principais')
    fig_periodo = px.bar(df_periodo, x='periodo_dia', y='preco_total')
    st.plotly_chart(fig_periodo, use_container_width=True)

    fig_supermercado = px.bar(df_supermercado, x='supermercado', y='preco_total')
    st.plotly_chart(fig_supermercado, use_container_width=True)

    fig_produto = px.bar(df_produto, x='produto', y='preco_total')
    st.plotly_chart(fig_produto, use_container_width=True)


def display_models(df: pd.DataFrame) -> None:
    """
    Descrição:
        Executa e exibe resultados de modelos estatísticos aplicados aos dados.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados processados.

    Referências:
        - Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning.
        
    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    df_flag = StatisticalModels.build_high_spend_flag(df)
    result_nb = StatisticalModels.train_naive_bayes(df_flag)
    result_rf = StatisticalModels.train_spend_regressor(df)
    result_kmeans = StatisticalModels.fit_kmeans(df)

    st.subheader('Modelos Estatísticos')
    st.write('**Naive Bayes de Gasto Alto**')
    st.write('Acurácia:', f'{result_nb.metrics.get("acuracia", 0):.2f}')
    st.write('**Regressão de Gasto Total**')
    st.write('MSE:', f'{result_rf.metrics.get("mse", 0):.2f}')
    st.write('**Clusterização de Notas**')
    st.write('Inércia:', f'{result_kmeans.metrics.get("inercia", 0):.2f}')

    st.write('Este painel aplica modelos de classificação e regressão para estimar comportamento de gastos e identificar padrões de notas fiscais.')


if __name__ == '__main__':
    main()