# Arquitetura do Projeto model_supermarket

Este projeto adota uma arquitetura modular inspirada no framework do `model_car`.

## Estrutura principal

- `config/` : Configurações YAML para caminhos, colunas e modelo.
- `src/core/` : Funções de infraestrutura, como carregamento de configuração e logging.
- `src/analysis/` : Processamento de notas fiscais, agregação de métricas e modelos estatísticos.
- `src/dashboard/` : Aplicação interativa de dashboard em Streamlit.
- `notebooks/` : Notebooks de exploração de dados e demonstração.

## Fluxo de dados

1. Carregar notas fiscais raw a partir de `data/notas_fiscais.csv` ou outro arquivo.
2. Normalizar colunas essenciais e garantir campos de tempo.
3. Calcular métricas de gasto total, médio, máximos e agregações por período, supermercado e produto.
4. Aplicar modelos estatísticos:
   - Naive Bayes para classificação de notas de alto gasto.
   - Regressão para previsão de valor total.
   - KMeans para clusterização de notas fiscais.
5. Exibir resultados em um painel interativo com gráficos e métricas.

## Componentes principais

- `ConfigLoader` : gerencia a configuração YAML e permite leitura por chave pontuada.
- `InvoiceLoader` : carrega e prepara registros de nota fiscal para análise.
- `SummaryAnalyzer` : gera indicadores essenciais e tabelas de agregação.
- `StatisticalModels` : treina e avalia modelos baseados em comportamento de gasto.
- `Streamlit app` : painel interativo para explorar os principais relatórios e modelos.
