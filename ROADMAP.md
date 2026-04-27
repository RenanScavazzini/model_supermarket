# ROADMAP - model_supermarket

## Fase 1: Painel Básico e Métricas Principais

- Carregar e normalizar dados de notas fiscais.
- Calcular totais gerais, por supermercado, por produto e por período do dia.
- Exibir métricas principais em painel interativo.
- Estruturar configuração em YAML.

## Fase 2: Modelos Estatísticos

- Treinar Naive Bayes para classificar notas de alto gasto.
- Treinar regressão para prever valor total da nota.
- Adicionar clusterização KMeans para segmentar notas fiscais.
- Gerar tabelas de coocorrência de produtos com frequência de compra conjunta.

## Fase 3: Expansão do Painel

- Adicionar análise temporal detalhada por dia, mês e ano.
- Suportar filtros interativos por supermercado, produto e período.
- Incluir previsões e recomendações no painel.
- Criar relatórios exportáveis em CSV ou PDF.

## Fase 4: Produção e Escalabilidade

- Suportar múltiplos arquivos de input e ingestão incremental.
- Adicionar banco de dados leve para histórico de notas fiscais.
- Construir API para consulta de métricas e modelos.
- Containerizar com Docker para implantação.
