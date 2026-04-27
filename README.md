# model_supermarket

**Versão**: 1.0

## Descrição

Este projeto implementa um painel interativo para análise de notas fiscais de supermercados, inspirado na arquitetura do model_car. O objetivo é extrair métricas de gasto, tendências por período, supermercado, produto e código de produto, além de aplicar modelos estatísticos e de segmentação.

## Funcionalidades

- Leitura e normalização de notas fiscais
- Totais por dia, mês, ano, supermercado e produto
- Análise por período do dia: manhã, tarde e noite
- Identificação de maiores e menores gastos
- Cálculo de gasto médio por nota fiscal
- Modelos estatísticos: Naive Bayes, regressão e clusterização
- Painel interativo com gráficos e métricas

## Estrutura do Projeto

```
model_supermarket/
├── config/
│   └── config.yaml
├── src/
│   ├── core/
│   │   ├── config_loader.py
│   │   └── logger.py
│   ├── analysis/
│   │   ├── invoice_loader.py
│   │   ├── summary_analyzer.py
│   │   └── statistical_models.py
│   └── dashboard/
│       └── app.py
├── data/
│   └── notas_fiscais.csv
├── notebooks/
│   └── 01_base_dados.ipynb
├── results/
│   └── logs/
├── requirements.txt
├── QUICKSTART.md
├── ARCHITECTURE.md
├── ROADMAP.md
└── README.md
```

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como usar

1. Ajuste o caminho do arquivo em config/config.yaml ou use o painel.
2. Execute o painel interativo:

```powershell
streamlit run src/dashboard/app.py
```

3. Explore os gráficos de gasto, os relatórios por período e as métricas de produto.

## Configuração

A configuração principal está em config/config.yaml e permite personalizar:
- caminhos de dados
- colunas de nota fiscal
- parâmetros de modelo

## Observações

O painel foi desenvolvido para ser adaptado a diferentes formatos de notas fiscais. Caso haja variações nas colunas de origem, basta ajustar os nomes em config/config.yaml.
