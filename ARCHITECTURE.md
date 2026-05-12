# Arquitetura do Projeto `model_supermarket`

## Visão Geral

O projeto `model_supermarket` adota uma arquitetura modular orientada a análise de dados, engenharia analítica e visualização interativa, seguindo princípios de separação de responsabilidades, reutilização de componentes e escalabilidade.

A estrutura foi desenvolvida para suportar:

- ingestão e tratamento de notas fiscais NFC-e
- exploração analítica e estatística
- geração de métricas e agregações
- visualização interativa via dashboard
- futura expansão para modelos de Machine Learning
- deploy web utilizando Streamlit Cloud

---

## Estrutura do Projeto

```text
model_supermarket/
│
├── config/
│   └── settings.yaml
│
├── data/
│   └── notas_fiscais_supermercado.xlsx
│
├── logs/
│   └── app.log
│
├── notebooks/
│   ├── 01_dados.ipynb
│   └── 02_eda.ipynb
│
├── src/
│
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── logger.py
|   |   ├── nfce_fetcher.py
|   |   ├── paths.py
│   │   └── tictoc.py
│
│   ├── analysis/
│   │   ├── invoice_loader.py
│   │   ├── summary_analyzer.py
│   │   ├── product_analyzer.py
│   │   ├── temporal_analyzer.py
│   │   └── statistical_models.py
│
│   ├── dashboard/
│   │   ├── app.py
│   │   │
│   │   ├── pages/
│   │   │   ├── overview.py
│   │   │   ├── products.py
│   │   │   ├── temporal.py
│   │   │   └── supermarkets.py
│   │   │
│   │   ├── components/
│   │   │   ├── filters.py
│   │   │   ├── metrics.py
│   │   │   └── charts.py
│   │   │
│   │   └── assets/
│   │       └── logo.png
│
│   └── utils/
│       ├── helpers.py
│       └── constants.py
│
├── requirements.txt
├── requirements-dev.txt
├── ARCHITECTURE.md
├── README.md
├── ROADMAP.md
│
└── .streamlit/
    └── config.toml