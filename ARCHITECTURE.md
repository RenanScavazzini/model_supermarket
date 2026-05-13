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
├── .streamlit/
│   └── config.toml
│
├── .venv/
│
├── config/
│   └── settings.yaml
│
├── data/
│   └── notas_fiscais_supermercado.xlsx
│
├── docs/
│
├── image/
│   └── logo.png
│
├── logs/
│   └── app.log
│
├── results/
│
├── src/
│
│   ├── analysis/
│   │   ├── invoice_loader.py
│   │   ├── product_analyzer.py
│   │   ├── statistical_models.py
│   │   ├── summary_analyzer.py
│   │   └── temporal_analyzer.py
│   │
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── nfce_fetcher.py
│   │   ├── paths.py
│   │   └── tictoc.py
│   │
│   ├── dashboard/
│   │   ├── app.py
│   │   │
│   │   ├── components/
│   │   │   ├── charts.py
│   │   │   ├── filters.py
│   │   │   └── metrics.py
│   │   │
│   │   └── views/
│   │       ├── database.py
│   │       ├── overview.py
│   │       ├── products.py
│   │       └── temporal.py
│   │
│   └── utils/
│       ├── constants.py
│       ├── formatters.py
│       └── helpers.py
│
├── 01_dados.ipynb
├── 02_eda.ipynb
├── 03_dashboard.ipynb
│
├── ARCHITECTURE.md
├── README.md
├── ROADMAP.md
│
├── requirements.txt
└── requirements-dev.txt
```