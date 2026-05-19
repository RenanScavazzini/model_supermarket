# Arquitetura do Projeto `model_supermarket`

## Visão Geral

O projeto `model_supermarket` adota uma arquitetura modular orientada a engenharia de dados, análise estatística, visualização interativa e futura integração com Inteligência Artificial e Machine Learning.

A estrutura foi desenvolvida seguindo princípios de:

- separação de responsabilidades
- modularização
- reutilização de componentes
- escalabilidade analítica
- extensibilidade para IA
- pipeline de feature engineering
- integração com fontes externas

O sistema suporta:

- ingestão e tratamento de notas fiscais NFC-e
- engenharia de atributos internos e externos
- exploração analítica e estatística
- geração de métricas e agregações
- visualização interativa via dashboard
- integração com APIs externas
- futura expansão para Machine Learning e LLMs
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
│   └── config.yaml
│
├── data/
│   │
│   ├── external/
│   │   └── weather/
│   │       └── weather_curitiba.parquet
│   │
│   ├── de_para_nfce.csv
│   ├── nfce_data_fetched.csv
│   ├── notas_fiscais_supermercado.xlsx
│   └── qrcodes.txt
│
├── docs/
│
├── image/
│   ├── background.png
│   └── ui/
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
│   │       ├── statistical_model.py
│   │       └── temporal.py
│   │
│   ├── external/
│   │   └── weather.py
│   │
│   └── utils/
│       ├── constants.py
│       ├── formatters.py
│       └── helpers.py
│
├── 01_dados.ipynb
├── 02_eda.ipynb
├── 03_modelagem.ipynb
├── 04_dashboard.ipynb
│
├── ARCHITECTURE.md
├── README.md
├── ROADMAP.md
│
├── requirements.txt
└── requirements-dev.txt
```