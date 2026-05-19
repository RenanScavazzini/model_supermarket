# model_supermarket

Dashboard analítico e framework modular para análise de notas fiscais de supermercados utilizando Python, Pandas, Streamlit, Estatística, Machine Learning e IA Generativa.

---

## Objetivo

O projeto tem como objetivo transformar dados de NFC-e em informações analíticas, visualizações interativas e insights inteligentes, permitindo:

- análise de gastos
- monitoramento temporal
- comparação entre supermercados
- análise de produtos
- exploração estatística
- engenharia de features
- integração com dados externos
- futura aplicação de modelos preditivos
- análises baseadas em IA e LLMs

---

## Principais Funcionalidades

- leitura e normalização de notas fiscais
- análises temporais
- KPIs interativos
- dashboard web responsivo
- análise de produtos
- histórico de preços
- agregações por categoria
- comparativos entre supermercados
- filtros dinâmicos
- visualização completa da base
- engenharia de variáveis
- integração com dados climáticos
- modelos estatísticos
- exportação de dados
- arquitetura modular para IA e Machine Learning

---

## Dashboard

O dashboard interativo possui as seguintes páginas:

### 📊 Overview
- KPIs gerais
- gastos por supermercado
- gastos por período
- gastos por categoria
- gastos por dia da semana
- gastos em feriados
- gastos por estação do ano
- gastos em dias chuvosos
- gastos por categoria de temperatura
- filtros analíticos

### 📅 Temporal
- evolução anual
- evolução mensal
- evolução diária
- análises temporais
- análises sazonais

### 🛒 Products
- pesquisa por produto
- pesquisa por código
- histórico de preços
- resumo analítico
- filtros múltiplos

### 🗄️ Database
- visualização completa da base
- filtros dinâmicos por coluna
- exploração analítica detalhada

### 📈 Modelo Estatístico
- área dedicada à modelagem estatística
- futuras análises preditivas
- integração com IA
- análises de Market Basket
- modelos de Machine Learning

---

## Engenharia de Features

O projeto possui pipeline de feature engineering para enriquecimento analítico da base.

### Features Internas

- período do dia
- dia da semana
- feriado
- estação do ano
- categoria de produto
- padronização de produtos
- normalização de unidades

### Features Externas

Integração com dados climáticos históricos via Open-Meteo:

- temperatura máxima
- temperatura mínima
- temperatura média
- volume de chuva
- categoria de temperatura
- indicador de dia chuvoso

---

## Inteligência Artificial e Modelagem

O projeto está sendo estruturado para incorporar:

### 📦 Market Basket Analysis
- FP-Growth
- Apriori
- regras de associação
- análise de produtos comprados juntos

### 🤖 IA Generativa
- insights automáticos
- geração de análises textuais
- integração com LLMs
- resumos inteligentes de consumo

### 📈 Modelagem Estatística
- previsão de gastos
- séries temporais
- clustering
- detecção de anomalias
- análise de comportamento de consumo

---

## Tecnologias

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn
- PyYAML
- OpenPyXL
- Plotly
- Holidays
- Requests
- PyArrow

---

## Fontes Externas

### 🌦️ Open-Meteo
Utilizado para enriquecimento climático da base.

- temperatura histórica
- precipitação
- features sazonais

---

## Estrutura do Projeto

```text
model_supermarket/
│
├── config/
├── data/
│   └── external/
│       └── weather/
├── docs/
├── image/
├── logs/
├── results/
├── src/
│   ├── analysis/
│   ├── core/
│   ├── dashboard/
│   ├── external/
│   │   └── weather.py
│   └── utils/
│
├── 01_dados.ipynb
├── 02_eda.ipynb
├── 03_modelagem.ipynb
├── 04_dashboard.ipynb
│
├── README.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── requirements.txt
└── requirements-dev.txt
```

---

## Autor

| Autor | GitHub | LinkedIn | Email |
|-------|--------|----------|-------|
| Renan Douglas Floriano Scavazzini | [@RenanScavazzini](https://github.com/RenanScavazzini) | [renan-scavazzini](https://www.linkedin.com/in/renan-scavazzini/) | renanscavazzini@gmail.com 