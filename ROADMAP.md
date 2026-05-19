# ROADMAP - model_supermarket

## Visão Geral

O projeto `model_supermarket` está evoluindo de um dashboard analítico de NFC-e para uma plataforma modular de análise comportamental, engenharia de dados, Machine Learning e Inteligência Artificial aplicada ao consumo.

---

# Versão Atual

## v2.0

### ✅ Implementado

- arquitetura modular
- dashboard Streamlit responsivo
- análise temporal
- análise de produtos
- KPIs analíticos
- normalização de dados
- logging centralizado
- configuração via YAML
- engenharia de features
- integração climática externa
- análises sazonais
- filtros dinâmicos
- gráficos interativos
- visualização mobile
- tema visual inspirado em Tensura

---

# Fase 1 — Consolidação Analítica

## Objetivos

Consolidar a camada analítica do dashboard e enriquecer o ecossistema de visualizações.

## Funcionalidades

### ✅ Implementado
- filtros globais
- análise por categoria
- dashboards comparativos
- análise temporal
- análise de sazonalidade
- clima histórico
- dia da semana
- feriados
- estação do ano

### 🚧 Em Desenvolvimento
- exportação CSV
- exportação Excel
- análise comparativa entre períodos
- análise climática avançada
- métricas financeiras pessoais
- inflação pessoal

---

# Fase 2 — Engenharia de Features

## Objetivos

Expandir o pipeline de feature engineering para suportar modelagem avançada e IA.

## Funcionalidades

### ✅ Implementado

#### Features internas
- período do dia
- dia da semana
- feriado
- estação do ano
- categorização de produtos
- padronização de produtos
- normalização de unidades

#### Features externas
- temperatura máxima
- temperatura mínima
- temperatura média
- precipitação
- categoria de temperatura
- indicador de dia chuvoso

### 🚧 Planejado

#### Calendário financeiro
- início/fim de mês
- período salarial
- quinzena
- dia útil

#### Feriados avançados
- véspera de feriado
- pós-feriado
- feriado prolongado

#### Features climáticas avançadas
- umidade
- vento
- pressão atmosférica

#### Outras integrações
- Google Trends
- commodities
- IPCA alimentos
- eventos esportivos
- sazonalidade comercial

---

# Fase 3 — Modelagem Estatística

## Objetivos

Adicionar modelos estatísticos, analíticos e preditivos.

## Funcionalidades

### 📈 Séries Temporais
- previsão de gastos
- previsão de ticket médio
- previsão por categoria
- forecast sazonal

### 🧠 Machine Learning
- regressão de gastos
- classificação de comportamento
- clusterização
- detecção de anomalias
- segmentação de consumo

### 🛒 Recommendation Systems
- recomendação de produtos
- recomendação de categorias
- recomendação contextual baseada em clima

---

# Fase 4 — Market Basket Analysis

## Objetivos

Implementar análise de associação e comportamento de compra.

## Funcionalidades

### 📦 Association Rules
- FP-Growth
- Apriori
- regras de associação
- lift
- confidence
- support

### 🧠 Análises Contextuais
- associações por clima
- associações por estação
- associações por supermercado
- associações por período do dia

### 📊 Visualizações
- redes de produtos
- grafos
- heatmaps
- embeddings de produtos

---

# Fase 5 — Inteligência Artificial

## Objetivos

Adicionar IA Generativa e análise automática de comportamento.

## Funcionalidades

### 🤖 LLMs
- geração automática de insights
- análises textuais automáticas
- resumo inteligente de consumo
- interpretação estatística automática

### 🧠 IA Analítica
- explicabilidade de modelos
- detecção automática de padrões
- geração automática de hipóteses
- interpretação comportamental

### 💬 Assistente Inteligente
- chatbot analítico
- perguntas em linguagem natural
- exploração conversacional da base

---

# Fase 6 — Engenharia de Dados

## Objetivos

Evoluir pipeline de dados e persistência.

## Funcionalidades

- ingestão incremental
- múltiplos arquivos
- automação de pipeline
- versionamento de datasets
- validação de schema
- armazenamento histórico
- banco de dados analítico
- parquet lakehouse

---

# Fase 7 — Deploy e Escalabilidade

## Objetivos

Preparar projeto para ambiente produtivo e cloud analytics.

## Funcionalidades

### ☁️ Infraestrutura
- Streamlit Cloud
- Docker
- CI/CD
- observabilidade
- monitoramento

### 🔌 APIs
- API REST
- endpoints analíticos
- serving de modelos

### 🔐 Segurança
- autenticação
- controle de acesso
- auditoria

---

# Fase 8 — Expansão Analítica

## Objetivos

Transformar o projeto em uma plataforma analítica completa de comportamento de consumo.

## Funcionalidades

### 🧾 Consumo Pessoal
- inflação pessoal
- análise financeira
- metas de gastos
- orçamento inteligente

### 🏪 Retail Analytics
- análise de pricing
- elasticidade de consumo
- sensibilidade climática
- comportamento sazonal

### 🌐 IA + Analytics
- embeddings
- vector search
- semantic analytics
- recommendation engine

---

# Roadmap Técnico

## Pipeline Atual

```text
NFC-e
→ ingestão
→ tratamento
→ feature engineering
→ clima externo
→ dashboard
→ modelagem
→ IA
```