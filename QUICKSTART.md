# QUICKSTART - model_supermarket

## 1. Instalação

1. Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale as dependências:

```powershell
pip install -r requirements.txt
```

## 2. Preparar os dados

Coloque o arquivo de notas fiscais em `data/notas_fiscais.csv` ou informe outro caminho no painel.

O arquivo deve conter colunas como:
- `nota_fiscal_id`
- `supermercado`
- `codigo_produto`
- `produto`
- `quantidade`
- `preco_unitario`
- `preco_total`
- `data_hora`

## 3. Executar o painel interativo

```powershell
streamlit run src/dashboard/app.py
```

## 4. O que você verá

- Totais por dia, mês, ano e supermercado
- Produtos mais caros e mais vendidos
- Gasto médio por nota fiscal
- Análise por período do dia (manhã, tarde, noite)
- Modelos de Naive Bayes e regressão para interpretar comportamento de gasto
