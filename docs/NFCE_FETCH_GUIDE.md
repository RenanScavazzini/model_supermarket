# Sistema de Atualização de Dados via QR Codes

## Visão Geral

Este sistema permite reconstruir o dataset `notas_fiscais_supermercado.xlsx` do zero a partir dos QR Codes armazenados em `data/qrcodes.txt`, buscando dados reais no site da Fazenda.

## Processo em 3 Etapas

### 1. Processamento de QR Codes (`QRCodeProcessor`)
- **Entrada:** `data/qrcodes.txt` (URLs de QR Codes)
- **Processo:** 
  - Extrai chaves NFCE das URLs
  - Anonimiza cada chave com SHA256
  - Cria mapeamento original → anônimo
- **Saída:** `data/de_para_nfce.xlsx` (mapeamento atualizado)

### 2. Busca de Dados Reais (`NFCeFetcher`)
- **Entrada:** `data/qrcodes.txt` (URLs de QR Codes)
- **Processo:**
  - Faz requisições HTTP a cada URL
  - Parser do HTML da página de DANFE
  - Extrai: data/hora real, CNPJ, valor total, etc.
- **Saída:** `data/nfce_data_fetched.csv` (dados brutos)
- **Status:** ⚠️ Opcional - ativa apenas se `FETCH_REAL_DATA=True`

### 3. Consolidação de Dados
- **Entrada:**
  - `data/nfce_data_fetched.csv` (se disponível) ou `data/notas_fiscais_supermercado.xlsx` (fallback)
  - `data/de_para_nfce.xlsx` (mapeamento)
- **Processo:**
  - Merge de dados com URLs de QR Codes
  - Validação e limpeza
  - Formatação de campos
- **Saída:** `results/notas_fiscais_preparado.csv` (base final)

## Como Usar

### Modo 1: Usar Dados Existentes + QR Codes (Padrão)

```python
FETCH_REAL_DATA = False  # Desabilitado
```

- Usa dados existentes em `data/notas_fiscais_supermercado.xlsx`
- Adiciona URLs de QR Codes do mapeamento
- **Tempo:** ~30 segundos
- **Dados:** Os mesmos que já estão no arquivo

### Modo 2: Buscar Dados Reais do Site

```python
FETCH_REAL_DATA = True  # Habilitado
```

- Busca dados completos (data/hora, CNPJ, valor) de cada NFCe no site da Fazenda
- Reconstrói o dataset do zero com dados reais
- **Tempo:** 2-3 minutos (138 NFCes × 1 segundo por requisição)
- **Dados:** Informações reais e atualizadas de cada nota

## Campos Disponíveis

### Quando usando Modo 1 (Dados Existentes)
```
CHAVE (anonimizada)
DATA (dd/mm/aaaa)
PERIODO (Manhã/Tarde/Noite)
SUPERMERCADO
PRODUTO
QUANTIDADE
VALOR_UNITARIO
VALOR_TOTAL
qrcode_url (adicionado)
data_hora (reconstruída a partir de PERIODO)
```

### Quando usando Modo 2 (Dados Fetched)
```
nfce_chave (original, não anonimizada)
data_emissao (dd/mm/aaaa)
hora_emissao (hh:mm:ss - REAL)
data_hora (aaaa-mm-dd hh:mm:ss)
cnpj_estabelecimento
valor_total
```

## Limitações Conhecidas

1. **Hora Real (Modo 1):** 
   - Arquivo original apenas tem PERIODO (categórico)
   - Horas são estimadas: Manhã=09:00, Tarde=15:00, Noite=20:00
   - Para ter hora real, use Modo 2

2. **Velocidade (Modo 2):**
   - ~1 segundo por NFCe + overhead de rede
   - Total esperado: 2-3 minutos para 138 notas
   - Pode ser lento se conexão for ruim

3. **Disponibilidade:**
   - Depende da disponibilidade do site da Fazenda
   - Se site estiver indisponível, Modo 2 falhará

## Configuração

**Arquivo:** `config/config.yaml`

```yaml
data:
  raw_invoice_path: data/notas_fiscais_supermercado.xlsx
  qrcodes_path: data/qrcodes.txt
  qrcode_mapping_path: data/de_para_nfce.xlsx
```

## Execução Recomendada

1. **Primeira vez:**
   - Deixe `FETCH_REAL_DATA = False`
   - Execute para validar pipeline

2. **Para obter dados reais:**
   - Mude para `FETCH_REAL_DATA = True`
   - Execute (aguarde 2-3 minutos)
   - Resultado em `data/nfce_data_fetched.csv`

3. **Próximas execuções:**
   - Se quiser dados atualizados: `FETCH_REAL_DATA = True`
   - Se quiser execução rápida: `FETCH_REAL_DATA = False` (reutiliza dados anterior)

## Troubleshooting

**Erro: "Connection aborted"**
- Site recusando conexões
- Solução: Tentar novamente depois, site pode estar sob proteção

**Erro: "Nenhum dado foi obtido"**
- Parser não encontrou informações na página
- Solução: Verificar estrutura HTML, pode ter mudado

**Arquivo fetched vazio**
- Todas as requisições falharam ou não retornaram dados
- Solução: Verificar conectividade, tentar Modo 1

## Próximas Melhorias

- [ ] Cache de dados já fetched
- [ ] Retry automático com backoff exponencial
- [ ] Proxy/VPN suporte para evitar bloqueios
- [ ] Parser JSON alternativo se HTML mudar
- [ ] Integração com API oficial (se disponível)
