import time
import hashlib
import pandas as pd
from pathlib import Path
from nfceget import app
import unicodedata


def run_nfce_fetch(
    fetch_enabled: bool,
    delay: float,
    qrcodes_path: Path,
    fetched_path: Path,
    mapping_path: Path,
) -> pd.DataFrame:
    """
    Descrição:
        Módulo responsável pelo processamento de QR Codes de NFC-e, incluindo a
        extração de dados das notas, normalização e validação de campos,
        anonimização implícita por não persistir chaves sensíveis, e geração
        de um DataFrame consolidado com os itens das notas.

    Parâmetros:
        fetch_enabled (bool): Habilita ou não a busca remota das NFC-e. Se
            False, a função tentará ler o CSV em fetched_path e retornará seu
            conteúdo.
        qrcodes_path (Path): Caminho para o arquivo contendo URLs/QR codes, um
            por linha.
        fetched_path (Path): Caminho do CSV onde os resultados serão gravados
            e/ou lidos quando fetch_enabled for False.
        mapping_path (Path | None): Caminho do CSV onde será gravado o mapa de
            anonimização (de_para_nfce.csv)
        delay (float): Tempo em segundos para aguardar entre requisições.

    Retorno:
        pd.DataFrame: DataFrame contendo os itens das NFC-e processadas ou um
        DataFrame vazio caso não haja dados válidos.

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    print(f"Configuração de busca (nfceget): FETCH_REAL_DATA={fetch_enabled}")
    # 🔹 Caso não faça fetch, apenas lê o CSV
    if not fetch_enabled:
        print(f"ℹ Lendo CSV existente: {fetched_path}")
        if fetched_path.exists():
            return pd.read_csv(fetched_path)
        return pd.DataFrame()
    # 🔹 Validação de entrada
    if not qrcodes_path.exists():
        print("⚠ QR Codes não encontrados")
        return pd.DataFrame()
    # 🔹 Leitura dos QR codes
    with open(qrcodes_path, 'r') as f:
        qrcodes = [linha.strip() for linha in f if linha.strip()]
    # 🔹 Remover duplicados preservando ordem
    qrcodes = list(dict.fromkeys(qrcodes))
    print(f"ℹ QR Codes únicos: {len(qrcodes)}")
    dataframes = []
    for i, url in enumerate(qrcodes, 1):
        print(f"[{i}/{len(qrcodes)}] Buscando...", end=' ')
        try:
            nota = app.json_from_qrcode_link(url)
            print("✅ Nota obtida")
            if not nota or 'itens' not in nota or not nota['itens']:
                print(f"⚠ Nota sem itens: {url}")
                continue
            df_nota = pd.DataFrame(nota['itens'])
            if df_nota.empty:
                print(f"⚠ DataFrame vazio: {url}")
                continue
            # 🔹 Garantir colunas mínimas
            for col in ['name', 'code', 'quantity', 'unit', 'unitaryValue', 'totalValue']:
                if col not in df_nota.columns:
                    df_nota[col] = None
            # 🔹 Rename direto para schema final
            df_nota = df_nota.rename(columns={
                'name': 'PRODUTO',
                'code': 'COD_PRODUTO',
                'quantity': 'QTDE',
                'unit': 'UNIDADE',
                'unitaryValue': 'VALOR_PRODUTO',
                'totalValue': 'VALOR_TOTAL_PRODUTO'
            })
            # 🔹 Conversões numéricas
            df_nota['QTDE'] = pd.to_numeric(df_nota['QTDE'], errors='coerce')
            df_nota['VALOR_PRODUTO'] = (
                df_nota['VALOR_PRODUTO']
                .astype(str)
                .str.replace(',', '.', regex=False)
            )
            df_nota['VALOR_PRODUTO'] = pd.to_numeric(df_nota['VALOR_PRODUTO'], errors='coerce')
            df_nota['VALOR_TOTAL_PRODUTO'] = (
                df_nota['VALOR_TOTAL_PRODUTO']
                .astype(str)
                .str.replace(',', '.', regex=False)
            )
            df_nota['VALOR_TOTAL_PRODUTO'] = pd.to_numeric(df_nota['VALOR_TOTAL_PRODUTO'], errors='coerce')
            # 🔹 Validações
            if df_nota['VALOR_PRODUTO'].isna().any():
                print(f"⚠ Preço unitário inválido: {url}")
                df_nota = df_nota.dropna(subset=['VALOR_PRODUTO'])

            if df_nota['VALOR_TOTAL_PRODUTO'].isna().any():
                print(f"⚠ Preço total inválido: {url}")
                df_nota = df_nota.dropna(subset=['VALOR_TOTAL_PRODUTO'])
            if df_nota.empty:
                print(f"⚠ Nenhum item válido: {url}")
                continue
            # 🔹 Dados da nota
            local = nota.get('local') or {}
            df_nota['SUPERMERCADO'] = local.get('name')
            df_nota['CNPJ'] = local.get('cnpj')
            nfce = nota.get('nfce') or {}
            totals = nota.get('totals') or {}
            df_nota['DATA'] = nfce.get('date') or totals.get('date')
            # 🔹 Quantidade total
            try:
                quantidade_total = totals.get('quantityItens')
                if quantidade_total is not None:
                    quantidade_total = int(str(quantidade_total))
                else:
                    quantidade_total = int(df_nota['QTDE'].sum())
            except Exception:
                quantidade_total = int(df_nota['QTDE'].sum())
            df_nota['QTDE_TOTAL_NOTA'] = quantidade_total
            # 🔹 Valores
            valor_total = totals.get('valueToPay') or nfce.get('total')
            if isinstance(valor_total, str):
                valor_total = valor_total.replace(',', '.')
            df_nota['VALOR_TOTAL_NOTA'] = pd.to_numeric(valor_total, errors='coerce')
            valor_tributos = totals.get('taxes') or nfce.get('taxes')
            if isinstance(valor_tributos, str):
                valor_tributos = valor_tributos.replace(',', '.')
            df_nota['VALOR_TOTAL_TRIBUTOS'] = pd.to_numeric(valor_tributos, errors='coerce')
            df_nota['CHAVE_ANONIMIZADA'] = nfce.get('chave')
            dataframes.append(df_nota)
            time.sleep(delay)
        except Exception as e:
            print(f"❌ Erro ao processar: {url}")
            print(e)
    # 🔹 Sem dados
    if not dataframes:
        print("⚠ Nenhuma nota válida encontrada")
        return pd.DataFrame()
    df = pd.concat(dataframes, ignore_index=True)
    # 🔹 Garantir diretório
    fetched_path.parent.mkdir(parents=True, exist_ok=True)
    # 🔹 Anonimização
    orig_chaves = df['CHAVE_ANONIMIZADA'].dropna().unique() if 'CHAVE_ANONIMIZADA' in df.columns else []
    map_rows = []
    for chave in orig_chaves:
        anon = hashlib.sha256(str(chave).encode('utf-8')).hexdigest()
        map_rows.append({
            'nfce_chave_original': chave,
            'nfce_chave_anon': anon
        })
    if map_rows:
        df_map = pd.DataFrame(map_rows)
        df_map.to_csv(mapping_path, index=False)
        map_dict = dict(zip(
            df_map['nfce_chave_original'],
            df_map['nfce_chave_anon']
        ))
        df['CHAVE_ANONIMIZADA'] = df['CHAVE_ANONIMIZADA'].map(
            lambda x: map_dict.get(x) if pd.notna(x) else x
        )
    else:
        pd.DataFrame().to_csv(mapping_path, index=False)
    # 🔹 Ordem final das colunas
    colunas_finais = [
        'CHAVE_ANONIMIZADA',
        'DATA',
        'CNPJ',
        'SUPERMERCADO',
        'QTDE_TOTAL_NOTA',
        'VALOR_TOTAL_NOTA',
        'VALOR_TOTAL_TRIBUTOS',
        'COD_PRODUTO',
        'PRODUTO',
        'UNIDADE',
        'QTDE',
        'VALOR_PRODUTO',
        'VALOR_TOTAL_PRODUTO',
    ]
    df = df[colunas_finais]
    # 🔹 Salvar dataset final
    df.to_csv(fetched_path, index=False)
    print("✅ DataFrame exportado para CSV:", fetched_path)
    return df


def process_base(
    fetch_enabled: bool,
    df_fetched: pd.DataFrame, 
    raw_path: Path
) -> pd.DataFrame:
    """
    Descrição:
        Módulo responsável pelo processamento e enriquecimento da base de NFC-e,
        incluindo filtragem, padronização, engenharia de atributos e classificação
        de produtos.

    Parâmetros:
        fetch_enabled (bool): Indica se a base deve ser processada (True) ou se
            deve apenas ser carregada do arquivo final já tratado (False).
        df_fetched (pd.DataFrame): DataFrame contendo os dados brutos processados
            anteriormente (run_nfce_fetch).
        raw_path (Path): Caminho do arquivo .xlsx onde a base tratada será salva
            ou lida caso fetch_enabled seja False.

    Retorno:
        pd.DataFrame

    Referências:
        ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    print(f"Configuração de busca (nfceget): FETCH_REAL_DATA={fetch_enabled}")
    # 🔹 MODO LEITURA (sem processamento)
    if not fetch_enabled:
        print(f"ℹ Lendo base já processada: {raw_path}")
        if raw_path.exists():
            return pd.read_excel(raw_path)
        print("⚠ Arquivo não encontrado")
        return pd.DataFrame()
    # 🔹 MODO PROCESSAMENTO
    df = df_fetched.copy()
    # 🔹 1. FILTRO CNPJ + NORMALIZAÇÃO SUPERMERCADO
    cnpj_map = {
        "76.189.406/0034-94": "CONDOR",
        "76.430.438/0070-01": "MAX",
        "06.057.223/0367-96": "ASSAI",
    }
    df = df[df["CNPJ"].isin(cnpj_map.keys())].copy()
    print("✅ CNPJ filtrados:", df["CNPJ"].nunique())
    df["SUPERMERCADO"] = df["CNPJ"].map(cnpj_map)
    print("✅ Supermercados normalizados:", df["SUPERMERCADO"].nunique())
    # 🔹 2. DATA → MES_ANO + PERIODO
    df["DATA"] = pd.to_datetime(
        df["DATA"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )
    df["MES_ANO"] = df["DATA"].dt.strftime("%Y%m")
    print("✅ Datas convertidas, meses/anos únicos:", df["MES_ANO"].nunique())
    def classificar_periodo(dt):
        h = dt.hour
        if 0 <= h < 5:
            return "MADRUGADA"
        elif 5 <= h < 12:
            return "MANHA"
        elif 12 <= h < 18:
            return "TARDE"
        else:
            return "NOITE"
    df["PERIODO"] = df["DATA"].apply(classificar_periodo)
    print("✅ Períodos classificados:", df["PERIODO"].nunique())
    # 🔹 3. PADRONIZAÇÃO PRODUTO
    df = df.sort_values("DATA")
    produto_map = (
        df.groupby("COD_PRODUTO")
        .last()["PRODUTO"]
        .to_dict()
    )
    df["PRODUTO"] = df["COD_PRODUTO"].map(produto_map)
    print("✅ Produtos padronizados:", df["PRODUTO"].nunique())
    # 🔹 4. CATEGORIZAÇÃO
    def remover_acentos(texto):
        return ''.join(
            c for c in unicodedata.normalize('NFD', str(texto))
            if unicodedata.category(c) != 'Mn'
        )
    categoria_produtos = {
        "mercearia": ["ARROZ","FEIJAO","FARINHA","TRIGO","ARROZ INTEGRAL","FEIJAO PRETO","FEIJ","ACUCAR","ACUC","SUGAR","OVO","OLEO","COAMO","FAROFA","YOKI","ARR","FARO"],
        "massa": ["MAC","MASSA","LASANHA","NISSIN","ESPAGUETE","PENNE","PINDUCA","TALHARIM","FETTUCCINE","GNOCCHI","RAVIOLES","MAC INTEGRAL","CONCHINHA","MAC GRANO","MASSA FRESCA","LAS PERD"],
        "cereal": ["CEREAL","NESTLE","NESCAU","SUCRILHOS","AVEIA","FLOCOS","KELLOGGS","GRANOLA","MUESLI","BARRA","CEREAIS INTEGRAL","FLOCOS MILHO","CEREAL MAT","BARRA DE CEREAL","AVEIA INSTANT","SNOW FLAKES"],
        "bebida nao alcoolica": ["BEB","FANTA","AG COC","COCA","SUCO","REFRI","ENERGETICO","AGUA","REFRIG","GATORADE","ADES","CRYSTAL","MONSTER","DELV","NATURAL O","LIFE","SUCOS","TEA","VITAMINA","MAIS","SOYA","TANG","GUARANA","CAFE","CHA","LEITE","SUCO DE FRUTA","REFRIG LITE","ENERG","PRATSY","REF","SCHWEPPES","SCHWEP","CITRUS","DEL VALLE","VALLE","YOPRO","PURITY","GELADINH","CRISTAL"],
        "bebida alcoolica": ["VODKA","CHAMPAGNE","CERVEJA","SKOL","HEINEKEN","BUDWEISER","SPATEN","PETRA","BRAHMA","STELLA","ESTRELLA","VINHO","ICE","PERIQUIT","SMIRNOFF","BEATS","PASSPORT","WHISKY","COROT","CAMPO","SUAVE","GIN","SAKE","TEQUILA","CERVEJA ARTESANAL","VINHO TINTO","VINHO BRANCO","WHISK","JACK DANIEL","CERV"],
        "padaria e frios": ["MORTADELA","PRESUNTO","QUEIJO","SALAME","PEPPA","MORTADELA FRANGO","FRIO VARIADOS","QUEIJO RALADO","PAO","BOLINHO","WAFER","BOLO","MUFFIN","BAUDUCCO","MARILAN","VISCONTI","BAGUETE","PAO INTEGRAL","PAO QUEIJO","PAO DOCE","PAO FRANCES","BOLO RECHEADO","PRES","AURORA","P FORMA","QJO","MUSS","MORT"],
        "congelado": ["PIZZA","SALSICHA","SORVETE","GELADO","CONGELADO","LASANHA","EMPANADO","SEARA","FRIMESA","ICE CREAM","BATATA FRITA","FRUTAS CONGELADAS","REFEICAO PRONTA","PEIXE CONGELADO","SALSICHA CONGELADA","TEKITOS","PERDIG","SALS"],
        "acougue": ["FRANGO","PEITO","FILE","BIFES","CARNES","BACON","CORTES","CARNE SUINA","COXA","LINGUICA","PRESUNTO","SALSICHA","PERDIGAO","SEARA","FRIMESA","CARNE BOV","ALCATRA","CONTRA FILE","COSTELA","COSTELA SUINA","CARNE MOIDA","BIF ACEBOLA","MEIO ASA","SADIA","FRALDINHA","COX MOLE","FRIBOI","FRIB","ACEM MO"],
        "biscoito e snack": ["BISCOITO","CHIPS","SNACK","FANDANGOS","CHEETOS","PRINGLES","WAFER","BOLACHA","BOLINHO","BISCOITOS INTEGRAL","SNACK SALG","BATATA PALHA","BOLINHO ARROZ","SNACK DOCE","SALGADOS","ELMA","SALG","PALHA","TORR","BISC"],
        "bomboniere": ["BALA","CHOCOLATE","BOMBOM","KINDER","RAFFAELLO","TRIDENT","HALLS","DORI","BALAS GOMA","PIPOCA DOCE","CONFETE","CHOC BRANCO","LACTA","NUTELLA","PIRULITO","GELATINA"],
        "laticinio": ["LEITE","QUEIJO","IOGURTE","CREME","MANTEIGA","REQUEIJAO","ACHOCOLATADO","NATURAL","PRESIDENT","TIROL","BATAVO","ELEGE","MUSSARELA","CREME LEITE","REFRIGERADO","UHT","REQUEIJAO CREMOSO","QUEIJO CREMOSO","IOGUR","CREM LTE","CREM LEI","LEITE COND","CR LEI","ACHOC","CHOCO MILK","IOG","LIDER","DANONINHO","CHOC","CHOCOMILK","BATAV","MOOCA"],
        "molho": ["MOLHO","KETCHUP","CATCHUP","MAIONESE","TEMPERO","MOSTARDA","SAZON","COND","HEMMER","HELLMANN","QUERO","POMAROLA","PICKLES","CREME CEBOLA","SALSA","MOLHO TARATAR","MOLHO PIMENTA","ELEFANT","CATCH","ELEF","MAION","MOST","HELLM","HELL"],
        "lataria e conserva": ["MILHO","ERVILHA","LATA","CONSERVA","SARDINHA","ATUM","PALMITO","EXTRATO","BONARE","FUGINI","SELETA","AZEITONA","BERINJELA","MILHO EM CONSERVA","ERVILHA EM LATA"],
        "condimento": ["ALHO","SAZON","TEMPERO","TEMP","PIMENTA","AZEITE","MANJER","CHIMICHURRI","SAL","KITANO","KININO","LOURO","CHIMICH","CHEIRO","TRIANGULO","OREGANO"],
        "limpeza e limpeza": ["LIMP","DETERG","SABAO","OMO","LYSOL","VEJA","DESINF","PINHO","SANIT","AJAX","UOL","ALVEJANTE","MULTIUSO","ESPONJA","DESENGORD","PAP H","SHAMPOO","COND","DESOD","SABONETE"],
        "hortifruti": ["CEBOLA","TOMATE","BANANA","CENOURA","ALFACE","MANDIOCA","UVA","MACA","BATATA","LARANJA"],
        "cosmetico": ["CREME","SHAMPOO","SABONETE","NIVEA","DOVE"],
        "utilidades": ["COPO","PRATO","GARFO","SACO","ROLO","LIXEIRA"],
        "pet": ["RACAO","WHISKAS","PEDIGREE","GATO","CACHORRO"],
        "outros": []
    }
    def classificar_produto(nome):
        nome = remover_acentos(str(nome).upper())
        for categoria, palavras in categoria_produtos.items():
            for palavra in palavras:
                if palavra in nome:
                    return remover_acentos(categoria.upper())
        return "OUTROS"
    df["CAT_PRODUTO"] = df["PRODUTO"].apply(classificar_produto)
    print("✅ Categorias de produtos classificadas:", df["CAT_PRODUTO"].nunique())
    # 🔹 EXPORTAÇÃO
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # 🔹 Padronizadao das unidades
    map_unidades = {
        "UN": "UNIDADE",
        "Un": "UNIDADE",
        "PC": "UNIDADE",
        "KG": "KG",
        "Kg": "KG",
        "GF": "GARRAFA",
        "GL": "GALAO",
        "BD": "BANDEJA",
        "LA": "LATA",
        "PT": "PACOTE",
        "FR": "FRASCO",
        "CT": "CARTELA",
        "AM": "AMARRADO"
    }
    df["UNIDADE"] = df["UNIDADE"].str.upper().map(map_unidades)
    print("✅ Unidades padronizadas:", df["UNIDADE"].nunique())
    # 🔹 Ordem final das colunas
    colunas_finais = [
        'CHAVE_ANONIMIZADA',
        'DATA',
        'MES_ANO',
        'PERIODO',
        'CNPJ',
        'SUPERMERCADO',
        'QTDE_TOTAL_NOTA',
        'VALOR_TOTAL_NOTA',
        'VALOR_TOTAL_TRIBUTOS',
        'COD_PRODUTO',
        'CAT_PRODUTO',
        'PRODUTO',
        'UNIDADE',
        'QTDE',
        'VALOR_PRODUTO',
        'VALOR_TOTAL_PRODUTO',
    ]
    df = df[colunas_finais]
    print("✅ DataFrame final com colunas ordenadas")
    df.to_excel(raw_path, index=False)
    print("✅ DataFrame exportado para Excel (XLSX):", raw_path)
    return df