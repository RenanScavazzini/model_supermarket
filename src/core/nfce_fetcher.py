import time
import hashlib
import pandas as pd
from pathlib import Path
from nfceget import app
import unicodedata

from src.external.weather import (
    download_weather_data,
    load_weather_data
)


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
        2.0 - 18/05/2026 - Correção do de-para da chave
        3.0 - 19/05/2026 - Correção na quantidade onde unidade é KG

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
            for col in [
                'name',
                'code',
                'quantity',
                'unit',
                'unitaryValue',
                'totalValue'
            ]:

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
            df_nota['QTDE'] = (

                df_nota['QTDE']

                .astype(str)

                .str.upper()

                .str.strip()

                .str.replace('\xa0', '', regex=False)

                .str.replace('KG', '', regex=False)

                .str.replace(',', '.', regex=False)

                .str.replace(r'[^0-9\.]', '', regex=True)
            )

            df_nota['QTDE'] = pd.to_numeric(

                df_nota['QTDE'],

                errors='coerce'
            )

            df_nota['VALOR_PRODUTO'] = (
                df_nota['VALOR_PRODUTO']
                .astype(str)
                .str.replace(',', '.', regex=False)
            )

            df_nota['VALOR_PRODUTO'] = pd.to_numeric(
                df_nota['VALOR_PRODUTO'],
                errors='coerce'
            )

            df_nota['VALOR_TOTAL_PRODUTO'] = (
                df_nota['VALOR_TOTAL_PRODUTO']
                .astype(str)
                .str.replace(',', '.', regex=False)
            )

            df_nota['VALOR_TOTAL_PRODUTO'] = pd.to_numeric(
                df_nota['VALOR_TOTAL_PRODUTO'],
                errors='coerce'
            )

            # 🔹 Validações
            if df_nota['VALOR_PRODUTO'].isna().any():

                print(f"⚠ Preço unitário inválido: {url}")

                df_nota = df_nota.dropna(
                    subset=['VALOR_PRODUTO']
                )

            if df_nota['VALOR_TOTAL_PRODUTO'].isna().any():

                print(f"⚠ Preço total inválido: {url}")

                df_nota = df_nota.dropna(
                    subset=['VALOR_TOTAL_PRODUTO']
                )

            if df_nota.empty:

                print(f"⚠ Nenhum item válido: {url}")
                continue

            # 🔹 Dados da nota
            local = nota.get('local') or {}

            df_nota['SUPERMERCADO'] = local.get('name')
            df_nota['CNPJ'] = local.get('cnpj')

            nfce = nota.get('nfce') or {}
            totals = nota.get('totals') or {}

            df_nota['DATA'] = (
                nfce.get('date')
                or totals.get('date')
            )

            # 🔹 Quantidade total
            try:

                quantidade_total = totals.get('quantityItens')

                if quantidade_total is not None:
                    quantidade_total = int(str(quantidade_total))

                else:
                    quantidade_total = round(df_nota['QTDE'].sum(), 2)

            except Exception:

                quantidade_total = round(df_nota['QTDE'].sum(), 2)

            df_nota['QTDE_TOTAL_NOTA'] = quantidade_total

            # 🔹 Valores
            valor_total = (
                totals.get('valueToPay')
                or nfce.get('total')
            )

            if isinstance(valor_total, str):
                valor_total = valor_total.replace(',', '.')

            df_nota['VALOR_TOTAL_NOTA'] = pd.to_numeric(
                valor_total,
                errors='coerce'
            )

            valor_tributos = (
                totals.get('taxes')
                or nfce.get('taxes')
            )

            if isinstance(valor_tributos, str):
                valor_tributos = valor_tributos.replace(',', '.')

            df_nota['VALOR_TOTAL_TRIBUTOS'] = pd.to_numeric(
                valor_tributos,
                errors='coerce'
            )

            # 🔹 Chave original
            df_nota['CHAVE_ANONIMIZADA'] = (
                nfce.get('chave')
            )

            dataframes.append(df_nota)

            time.sleep(delay)

        except Exception as e:

            print(f"❌ Erro ao processar: {url}")
            print(e)

    # 🔹 Sem dados
    if not dataframes:

        print("⚠ Nenhuma nota válida encontrada")
        return pd.DataFrame()

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # 🔹 Garantir diretório
    fetched_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================================
    # 🔹 ANONIMIZAÇÃO
    # ==========================================================

    orig_chaves = (
        df['CHAVE_ANONIMIZADA']
        .dropna()
        .astype(str)
        .unique()
        if 'CHAVE_ANONIMIZADA' in df.columns
        else []
    )

    map_rows = []

    for chave in orig_chaves:

        anon = hashlib.sha256(
            str(chave).encode('utf-8')
        ).hexdigest()

        map_rows.append({
            'nfce_chave_original': str(chave),
            'nfce_chave_anon': str(anon)
        })

    if map_rows:

        df_map = pd.DataFrame(map_rows)

        # 🔹 Garantir TEXTO no Excel
        df_map['nfce_chave_original'] = (
            "'" + df_map['nfce_chave_original'].astype(str)
        )

        df_map['nfce_chave_anon'] = (
            df_map['nfce_chave_anon']
            .astype(str)
        )

        # 🔹 Exportar CSV
        df_map.to_csv(
            mapping_path,
            index=False,
            encoding='utf-8-sig'
        )

        # 🔹 Dicionário de anonimização
        map_dict = dict(zip(
            df_map['nfce_chave_original'].str.replace("'", ""),
            df_map['nfce_chave_anon']
        ))

        # 🔹 Aplicar anonimização
        df['CHAVE_ANONIMIZADA'] = (
            df['CHAVE_ANONIMIZADA']
            .astype(str)
            .map(
                lambda x: (
                    map_dict.get(x)
                    if pd.notna(x)
                    else x
                )
            )
        )

    else:

        pd.DataFrame().to_csv(
            mapping_path,
            index=False
        )

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
    df.to_csv(
        fetched_path,
        index=False
    )

    print(
        "✅ DataFrame exportado para CSV:",
        fetched_path
    )

    return df


def process_base(
    fetch_enabled: bool,
    df_fetched: pd.DataFrame,
    raw_path: Path,
    weather_enabled: bool,
    weather_output_path: str,
    latitude: float,
    longitude: float
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
        weather_enabled (bool): Habilita integração com clima.
        weather_output_path (str): Caminho parquet da base climática.
        latitude (float): Latitude da localização.
        longitude (float): Longitude da localização.

    Retorno:
        pd.DataFrame

    Referências:
    ---

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026
        2.0 - 18/05/2026 - Adição de integração com dados climáticos históricos.

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    import holidays
    import unicodedata

    print(f"Configuração de busca (nfceget): FETCH_REAL_DATA={fetch_enabled}")

    # =====================================================
    # MODO LEITURA
    # =====================================================

    if not fetch_enabled:

        print(f"ℹ Lendo base já processada: {raw_path}")

        if raw_path.exists():

            return pd.read_excel(raw_path)

        print("⚠ Arquivo não encontrado")

        return pd.DataFrame()

    # =====================================================
    # MODO PROCESSAMENTO
    # =====================================================

    df = df_fetched.copy()

    # =====================================================
    # FILTRO CNPJ + NORMALIZAÇÃO
    # =====================================================

    cnpj_map = {
        "76.189.406/0034-94": "CONDOR",
        "76.189.406/0050-04": "CONDOR",
        "76.430.438/0070-01": "MAX",
        "76.430.438/0065-36": "MAX",
        "06.057.223/0367-96": "ASSAI",
        "40.541.861/0001-00": "HONESTY",
        # "00.776.574/1699-08": "AMERICANAS",
        # "78.116.670/0035-04": "FESTVAL",
    }

    df = df[df["CNPJ"].isin(cnpj_map.keys())].copy()

    print("✅ CNPJ filtrados:", df["CNPJ"].nunique())

    df["SUPERMERCADO"] = df["CNPJ"].map(cnpj_map)

    print(
        "✅ Supermercados normalizados:",
        df["SUPERMERCADO"].nunique()
    )

    # =====================================================
    # DATA
    # =====================================================

    df["DATA"] = pd.to_datetime(

        df["DATA"],

        format="%d/%m/%Y %H:%M:%S",

        errors="coerce"
    )

    df["MES_ANO"] = df["DATA"].dt.strftime("%Y-%m")

    print(
        "✅ Datas convertidas, meses/anos únicos:",
        df["MES_ANO"].nunique()
    )

    # =====================================================
    # PERÍODO
    # =====================================================

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

    df["PERIODO"] = df["DATA"].apply(
        classificar_periodo
    )

    print(
        "✅ Períodos classificados:",
        df["PERIODO"].nunique()
    )

    # =====================================================
    # DIA DA SEMANA
    # =====================================================

    dias_semana = {

        0: 'SEGUNDA',

        1: 'TERCA',

        2: 'QUARTA',

        3: 'QUINTA',

        4: 'SEXTA',

        5: 'SABADO',

        6: 'DOMINGO'
    }

    df['DIA_SEMANA'] = (

        df['DATA']

        .dt.dayofweek

        .map(dias_semana)
    )

    # =====================================================
    # FERIADOS PARANÁ
    # =====================================================

    br_holidays = holidays.Brazil(
        subdiv='PR'
    )

    df['FERIADO'] = (

        df['DATA']

        .dt.date

        .apply(

            lambda x:

            'SIM'

            if x in br_holidays

            else 'NAO'
        )
    )

    print("✅ Features temporais criadas")

    # =====================================================
    # ESTAÇÃO DO ANO
    # =====================================================

    def classificar_estacao(data):

        mes = data.month

        dia = data.day

        # =============================================
        # VERÃO
        # =============================================

        if (

            (mes == 12 and dia >= 21)

            or mes in [1, 2]

            or (mes == 3 and dia < 20)

        ):

            return "VERAO"

        # =============================================
        # OUTONO
        # =============================================

        elif (

            (mes == 3 and dia >= 20)

            or mes in [4, 5]

            or (mes == 6 and dia < 21)

        ):

            return "OUTONO"

        # =============================================
        # INVERNO
        # =============================================

        elif (

            (mes == 6 and dia >= 21)

            or mes in [7, 8]

            or (mes == 9 and dia < 23)

        ):

            return "INVERNO"

        # =============================================
        # PRIMAVERA
        # =============================================

        else:

            return "PRIMAVERA"

    df["ESTACAO_ANO"] = (

        df["DATA"]

        .apply(classificar_estacao)
    )

    ordem_estacao = [

        "VERAO",

        "OUTONO",

        "INVERNO",

        "PRIMAVERA"
    ]

    df["ESTACAO_ANO"] = pd.Categorical(

        df["ESTACAO_ANO"],

        categories=ordem_estacao,

        ordered=True
    )

    print(
        "✅ Estação do ano criada"
    )

    # =====================================================
    # WEATHER FEATURES
    # =====================================================

    if weather_enabled:

        print("🌦 Processando dados climáticos...")

        # =================================================
        # DOWNLOAD WEATHER
        # =================================================

        start_date = (

            df["DATA"]

            .min()

            .strftime("%Y-%m-%d")
        )

        end_date = (

            df["DATA"]

            .max()

            .strftime("%Y-%m-%d")
        )

        weather_df = download_weather_data(

            start_date=start_date,

            end_date=end_date,

            latitude=latitude,

            longitude=longitude,

            output_path=weather_output_path
        )

        # =================================================
        # MERGE WEATHER
        # =================================================

        df["DATA_DIA"] = (

            df["DATA"]

            .dt.normalize()
        )

        weather_df["DATA_DIA"] = (

            weather_df["DATA"]

            .dt.normalize()
        )

        weather_columns = [

            "DATA_DIA",

            "TEMPERATURA_MAX",

            "TEMPERATURA_MIN",

            "TEMPERATURA_MEDIA",

            "CHUVA_MM",

            "CAT_TEMPERATURA",

            "DIA_CHUVOSO"
        ]

        df = df.merge(

            weather_df[
                weather_columns
            ],

            on="DATA_DIA",

            how="left"
        )

        print(
            "✅ Features climáticas adicionadas"
        )

    # =====================================================
    # PADRONIZAÇÃO PRODUTO
    # =====================================================

    df = df.sort_values("DATA")

    produto_map = (

        df.groupby("COD_PRODUTO")

        .last()["PRODUTO"]

        .to_dict()
    )

    df["PRODUTO"] = (

        df["COD_PRODUTO"]

        .map(produto_map)
    )

    print(
        "✅ Produtos padronizados:",
        df["PRODUTO"].nunique()
    )

    # =====================================================
    # REMOVER ACENTOS
    # =====================================================

    def remover_acentos(texto):

        return ''.join(

            c for c in unicodedata.normalize(
                'NFD',
                str(texto)
            )

            if unicodedata.category(c) != 'Mn'
        )

    # =====================================================
    # CATEGORIZAÇÃO
    # =====================================================

    categoria_produtos = {

        "mercearia": [
            "ARROZ","ARR","ARZ","ARRO","ARROZ T1","ARROZ TP1",
            "FEIJAO","FEIJ","FJ","FEIJAO PTO","FEIJAO CAR",
            "FARINHA","FAR","FAR TRIGO","FAR TRI","TRIGO","TRIG",
            "FUBA","FUBA MIM","POLVILHO","POLV","TAPIOCA","TAP",
            "ACUCAR","ACUC","ACU","SUGAR","DEM","MASC","REFIN",
            "SAL","SALG","SAL GR","SAL GROSSO",
            "OLEO","OLE","OL SOJA","AZEITE","AZT","COAMO",
            "OVO","OVOS","OV BR","OV VERM",
            "FAROFA","FARO","YOKI","AMIDO","MAIZ","CANJICA",
            "LENTILHA","LENT","GRAO BICO","GR BICO",
            "ERVILHA SECA","ERV SEC","MILHO PIPOCA","PIPOCA",
            "CANJ","COCO R","FILTRO","F MAND","M PIP"
        ],

        "massa": [
            "MAC","MACAR","MACARRAO","MASSA","MASS",
            "ESPAG","ESPAGUETE","PENNE","PENE",
            "TALHARIM","TALH","FETTUCCINE","FETT",
            "LASANHA","LAS","LAS PERD","MIOJO",
            "NISSIN","NOODLES","YAKISOBA","YAKI",
            "PARAFUSO","PARAF","CONCHINHA","CONCH",
            "CAPELETTI","CAPEL","RAVIOLI","RAVI",
            "GNOCCHI","NHOQUE","PASTEL","DISC PAST",
            "PANQUECA","TORTELLINI","TORTEL"
        ],

        "cereal": [
            "CEREAL","CERE","NESCAU","NESC","SUCRILHOS",
            "SUCRI","AVEIA","AVE","FLOCOS","FLOC",
            "GRANOLA","GRAN","MUESLI","MUSLI",
            "BARRA","BAR CERE","CRUNCH","CORN",
            "SNOW","KELLOG","NESFIT","FIBRAS",
            "CHOCOBALL","CHOCB","MATINAL"
        ],

        "bebida nao alcoolica": [
            "REF","REFRI","REFRIG","REFRIGERANTE",
            "COCA","COKE","PEPSI","GUARANA","GUA",
            "SPRITE","FANTA","SCHWEP","SCHWEPPES",
            "SUCO","SUC","DEL VALLE","VALLE","MAGUARY",
            "TANG","ADES","SOYA","YOPRO",
            "ENERG","ENERGETICO","RED BULL","MONSTER",
            "MONST","FUSION","AGUA","AG","AG MIN",
            "CRYSTAL","CRISTAL","MINALBA","BONAFONT",
            "ISOTONICO","GATORADE","POWERADE",
            "CHA","CHA MAT","TEA","MATE","LEAO",
            "CAFE","CAF","PILAO","3 COR","CAPPUCC",
            "ACHOC","TODDY","NESCAU","CHOCOMILK",
            "LEITE","LEIT","VITAMINA","H2OH",
            "ICE TEA","CITRUS","PURITY","PRATSY",
            "BEB","BEB LAC","NESQUIK","XAROPE",
            "SHAK","LTE COC","LTE ITALAC"
        ],

        "bebida alcoolica": [
            "CERV","CERVEJA","LONG NECK","LN",
            "SKOL","HEINEKEN","HEINE","BUD","BUDW",
            "SPATEN","PETRA","BRAHMA","BRAHM",
            "STELLA","AMSTEL","CORONA",
            "VINHO","VIN TTO","VIN BCO",
            "ESPUM","CHAMP","WHISK","WHISKY",
            "RED LABEL","BLACK LABEL","PASSPORT",
            "VODKA","SMIRNOFF","ABSOLUT",
            "GIN","TEQUILA","RUM","CACHA",
            "ICE","BEATS","APEROL","CAMPARI",
            "LICOR","SAKE","JACK DANIELS",
            "JACK","PASSAPORT","JOHNNIE WALKER",
            "JOHNNIE","WHISKEY","CHOPP",
            "IPA","COQ","COQUET","COROT",
            "BELLA ROMA"
        ],

        "padaria e frios": [
            "PAO","PAO FR","P FRANCES","P FORMA",
            "PAO INT","P INT","PAO DOCE",
            "P QUEIJO","BAGUETE","CROISSANT",
            "SONHO","ROSCA","CUCA",
            "BOLO","BOL","MUFFIN","TORRADA",
            "MORT","MORTADELA","PRES","PRESUNTO",
            "SALAME","QJO","QUEIJO","MUSS",
            "MUSSARELA","MUSSAR","PROVOL",
            "RICOTA","REQUEIJAO","REQ",
            "PEITO PERU","BLANQUET",
            "CREAM CHEESE","PARMESAO",
            "CUPCAKE","PANET","TORR BAUD",
            "TORR ISABELA"
        ],

        "congelado": [
            "CONG","CONGELADO","PIZZA","PIZ",
            "LASANHA","EMPANADO","EMP",
            "NUGGETS","HAMB","HAMBURGUER",
            "BATATA","BAT FRITA","SMILE",
            "SORVETE","SORV","PICOL","ICE CREAM",
            "HOT POCKET","TEKITOS","REF PRONTA",
            "LEG CONG","FRUT CONG","FRANGO CONG",
            "PEIXE CONG","SEARA","SADIA","PERDIG"
        ],

        "acougue": [
            "FRANGO","FRANG","PEITO","PT FR",
            "COXA","SOBRECOXA","ASA","MEIO ASA",
            "FILE","FILEZ","BIFE","BIF",
            "ALCATRA","PICANHA","FRALDINHA",
            "PATINHO","ACEM","ACEM MO",
            "CARNE","CARN BOV","CAR MOIDA",
            "COSTELA","COST SUINA","PERNIL",
            "LOMBO","LING","LINGUICA","CALAB",
            "BACON","SALS","SALSICHA",
            "FRIBOI","FRIB","SEARA","SADIA",
            "CNE FG","CORTES","COX",
            "COXINHA","LI FGO","MEIO PEIT"
        ],

        "biscoito e snack": [
            "BISC","BISCOITO","BOLACHA",
            "WAFER","CHIPS","SNACK",
            "SALG","SALGAD","DORITOS",
            "RUFFLES","PRINGLES","FANDANGOS",
            "CHEETOS","ELMA","BAT PALHA",
            "PIPOCA","AMENDOIM","OREO",
            "NEGRESCO","TRAKINAS","PASSATEMPO",
            "CLUB SOCIAL","CRACKER",
            "TORR","TORT","WAF",
            "POPCOR","POPCORN"
        ],

        "bomboniere": [
            "BALA","BAL","PIRUL","CHIC",
            "TRIDENT","HALLS","MENTOS",
            "CHOC","CHOCOLATE","BOMBOM",
            "CONFETE","KINDER","FERRERO",
            "LACTA","GAROTO","NESTLE",
            "BIS","NUTELLA","PACOCA",
            "FINI","DORI","GELATINA",
            "CH HERSH","HERSHEY",
            "CONFEITO","MEMS","PACOQ",
            "GELADINH"
        ],

        "laticinio": [
            "LEITE","LEIT","UHT","DESN",
            "INT","INTEGRAL","QUEIJO",
            "QJO","MUSS","MUSSARELA",
            "IOG","IOGUR","IOGURTE",
            "DANONE","DANONINHO",
            "REQ","REQUEIJAO","CREME LEITE",
            "CREM LEI","MANTEIGA","MARGARINA",
            "ACHOC","CHOCOMILK","YOPRO",
            "BATAV","BATAVO","ELEGE",
            "PIRACANJUBA","TIROL",
            "CREME LTE","MIST MOOCA",
            "MIX SABOR","DOCE FRIMESA"
        ],

        "molho": [
            "MOLHO","MOL","KETCHUP","CATCHUP",
            "MAION","MAIONESE","MOST","MOSTARDA",
            "TEMP","TEMPERO","SAZON",
            "POMAROLA","QUERO","HEMMER",
            "HELLM","HELLMANN","ELEF",
            "BARBECUE","SHOYU","MOL PIM",
            "CATCH SUAVIT"
        ],

        "lataria e conserva": [
            "LATA","LAT","CONSERVA",
            "MILHO","ERVILHA","ATUM",
            "SARDINHA","PALMITO","AZEITONA",
            "EXTRATO","MOL TOM","BONARE",
            "FUGINI","SELETA","PEPINO",
            "AZEITON","COGUMEL",
            "COG","PATE"
        ],

        "condimento": [
            "TEMP","TEMPERO","ALHO",
            "CEBOLA","PIMENTA","OREG",
            "CHIMICH","LOURO","CURRY",
            "PAPRICA","SALSA","CEBOLINHA",
            "AZEITE","VINAGRE","KITANO",
            "ALECRIM","AROMA","COLORIF",
            "LEMON","MANJER","M PIM",
            "POLPA","SHAK","KININO",
            "TRIANGULO"
        ],

        "higiene e limpeza": [
            "LIMP","DETERG","DET","SABAO",
            "OMO","TIXAN","LYSOL",
            "VEJA","DESINF","PINHO",
            "SANIT","AJAX","ALVEJ",
            "MULTIUSO","ESPONJA",
            "PAP H","PAPEL HIG","PAP TOALHA",
            "SHAMP","SHAMPOO","COND",
            "DESOD","SABON","SABONETE",
            "COLGATE","ORAL B","ABS",
            "FRALDA","LENCO UMED",
            "ALCOOL","ALG","ALGODAO",
            "AMACIANTE","AM DOWN",
            "AM COMF","CD COLG",
            "COLG","CRE DENT",
            "CREME DENT","ESC DENT",
            "ESCOVA DENTAL","ESCOVA",
            "FIO DENT","LENCO",
            "PAMPERS","PAMP",
            "LV R BRILHANTE",
            "BRILHANTE","PURIFIC",
            "BOM AR","REPEL",
            "SH HEAD","HEAD SHOUL",
            "SH LOREAL",
            "TOALHA HUGGIES",
            "TOALHA MEU BEBE",
            "SBT PALM","SCOTCH",
            "PANO","LA BOM BRIL",
            "BOM BRIL", "DIABO VD"
        ],

        "hortifruti": [
            "BANANA","MACA","MAÇA","PERA",
            "UVA","LARANJA","LIMAO",
            "ABACAXI","MELANCIA","MAMAO",
            "MORANGO","KIWI","TOMATE",
            "CEBOLA","BATATA","MANDIOCA",
            "CENOURA","ALFACE","RUCULA",
            "BROCOLIS","COUVE","BETERRABA",
            "PEPINO","ABOBR","PIMENTAO",
            "REPOLHO","CHEIRO VERDE",
            "ABACATE","ABOBORA",
            "ABOBORA ITALIA",
            "ABOBORA CAVALLI",
            "ALF CRESPA","BROC",
            "CHEIRO V","MANGA",
            "MANGA PALMER",
            "MANGA TOMY"
        ],

        "cosmetico": [
            "NIVEA","DOVE","PANTENE",
            "SEDA","HIDRAT","PERFUME",
            "COLONIA","MAQUIAGEM",
            "BATOM","DESOD","SHAMP",
            "CONDIC","SABONETE",
            "NIELY","CREME"
        ],

        "utilidades": [
            "COPO","PRATO","GARFO",
            "FACA","COLHER","PANELA",
            "POTE","VASSOURA","RODO",
            "BALDE","LIXEIRA","SACO",
            "PILHA","FOSFORO","VELA",
            "ALUMINIO","PVC","GUARDANAPO",
            "CARVAO","CESTO",
            "CP NADIR","NADIR",
            "EXTENS","LAMPADA",
            "PALITO","PINO T",
            "PRAT NAD","ROLO",
            "FILM","BOMPACK",
            "WYDA", "COLA SUPER BON",
            "FOS PARANA", "PREND PARANA",
            "PREDEND PARANA", "TOALHA PAP"
        ],

        "pet": [
            "RACAO","RAC","WHISKAS",
            "PEDIGREE","GATO","CACHORRO",
            "AREIA GATO","PETISCO",
            "SACHE","GRANPLUS","FRISKIES",
            "CAT CHOW","DOG CHOW"
        ],

        "outros": []
    }

    def classificar_produto(nome):

        nome = remover_acentos(
            str(nome).upper()
        )

        for categoria, palavras in categoria_produtos.items():

            for palavra in palavras:

                if palavra in nome:

                    return remover_acentos(
                        categoria.upper()
                    )

        return "OUTROS"

    df["CAT_PRODUTO"] = (

        df["PRODUTO"]

        .apply(classificar_produto)
    )

    print(
        "✅ Categorias classificadas:",
        df["CAT_PRODUTO"].nunique()
    )

    # =====================================================
    # UNIDADES
    # =====================================================

    map_unidades = {

        "UN": "UNIDADE",

        "PC": "UNIDADE",

        "KG": "KG",

        "GF": "GARRAFA",

        "GL": "GALAO",

        "BD": "BANDEJA",

        "LA": "LATA",

        "PT": "PACOTE",

        "FR": "FRASCO",

        "CT": "CARTELA",

        "AM": "AMARRADO"
    }

    df["UNIDADE"] = (

        df["UNIDADE"]

        .str.upper()

        .map(map_unidades)
    )

    print(
        "✅ Unidades padronizadas:",
        df["UNIDADE"].nunique()
    )

    # =====================================================
    # PADRONIZAÇÃO FINAL PRODUTO
    # =====================================================

    maior_codigo_produto = (

        df.groupby("PRODUTO")[
            "COD_PRODUTO"
        ]

        .max()
    )

    df["COD_PRODUTO"] = (

        df["PRODUTO"]

        .map(maior_codigo_produto)
    )

    produto_mais_recente = (

        df.drop_duplicates(

            subset="COD_PRODUTO",

            keep="last"
        )

        .set_index("COD_PRODUTO")[
            "PRODUTO"
        ]
    )

    df["PRODUTO"] = (

        df["COD_PRODUTO"]

        .map(produto_mais_recente)
    )

    print("✅ Nome dos produtos padronizados")

    # =====================================================
    # COLUNAS FINAIS
    # =====================================================

    colunas_finais = [

        'CHAVE_ANONIMIZADA',

        'DATA',

        'MES_ANO',

        'DIA_SEMANA',

        'FERIADO',

        'ESTACAO_ANO',

        'TEMPERATURA_MAX',

        'TEMPERATURA_MIN',

        'TEMPERATURA_MEDIA',

        'CHUVA_MM',

        'CAT_TEMPERATURA',

        'DIA_CHUVOSO',

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

    # =====================================================
    # PADRÃO DASHBOARD
    # =====================================================

    df.columns = [

        'chave_anonimizada',

        'data_hora',

        'mes_ano',

        'dia_semana',

        'feriado',

        'estacao_ano',

        'temperatura_max',

        'temperatura_min',

        'temperatura_media',

        'chuva_mm',

        'cat_temperatura',

        'dia_chuvoso',

        'periodo_dia',

        'cnpj',

        'supermercado',

        'qtd_total_nota',

        'valor_total_nota',

        'valor_total_tributos',

        'cod_produto',

        'categoria_produto',

        'produto',

        'unidade',

        'quantidade',

        'preco_unitario',

        'preco_total',
    ]

    print("✅ DataFrame final com colunas ordenadas")

    # =====================================================
    # EXPORTAÇÃO
    # =====================================================

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_excel(
        raw_path,
        index=False
    )

    print(
        "✅ DataFrame exportado:",
        raw_path
    )

    return df