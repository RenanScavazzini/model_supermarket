"""
Descrição:
    Página do modelo estatístico do dashboard
    responsável pelas análises inteligentes,
    Market Basket Analysis e comportamento
    de consumo utilizando IA e regras
    de associação.

Autor:
    Renan Douglas Floriano Scavazzini

Versão:
    1.0 - 13/05/2026 - Estrutura inicial da página.
    2.0 - 19/05/2026 - Adição de Market Basket Analysis.
    3.0 - 19/05/2026 - Adição de storytelling analítico.
    4.0 - 19/05/2026 - Combos inteligentes e melhoria visual.
    5.0 - 19/05/2026 - Explicações analíticas e refinamento visual.
    6.0 - 19/05/2026 - Correção de performance e renderização.
    7.0 - 19/05/2026 - Correção visual e regras dinâmicas.
    8.0 - 19/05/2026 - Correção de duplicidade de combos.
    9.0 - 19/05/2026 - Padronização visual completa.
    10.0 - 19/05/2026 - MBA adaptativo inteligente.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import time

import streamlit as st
import pandas as pd

from src.core.logger import setup_logger

from src.dashboard.components.filters import (
    apply_filters
)

from src.core.model import (
    run_market_basket_analysis
)

from src.utils.formatters import (
    format_number
)


logger = setup_logger(__name__)


# ==========================================================
# FORMATA LISTA HTML
# ==========================================================

def format_rule_items(
    text: str
) -> str:
    """
    Descrição:
        Formata itens separados por vírgula
        em lista HTML.
    """

    items = [

        item.strip()

        for item in text.split(',')
    ]

    html = "<ol style='padding-left:18px; margin:0;'>"

    for item in items:

        html += f"<li>{item}</li>"

    html += "</ol>"

    return html


# ==========================================================
# CARD MÉTRICA
# ==========================================================

def metric_card(
    title: str,
    description: str
):
    """
    Descrição:
        Cria card padronizado de métrica.
    """

    st.markdown(

        f"""
        <div style="
            background-color:rgba(255,255,255,0.03);
            padding:16px;
            border-radius:18px;
            border:1px solid rgba(255,255,255,0.06);
            min-height:180px;
            line-height:1.40;
            font-size:14px;
        ">

        <h4 style="
            margin-top:0;
            margin-bottom:10px;
        ">
            {title}
        </h4>

        {description}

        </div>
        """,

        unsafe_allow_html=True
    )


# ==========================================================
# SELEÇÃO DE COMBOS STORYTELLING
# ==========================================================

def select_storytelling_rules(
    rules: pd.DataFrame
) -> list:
    """
    Descrição:
        Seleciona regras variadas para
        storytelling do dashboard evitando:

        - repetição de produtos
        - mesmas combinações invertidas
        - regras visualmente redundantes
    """

    desired_patterns = [

        (1, 1),

        (2, 1),

        (1, 2),

        (3, 1),

        (1, 3),

        (2, 2)
    ]

    selected_rules = []

    used_full_sets = []

    # ======================================================
    # PADRÕES PRINCIPAIS
    # ======================================================

    for antecedent_size, consequent_size in desired_patterns:

        best_candidate = None

        best_score = -999

        for _, row in rules.iterrows():

            antecedents = [

                item.strip()

                for item in row[
                    'antecedents'
                ].split(',')
            ]

            consequents = [

                item.strip()

                for item in row[
                    'consequents'
                ].split(',')
            ]

            if (

                len(antecedents)
                != antecedent_size

                or

                len(consequents)
                != consequent_size
            ):

                continue

            current_set = frozenset(
                antecedents + consequents
            )

            # ==================================================
            # EVITA COMBOS INVERTIDOS
            # ==================================================

            if current_set in used_full_sets:

                continue

            score = (

                row['lift'] * 100

                +

                row['confidence'] * 50

                +

                row['support'] * 10
            )

            if score > best_score:

                best_candidate = row

                best_score = score

        if best_candidate is not None:

            antecedents = [

                item.strip()

                for item in best_candidate[
                    'antecedents'
                ].split(',')
            ]

            consequents = [

                item.strip()

                for item in best_candidate[
                    'consequents'
                ].split(',')
            ]

            current_set = frozenset(
                antecedents + consequents
            )

            selected_rules.append(
                best_candidate
            )

            used_full_sets.append(
                current_set
            )

    # ======================================================
    # COMPLETA COM MAIS COMBOS
    # ======================================================

    if len(selected_rules) < 12:

        for _, row in rules.iterrows():

            antecedents = [

                item.strip()

                for item in row[
                    'antecedents'
                ].split(',')
            ]

            consequents = [

                item.strip()

                for item in row[
                    'consequents'
                ].split(',')
            ]

            current_set = frozenset(
                antecedents + consequents
            )

            if current_set in used_full_sets:

                continue

            selected_rules.append(
                row
            )

            used_full_sets.append(
                current_set
            )

            if len(selected_rules) >= 12:

                break

    return selected_rules[:12]


# ==========================================================
# RENDER
# ==========================================================

def render(
    df: pd.DataFrame
) -> None:
    """
    Descrição:
        Renderiza página de modelo estatístico.
    """

    logger.info(
        'Renderizando página Modelo Estatístico'
    )

    # ======================================================
    # FILTROS
    # ======================================================

    filtered_df, metric = apply_filters(df)

    # ======================================================
    # TÍTULO
    # ======================================================

    st.title(
        '🛒 Market Basket Analysis'
    )

    st.caption(
        '''
        O modelo aprende continuamente seus
        padrões de consumo conforme novas
        compras são adicionadas à base.
        '''
    )

    st.divider()

    # ======================================================
    # EXECUÇÃO MODELAGEM ADAPTATIVA
    # ======================================================

    with st.spinner(
        '🧠 Analisando padrões de compra...'
    ):

        start_time = time.time()

        search_configs = [

            {
                'min_support': 0.015,
                'min_threshold': 1.0,
                'max_len': 4,
                'top_n_rules': 120,
                'top_products_limit': 150
            },

            {
                'min_support': 0.010,
                'min_threshold': 0.9,
                'max_len': 4,
                'top_n_rules': 180,
                'top_products_limit': 180
            },

            {
                'min_support': 0.008,
                'min_threshold': 0.85,
                'max_len': 5,
                'top_n_rules': 250,
                'top_products_limit': 220
            },

            {
                'min_support': 0.005,
                'min_threshold': 0.8,
                'max_len': 5,
                'top_n_rules': 400,
                'top_products_limit': 250
            }
        ]

        basket_results = None

        rules = pd.DataFrame()

        selected_rules = []

        # ==================================================
        # BUSCA ADAPTATIVA
        # ==================================================

        for config in search_configs:

            elapsed = (

                time.time()
                - start_time
            )

            # ==============================================
            # TIMEOUT
            # ==============================================

            if elapsed >= 300:

                logger.warning(
                    'Timeout MBA atingido'
                )

                break

            logger.info(
                f'Executando MBA: {config}'
            )

            current_results = (

                run_market_basket_analysis(

                    df=filtered_df,

                    min_support=config[
                        'min_support'
                    ],

                    metric='lift',

                    min_threshold=config[
                        'min_threshold'
                    ],

                    max_len=config[
                        'max_len'
                    ],

                    top_n_rules=config[
                        'top_n_rules'
                    ],

                    top_products_limit=config[
                        'top_products_limit'
                    ]
                )
            )

            current_rules = current_results.get(
                'top_rules',
                pd.DataFrame()
            )

            if current_rules.empty:

                continue

            current_selected_rules = (

                select_storytelling_rules(
                    current_rules
                )
            )

            logger.info(

                f'Combos encontrados: '

                f'{len(current_selected_rules)}'
            )

            # ==============================================
            # MELHOR RESULTADO
            # ==============================================

            if (

                len(current_selected_rules)

                >

                len(selected_rules)
            ):

                basket_results = current_results

                rules = current_rules

                selected_rules = (

                    current_selected_rules
                )

            # ==============================================
            # PARADA INTELIGENTE
            # ==============================================

            if len(selected_rules) >= 11:

                logger.info(
                    'Quantidade ideal encontrada'
                )

                break

    # ======================================================
    # FALLBACK
    # ======================================================

    if basket_results is None:

        basket_results = {}

        rules = pd.DataFrame()

        selected_rules = []

    metrics_summary = basket_results.get(
        'metrics_summary',
        {}
    )

    # ======================================================
    # VALIDAÇÃO
    # ======================================================

    if rules.empty:

        st.warning(
            '''
            Nenhuma regra de associação foi
            encontrada com os filtros atuais.
            '''
        )

        return

    # ======================================================
    # HERO SECTION
    # ======================================================

    rimuru_col, hero_col = st.columns(
        [0.18, 0.82],
        gap="medium"
    )

    with rimuru_col:

        st.markdown(
            "<div style='height:25px'></div>",
            unsafe_allow_html=True
        )

        st.image(
            'image/ui/rimuru.png',
            width=210
        )

    with hero_col:

        st.subheader(
            '🧠 O que o modelo aprendeu sobre você'
        )

        st.markdown(

            f"""
            <div style="
                background-color:rgba(255,255,255,0.04);
                padding:16px;
                border-radius:18px;
                border:1px solid rgba(255,255,255,0.08);
                line-height:1.45;
                font-size:15px;
            ">

            📦 O modelo analisou
            <b>{format_number(metrics_summary.get('total_notas', 0), 0)}</b>
            notas fiscais.

            <br>

            🛒 Foram identificados
            <b>{format_number(metrics_summary.get('total_produtos', 0), 0)}</b>
            produtos únicos.

            <br>

            🔥 O sistema descobriu
            <b>{format_number(metrics_summary.get('total_regras', 0), 0)}</b>
            padrões relevantes de compra.

            <br>

            📈 Lift médio:
            <b>{metrics_summary.get('media_lift', 0)}</b>

            <br>

            🎯 Confidence média:
            <b>{metrics_summary.get('media_confidence', 0):.2%}</b>

            <br>

            📦 Support médio:
            <b>{metrics_summary.get('media_support', 0):.2%}</b>

            </div>
            """,

            unsafe_allow_html=True
        )

    st.divider()

    # ======================================================
    # MÉTRICAS EXPLICAÇÃO
    # ======================================================

    st.subheader(
        '📚 Como interpretar as métricas'
    )

    metric_col1, metric_col2, metric_col3 = st.columns(
        3,
        gap="medium"
    )

    with metric_col1:

        metric_card(

            title='📈 Lift',

            description="""
            Mede a força da relação entre produtos.

            
            Quanto maior o lift,
            mais forte é a associação
            entre os itens.
            """
        )

    with metric_col2:

        metric_card(

            title='🎯 Confidence',

            description="""
            Representa a probabilidade
            de um produto ser comprado
            junto com outro.

            Quanto maior,
            mais frequente é essa relação.
            """
        )

    with metric_col3:

        metric_card(

            title='📦 Support',

            description="""
            Mede a frequência total
            que uma combinação aparece
            nas compras.

            Indica relevância estatística
            do padrão encontrado.
            """
        )

    st.divider()

    # ======================================================
    # COMBOS MAIS FORTES
    # ======================================================

    st.subheader(
        '🔥 Combinações Mais Fortes'
    )

    st.caption(
        '''
        Produtos frequentemente comprados juntos
        identificados automaticamente pelo modelo.
        '''
    )

    cols = st.columns(4)

    for idx, row in enumerate(
        selected_rules
    ):

        col = cols[idx % 4]

        antecedents_html = format_rule_items(
            row['antecedents']
        )

        consequents_html = format_rule_items(
            row['consequents']
        )

        with col:

            st.markdown(

                f"""
                <div style="
                    background-color:rgba(255,255,255,0.03);
                    padding:11px;
                    border-radius:18px;
                    border:1px solid rgba(255,255,255,0.06);
                    margin-bottom:10px;
                    min-height:320px;
                    line-height:1.22;
                    font-size:13px;
                ">

                <h4 style="
                    margin-bottom:5px;
                ">
                🛒 Combo #{idx + 1}
                </h4>

                <b>Antecedentes:</b>

                {antecedents_html}

                <br>

                <b>Consequentes:</b>

                {consequents_html}

                <br>

                📈 Lift:
                <b>{row['lift']:.2f}</b>

                <br>

                🎯 Confidence:
                <b>{row['confidence']:.2%}</b>

                <br>

                📦 Support:
                <b>{row['support']:.2%}</b>

                <br><br><br><br><br><br>

                </div>
                """,

                unsafe_allow_html=True
            )

    st.divider()

    # ======================================================
    # INSIGHTS AUTOMÁTICOS
    # ======================================================

    st.subheader(
        '🤖 Insights Inteligentes'
    )

    st.caption(
        '''
        Interpretações automáticas geradas
        com base nos padrões aprendidos.
        '''
    )

    insight_left, raphael_col = st.columns(
        [0.74, 0.26],
        gap="medium"
    )

    with insight_left:

        insight_col1, insight_col2 = st.columns(
            2,
            gap="medium"
        )

        strongest_rule = rules.iloc[0]

        with insight_col1:

            st.markdown(

                f"""
                <div style="
                    background-color:rgba(255,255,255,0.03);
                    padding:14px;
                    border-radius:18px;
                    border:1px solid rgba(255,255,255,0.06);
                    min-height:100px;
                    line-height:1.25;
                    font-size:13px;
                ">

                <h4>🧠 Padrão Mais Forte</h4>

                O modelo identificou forte relação entre:

                <br>

                <b>{strongest_rule['antecedents']}</b>

                <br>

                ⬇

                <br>

                <b>{strongest_rule['consequents']}</b>

                <br><br>

                📈 Lift:
                <b>{strongest_rule['lift']:.2f}</b>

                </div>
                """,

                unsafe_allow_html=True
            )

        with insight_col2:

            st.markdown(

                f"""
                <div style="
                    background-color:rgba(255,255,255,0.03);
                    padding:14px;
                    border-radius:18px;
                    border:1px solid rgba(255,255,255,0.06);
                    min-height:100px;
                    line-height:1.25;
                    font-size:13px;
                ">

                <h4>📊 Evolução do Modelo</h4>

                🔄 Modelo vivo:
                <b>Ativo</b>

                <br><br>

                📅 Atualização:
                <b>Tempo real</b>

                <br><br>

                🧠 O sistema aprende continuamente
                conforme novas compras são adicionadas.

                </div>
                """,

                unsafe_allow_html=True
            )

    with raphael_col:

        st.image(
            'image/ui/raphael_bola.png',
            width=300
        )

    st.divider()

    # ======================================================
    # TABELA COMPLETA
    # ======================================================

    st.subheader(
        '📋 Regras de Associação'
    )

    display_rules = rules[[

        'antecedents',

        'consequents',

        'support',

        'confidence',

        'lift'
    ]].copy()

    display_rules['support'] = (

        display_rules['support']

        .apply(
            lambda x:
            f'{x:.2%}'
        )
    )

    display_rules['confidence'] = (

        display_rules['confidence']

        .apply(
            lambda x:
            f'{x:.2%}'
        )
    )

    display_rules['lift'] = (

        display_rules['lift']

        .apply(
            lambda x:
            f'{x:.2f}'
        )
    )

    st.dataframe(

        display_rules,

        use_container_width=True,

        height=450
    )

    logger.info(
        'Página Modelo Estatístico renderizada com sucesso'
    )