"""
Descrição:
    Aplicação principal do dashboard interativo do projeto model_supermarket.

Autor:
    Renan Douglas Floriano Scavazzini

Versão:
    1.0 - 12/05/2026
    2.0 - 12/05/2026 - Adição de novas funcionalidades e melhorias na interface do usuário.
    3.0 - 13/05/2026 - Adicionando design do anime That Time I Got Reincarnated as a Slime.
    4.0 - 13/05/2026 - Tradução das abas para PT-BR e melhoria visual da navegação.
    5.0 - 13/05/2026 - Adição de responsividade mobile.
    6.0 - 13/05/2026 - Ajustes visuais mobile e melhorias tipográficas.
    7.0 - 13/05/2026 - Adição da aba Modelo Estatístico.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import base64
from pathlib import Path
import sys

from streamlit_option_menu import option_menu

sys.path.append(
    str(
        Path(__file__).resolve().parents[2]
    )
)

from src.core.config_loader import ConfigLoader

from src.analysis.invoice_loader import InvoiceLoader

from src.dashboard.views import (
    overview,
    products,
    temporal,
    database,
    statistical_model
)


def get_base64_image(
    image_path: str
) -> str:
    """
    Descrição:
        Converte imagem para base64 para uso em CSS.

    Parâmetros:
        image_path (str): Caminho da imagem.

    Retorno:
        str: Imagem codificada em base64.
    """

    with open(
        image_path,
        "rb"
    ) as img_file:

        return base64.b64encode(
            img_file.read()
        ).decode()


st.set_page_config(
    page_title='Model Supermarket',
    page_icon='🛒',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# =====================================================
# BACKGROUND
# =====================================================

background_path = (
    Path("image/background.png")
)

background_base64 = get_base64_image(
    background_path
)

st.markdown(

    f"""
    <style>

    .stApp {{

        background-image:
        linear-gradient(
            rgba(0, 0, 0, 0.85),
            rgba(0, 0, 0, 0.85)
        ),
        url("data:image/png;base64,{background_base64}");

        background-size: cover;

        background-position: center;

        background-attachment: fixed;
    }}

    </style>
    """,

    unsafe_allow_html=True
)

# =====================================================
# RESPONSIVIDADE MOBILE
# =====================================================

st.markdown(

    """
    <style>

    /* =================================================
       MOBILE RESPONSIVO
    ================================================= */

    @media (max-width: 768px) {

        /* =============================================
           TABS
        ============================================= */

        .nav-link {

            font-size: 13px !important;

            padding: 10px 12px !important;

            margin: 2px !important;

            border-radius: 12px !important;
        }

        /* =============================================
           ÍCONES
        ============================================= */

        .nav-link i {

            font-size: 14px !important;
        }

        /* =============================================
        TÍTULOS MOBILE
        ============================================= */

        h1 {

            font-size: 42px !important;

            font-weight: 800 !important;

            line-height: 1.2 !important;
        }

        h2 {

            font-size: 34px !important;

            font-weight: 700 !important;

            line-height: 1.2 !important;
        }

        h3 {

            font-size: 30px !important;

            font-weight: 700 !important;

            line-height: 1.2 !important;
        }

        /* =============================================
        STREAMLIT HEADERS
        ============================================= */

        [data-testid="stMarkdownContainer"] h1 {

            font-size: 42px !important;

            font-weight: 800 !important;
        }

        [data-testid="stMarkdownContainer"] h2 {

            font-size: 34px !important;

            font-weight: 700 !important;
        }

        [data-testid="stMarkdownContainer"] h3 {

            font-size: 30px !important;

            font-weight: 700 !important;
        }

        /* =============================================
           MÉTRICAS
        ============================================= */

        [data-testid="metric-container"] {

            padding: 8px !important;
        }

        /* =============================================
           IMAGENS
        ============================================= */

        img {

            max-width: 100% !important;

            height: auto !important;
        }

        /* =============================================
           PLOTLY
        ============================================= */

        .js-plotly-plot {

            width: 100% !important;
        }

        /* =============================================
           SIDEBAR
        ============================================= */

        section[data-testid="stSidebar"] {

            width: 260px !important;
        }

        /* =============================================
           COLUNAS
        ============================================= */

        div[data-testid="column"] {

            width: 100% !important;

            flex: 1 1 100% !important;

            min-width: 100% !important;
        }

        /* =============================================
           MÉTRICAS PERSONALIZADAS
        ============================================= */

        div[data-testid="stHorizontalBlock"] {

            gap: 8px !important;
        }

        /* =============================================
           FONTES MOBILE
        ============================================= */

        label {

            font-size: 15px !important;
        }

        span {

            font-size: 15px !important;
        }

        /* =============================================
           DATAFRAME
        ============================================= */

        .stDataFrame {

            overflow-x: auto !important;
        }
    }

    </style>
    """,

    unsafe_allow_html=True
)

# =====================================================
# LOGO SIDEBAR
# =====================================================

logo_path = (
    Path(
        "image/ui/logo_market.png"
    )
)

st.sidebar.markdown(

    """
    <style>

    [data-testid="stSidebar"] img {

        display: block;

        margin-left: auto;

        margin-right: auto;
    }

    </style>
    """,

    unsafe_allow_html=True
)

col1, col2, col3 = st.sidebar.columns(
    [0.1, 3, 0.1]
)

with col2:

    st.image(
        str(logo_path),
        width=220
    )

# =====================================================
# CONFIGURAÇÕES
# =====================================================

config = ConfigLoader(
    'config/settings.yaml'
)

loader = InvoiceLoader()

data_path = config.get(
    'paths.data'
)

df = loader.load(data_path)

# =====================================================
# TABS
# =====================================================

tab_image_path = (
    Path("image/ui/selected_tab_blue.png")
)

tab_image_base64 = get_base64_image(
    tab_image_path
)

selected = option_menu(

    menu_title=None,

    options=[
        'Visão Geral',
        'Temporal',
        'Produtos',
        'Base de Dados',
        'Modelo'
    ],

    icons=[
        'house',
        'calendar',
        'cart',
        'database',
        'activity'
    ],

    orientation='horizontal',

    styles={

        "container": {

            "margin": "0 !important",

            "padding": "0 !important"
        },

        "icon": {

            "color": "white",

            "font-size": "22px"
        },

        "nav-link": {

            "border-radius": "18px",

            "color": "white",

            "background-color":
            "rgba(255,255,255,0.05)",

            "padding": "14px 30px",

            "margin": "0 6px",

            "transition": "0.25s ease",

            "font-size": "20px",

            "font-weight": "600",
        },

        "nav-link:hover": {

            "background-color":
            "rgba(120,180,255,0.10)",
        },

        "nav-link-selected": {

            "background-image":
            f"url(data:image/png;base64,{tab_image_base64})",

            "background-size": "100% 100%",

            "background-position": "center",

            "background-repeat": "no-repeat",

            "background-color": "transparent",

            "padding": "16px 34px",

            "overflow": "visible",

            "color": "white",

            "font-weight": "700",

            "border": "none",

            "box-shadow":
            "0 0 18px rgba(80,180,255,0.55)",
        },
    }
)

# =====================================================
# RENDERIZAÇÃO DAS PÁGINAS
# =====================================================

if selected == 'Visão Geral':

    overview.render(df)

elif selected == 'Temporal':

    temporal.render(df)

elif selected == 'Produtos':

    products.render(df)

elif selected == 'Base de Dados':

    database.render(df)

elif selected == 'Modelo':

    statistical_model.render(df)