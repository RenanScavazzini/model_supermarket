"""
Descrição:
    Aplicação principal do dashboard interativo do projeto model_supermarket.

Autor:
    Renan Douglas Floriano Scavazzini

Versão:
    1.0 - 12/05/2026
    2.0 - 12/05/2026 - Adição de novas funcionalidades e melhorias na interface do usuário.
    3.0 - 13/05/2026 - Adicionando design do anime That Time I Got Reincarnated as a Slime.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import base64
from pathlib import Path

from streamlit_option_menu import option_menu

from src.core.config_loader import ConfigLoader

from src.analysis.invoice_loader import InvoiceLoader

from src.dashboard.views import (
    overview,
    products,
    temporal,
    database
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

config = ConfigLoader(
    'config/settings.yaml'
)

loader = InvoiceLoader()

data_path = config.get(
    'paths.data'
)

df = loader.load(data_path)

tab_image_path = (
    Path("image/ui/selected_tab_blue.png")
)

tab_image_base64 = get_base64_image(
    tab_image_path
)

selected = option_menu(
    menu_title=None,
    options=['Overview', 'Temporal', 'Products', 'Database'],
    icons=['house', 'calendar', 'cart', 'database'],
    orientation='horizontal',
    styles={
        "container": {"margin": "0 !important", "padding": "0 !important"},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {
            "border-radius": "18px",
            "color": "white",
            "background-color": "rgba(255,255,255,0.05)",
            "padding": "12px 26px",
            "margin": "0 6px",
            "transition": "0.25s ease",
        },
        "nav-link:hover": {
            "background-color": "rgba(120,180,255,0.10)",
        },
        "nav-link-selected": {
            "background-image": f"url(data:image/png;base64,{tab_image_base64})",
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

if selected == 'Overview':

    overview.render(df)

elif selected == 'Temporal':

    temporal.render(df)

elif selected == 'Products':

    products.render(df)

elif selected == 'Database':

    database.render(df)