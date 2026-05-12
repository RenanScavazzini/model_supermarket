"""
Descrição:
    Aplicação principal do dashboard Streamlit responsável pela
    inicialização da interface, carregamento dos dados e controle
    de navegação entre páginas analíticas.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st

from streamlit_option_menu import option_menu

from src.core.config_loader import ConfigLoader
from src.core.logger import setup_logger

from src.analysis.invoice_loader import InvoiceLoader

from src.dashboard.pages import (
    overview,
    products,
    temporal
)


logger = setup_logger(__name__)


@st.cache_data
def load_data():
    """
    Descrição:
        Realiza carregamento e cache dos dados utilizados
        pelo dashboard.

    Parâmetros:
        ---

    Referências:
        - Streamlit Inc. (2024). Streamlit Documentation.
    """

    config = ConfigLoader(
        "config/settings.yaml"
    )

    data_path = config.get(
        "paths.data"
    )

    logger.info(
        f'Carregando dados do dashboard: {data_path}'
    )

    loader = InvoiceLoader()

    df = loader.load(data_path)

    logger.info(
        'Dados carregados com sucesso'
    )

    return df


st.set_page_config(
    page_title="Model Supermarket",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info(
    'Inicializando dashboard'
)

df = load_data()

st.title(
    "🛒 Model Supermarket Dashboard"
)

selected = option_menu(
    menu_title=None,

    options=[
        "Overview",
        "Products",
        "Temporal"
    ],

    icons=[
        "house",
        "cart",
        "calendar"
    ],

    orientation="horizontal"
)

logger.info(
    f'Página selecionada: {selected}'
)

if selected == "Overview":

    overview.render(df)

elif selected == "Products":

    products.render(df)

elif selected == "Temporal":

    temporal.render(df)

logger.info(
    'Renderização finalizada'
)