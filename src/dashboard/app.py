"""
Descrição:
    Aplicação principal do dashboard interativo do projeto model_supermarket.

Autor:
    Renan Douglas Floriano Scavazzini

Versão:
    1.0 - 12/05/2026
    2.0 - 12/05/2026 - Adição de novas funcionalidades e melhorias na interface do usuário.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st

from streamlit_option_menu import option_menu

from src.core.config_loader import ConfigLoader

from src.analysis.invoice_loader import InvoiceLoader

from src.dashboard.views import (
    overview,
    products,
    temporal,
    database
)


st.set_page_config(
    page_title='Model Supermarket',
    page_icon='🛒',
    layout='wide',
    initial_sidebar_state='collapsed'
)

config = ConfigLoader(
    'config/settings.yaml'
)

loader = InvoiceLoader()

data_path = config.get(
    'paths.data'
)

df = loader.load(data_path)

selected = option_menu(

    menu_title=None,

    options=[
        'Overview',
        'Temporal',
        'Products',
        'Database'
    ],

    icons=[
        'house',
        'calendar',
        'cart',
        'database'
    ],

    orientation='horizontal'
)

if selected == 'Overview':

    overview.render(df)

elif selected == 'Temporal':

    temporal.render(df)

elif selected == 'Products':

    products.render(df)

elif selected == 'Database':

    database.render(df)