"""
Descrição:
    Página do modelo estatístico do dashboard.

Autor:
    Renan Douglas Floriano Scavazzini

Versão:
    1.0 - 13/05/2026 - Estrutura inicial da página.

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import streamlit as st
import pandas as pd

from src.core.logger import setup_logger


logger = setup_logger(__name__)


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

    st.title(
        '📈 Modelo Estatístico'
    )

    st.markdown(
        """
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            height:65vh;
            width:100%;
        ">
            <div style="
                font-size:72px;
                font-weight:800;
                color:white;
                opacity:0.85;
                text-align:center;
            ">
                🚧 Em Construção...
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    logger.info(
        'Página Modelo Estatístico renderizada com sucesso'
    )