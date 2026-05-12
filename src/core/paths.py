"""
Descrição:
    Módulo responsável pelo gerenciamento centralizado de caminhos
    e diretórios do projeto.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT_DIR / 'config'

DATA_DIR = ROOT_DIR / 'data'

LOGS_DIR = ROOT_DIR / 'logs'

NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'

SRC_DIR = ROOT_DIR / 'src'

DASHBOARD_DIR = SRC_DIR / 'dashboard'

ANALYSIS_DIR = SRC_DIR / 'analysis'

UTILS_DIR = SRC_DIR / 'utils'

STREAMLIT_DIR = ROOT_DIR / '.streamlit'


def ensure_directories() -> None:
    """
    Descrição:
        Garante a existência dos diretórios principais do projeto.

    Parâmetros:
        ---

    Referências:
        ---
    """

    directories = [
        CONFIG_DIR,
        DATA_DIR,
        LOGS_DIR,
        NOTEBOOKS_DIR,
        STREAMLIT_DIR
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )