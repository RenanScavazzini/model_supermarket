"""
Descrição:
    Módulo responsável pela configuração e gerenciamento de logging da aplicação,
    permitindo registro de eventos em console e arquivo para monitoramento e debugging.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 29/04/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import logging
import os
from pathlib import Path


class Logger:
    """
    Descrição:
        Classe responsável pela criação e configuração de um logger padronizado para a aplicação,
        com suporte a saída em console e persistência em arquivo.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    def __init__(self, name: str = 'model_supermarket', log_dir: str = 'results/logs', level: str = 'INFO'):
        """
        Descrição:
            Inicializa o logger com nome, diretório de logs e nível de severidade.

        Parâmetros:
            name (str): Nome do logger.
            log_dir (str): Diretório onde os arquivos de log serão armazenados.
            level (str): Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).

        Referências:
            ---
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.logger = self._create_logger()

    def _create_logger(self) -> logging.Logger:
        """
        Descrição:
            Cria e configura um logger com handlers para saída em console e escrita em arquivo,
            incluindo definição de nível de log e formatação das mensagens.

        Parâmetros:
            ---

        Referências:
            - Gamma, E. et al. (1994). Design Patterns.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level.upper(), logging.INFO))

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_format)

            file_handler = logging.FileHandler(self.log_dir / 'app.log', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(console_format)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        """
        Descrição:
            Retorna a instância configurada do logger para uso na aplicação.

        Parâmetros:
            ---

        Referências:
            ---
        """
        return self.logger