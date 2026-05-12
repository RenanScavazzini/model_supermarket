"""
Descrição:
    Módulo responsável pela configuração e gerenciamento de logging da aplicação,
    permitindo registro de eventos em console e arquivo para monitoramento,
    debugging e rastreabilidade analítica.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    2.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import logging

from pathlib import Path


class Logger:
    """
    Descrição:
        Classe responsável pela criação e configuração de loggers padronizados
        para a aplicação, com suporte a saída em console e persistência em arquivo.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        2.0 - 12/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    def __init__(
        self,
        name: str = 'model_supermarket',
        log_dir: str = 'logs',
        level: str = 'INFO'
    ):
        """
        Descrição:
            Inicializa o logger com nome, diretório de logs e nível de severidade.

        Parâmetros:
            name (str): Nome do logger.
            log_dir (str): Diretório onde os arquivos de log serão armazenados.
            level (str): Nível de logging.

        Referências:
            ---
        """

        self.name = name

        self.log_dir = Path(log_dir)

        self.level = level

        self.logger = self._create_logger()

    def _create_logger(
        self
    ) -> logging.Logger:
        """
        Descrição:
            Cria e configura logger com handlers para console e arquivo.

        Parâmetros:
            ---

        Referências:
            - Gamma, E. et al. (1994). Design Patterns.
        """

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        logger = logging.getLogger(
            self.name
        )

        logger.setLevel(
            getattr(
                logging,
                self.level.upper(),
                logging.INFO
            )
        )

        logger.propagate = False

        if not logger.handlers:

            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )

            console_handler = logging.StreamHandler()

            console_handler.setLevel(
                logging.DEBUG
            )

            console_handler.setFormatter(
                formatter
            )

            file_handler = logging.FileHandler(
                self.log_dir / 'app.log',
                encoding='utf-8'
            )

            file_handler.setLevel(
                logging.DEBUG
            )

            file_handler.setFormatter(
                formatter
            )

            logger.addHandler(
                console_handler
            )

            logger.addHandler(
                file_handler
            )

        return logger

    def get_logger(
        self
    ) -> logging.Logger:
        """
        Descrição:
            Retorna a instância configurada do logger.

        Parâmetros:
            ---

        Referências:
            ---
        """

        return self.logger


def setup_logger(
    name: str = 'model_supermarket',
    log_dir: str = 'logs',
    level: str = 'INFO'
) -> logging.Logger:
    """
    Descrição:
        Função utilitária responsável por criar e retornar
        uma instância padronizada de logger.

    Parâmetros:
        name (str): Nome do logger.
        log_dir (str): Diretório de armazenamento dos logs.
        level (str): Nível de logging.

    Referências:
        - Python Software Foundation. Logging Documentation.
    """

    return Logger(
        name=name,
        log_dir=log_dir,
        level=level
    ).get_logger()