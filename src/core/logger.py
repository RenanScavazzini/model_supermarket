import logging
import os
from pathlib import Path


class Logger:
    """Logger configurável para o projeto model_supermarket."""

    def __init__(self, name: str = 'model_supermarket', log_dir: str = 'results/logs', level: str = 'INFO'):
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.logger = self._create_logger()

    def _create_logger(self) -> logging.Logger:
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
        """Retorna o logger configurado."""
        return self.logger
