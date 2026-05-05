"""
Descrição:
    Módulo responsável pelo carregamento, gerenciamento e acesso a configurações
    de aplicação a partir de arquivos YAML, incluindo suporte a sobrescrita dinâmica
    de parâmetros.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 29/04/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """
    Descrição:
        Classe responsável pelo carregamento e gerenciamento de configurações em formato YAML,
        permitindo acesso estruturado e sobrescrita de parâmetros.

    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 29/04/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """

    def __init__(self, config_path: str = 'config/config.yaml', override_params: Dict[str, Any] = None):
        """
        Descrição:
            Inicializa o carregador de configuração e aplica possíveis sobrescritas.

        Parâmetros:
            config_path (str): Caminho para o arquivo de configuração YAML.
            override_params (Dict[str, Any]): Dicionário de parâmetros para sobrescrita.

        Referências:
            ---
        """
        self.config_path = config_path
        self.config = self._load_yaml()

        if override_params:
            self._override_config(override_params)

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Descrição:
            Carrega os dados de configuração a partir de um arquivo YAML.

        Parâmetros:
            ---

        Referências:
            - Evans, E. (2003). Domain-Driven Design.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f'Arquivo de configuração não encontrado: {self.config_path}')

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config if config else {}

    def _override_config(self, overrides: Dict[str, Any]) -> None:
        """
        Descrição:
            Realiza sobrescrita recursiva de parâmetros de configuração com base em um
            dicionário de overrides.

        Parâmetros:
            overrides (Dict[str, Any]): Dicionário contendo os valores a serem sobrescritos.

        Referências:
            - Fowler, M. (2018). Patterns of Enterprise Application Architecture.
        """
        def deep_update(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = deep_update(self.config, overrides)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Descrição:
            Recupera um valor da configuração utilizando notação de chave hierárquica.

        Parâmetros:
            key (str): Chave no formato hierárquico (ex: "database.host").
            default (Any): Valor padrão caso a chave não exista.

        Referências:
            - Fowler, M. (2018). Patterns of Enterprise Application Architecture.
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        """
        Descrição:
            Permite acesso aos valores de configuração utilizando sintaxe de colchetes.

        Parâmetros:
            key (str): Chave de acesso.

        Referências:
            ---
        """
        return self.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """
        Descrição:
            Retorna toda a configuração carregada como um dicionário.

        Parâmetros:
            ---

        Referências:
            ---
        """
        return self.config.copy()


def get_default_config() -> ConfigLoader:
    """
    Descrição:
        Retorna uma instância padrão do carregador de configuração utilizando o caminho padrão.

    Parâmetros:
        ---

    Referências:
        ---
    """
    return ConfigLoader()