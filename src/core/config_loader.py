import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Carrega e gerencia arquivos de configuração YAML."""

    def __init__(self, config_path: str = 'config/config.yaml', override_params: Dict[str, Any] = None):
        """Inicializa o carregador de configuração."""
        self.config_path = config_path
        self.config = self._load_yaml()

        if override_params:
            self._override_config(override_params)

    def _load_yaml(self) -> Dict[str, Any]:
        """Carrega a configuração do arquivo YAML."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f'Arquivo de configuração não encontrado: {self.config_path}')

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config if config else {}

    def _override_config(self, overrides: Dict[str, Any]) -> None:
        """Substitui valores na configuração com um dicionário de overrides."""
        def deep_update(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = deep_update(self.config, overrides)

    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração por chave de ponto."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        """Permite acesso à configuração via colchetes."""
        return self.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """Retorna toda a configuração como dicionário."""
        return self.config.copy()


def get_default_config() -> ConfigLoader:
    """Retorna uma instância padrão do carregador de configuração."""
    return ConfigLoader()
