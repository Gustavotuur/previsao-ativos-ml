"""Configuracao do pytest: permite importar os modulos do projeto nos testes."""
import importlib
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


@pytest.fixture(scope="session")
def prep():
    """Modulo 02_preparar_variaveis (o nome comeca com numero, por isso o importlib)."""
    return importlib.import_module("02_preparar_variaveis")


@pytest.fixture(scope="session")
def modelar():
    return importlib.import_module("03_modelar")
