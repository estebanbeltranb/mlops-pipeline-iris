"""
test_model.py - Paso 4 del pipeline: pruebas automáticas con pytest

Verifica que el pipeline haya funcionado correctamente:
  - Los datos procesados existen y tienen la columna objetivo.
  - El modelo entrenado se guardó y tiene un accuracy mínimo de 0.8.
  - La validación acepta la entrada válida del contrato y rechaza la inválida.
Basado en el código de la Semana 2 del portal del curso.

Uso:
    python -m pytest tests/ -v
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Permite importar las funciones de src/validate.py desde esta carpeta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from validate import validar_registro  # noqa: E402

# Cargar el contrato de datos una sola vez para las pruebas
with open("data_contract.json", encoding="utf-8") as archivo:
    CONTRATO = json.load(archivo)


def test_data_loading():
    """Los datos procesados existen, no están vacíos y tienen la columna 'species'."""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    assert not train.empty and not test.empty
    assert "species" in train.columns


def test_model_exists():
    """El modelo entrenado se guardó correctamente."""
    model = joblib.load("models/model.pkl")
    assert model is not None


def test_model_accuracy():
    """El modelo alcanza un accuracy mínimo de 0.8 en los datos de prueba."""
    model = joblib.load("models/model.pkl")
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("species", axis=1)
    y_test = test["species"]

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.8


def test_contrato_acepta_entrada_valida():
    """La validación acepta el ejemplo de entrada válida del contrato (0 errores)."""
    errores = validar_registro(CONTRATO["ejemplo_entrada_valida"], CONTRATO, con_objetivo=False)
    assert errores == []


def test_contrato_rechaza_entrada_invalida():
    """La validación rechaza el ejemplo inválido y detecta sus 4 errores."""
    errores = validar_registro(CONTRATO["ejemplo_entrada_invalida"], CONTRATO, con_objetivo=False)
    assert len(errores) == 4