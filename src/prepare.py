"""
prepare.py - Paso 1 del pipeline: Preparación de los datos

Se carga el dataset Iris, lo divide en entrenamiento (80 %) y prueba (20 %)
y guarda ambos conjuntos como archivos CSV en data/processed/.
Basado en el código de la Semana 2 del portal del curso.
"""

import os  # para crear carpetas (viene con Python)

import pandas as pd                                    # para manejar tablas de datos
from sklearn.datasets import load_iris                 # el dataset de flores Iris
from sklearn.model_selection import train_test_split   # para dividir los datos

# 1. Cargar los datos de Iris y convertirlos en una tabla
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["species"] = iris.target  # especie: 0 = setosa, 1 = versicolor, 2 = virginica

# 2. Dividir en entrenamiento (80 %) y prueba (20 %)
#    random_state=42 fija la división para que siempre salga igual (reproducibilidad)
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 3. Crear la carpeta de destino si no existe y guardar los archivos
os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"Datos preparados: {len(train)} filas de entrenamiento y {len(test)} de prueba")
print("Archivos guardados en data/processed/")