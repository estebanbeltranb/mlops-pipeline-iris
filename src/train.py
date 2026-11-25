"""
train.py - Paso 3 del pipeline: entrenamiento del modelo

Entrena un Random Forest con los datos validados (data/processed/),
registra el experimento en MLflow y guarda el modelo en models/model.pkl.
Basado en el código de la Semana 2 del portal del curso.

Se usa la configuración elegida en la Actividad 1.1 (Run_1_n50_depth3):
50 árboles y profundidad máxima 3, el modelo más simple y con menor sobreajuste.
"""

import os  # para crear carpetas (viene con Python)

import joblib                                     # para guardar el modelo en un archivo
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- Hiperparámetros del modelo ---
N_ESTIMATORS = 50  # número de árboles del bosque
MAX_DEPTH = 3      # profundidad máxima de cada árbol
SEMILLA = 42       # semilla fija para que los resultados sean reproducibles

# 1. Configurar MLflow: dónde se guarda el registro y en qué experimento
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Proyecto_Final_Grado")

with mlflow.start_run(run_name="Pipeline_n50_depth3"):
    # 2. Registrar los hiperparámetros
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("random_state", SEMILLA)

    # 3. Cargar los datos de entrenamiento (ya validados)
    train = pd.read_csv("data/processed/train.csv")
    X_train = train.drop("species", axis=1)  # las 4 medidas
    y_train = train["species"]               # la especie

    # 4. Entrenar el modelo
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=SEMILLA
    )
    model.fit(X_train, y_train)

    # 5. Evaluar con los datos de prueba
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("species", axis=1)
    y_test = test["species"]

    accuracy = model.score(X_test, y_test)           # accuracy en prueba
    train_accuracy = model.score(X_train, y_train)   # accuracy en entrenamiento
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("train_accuracy", train_accuracy)

    # 6. Guardar el modelo en MLflow (como artefacto)
    mlflow.sklearn.log_model(
        model, name="iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # 7. Guardar el modelo también como archivo, para las pruebas del pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print(f"Modelo entrenado con accuracy: {accuracy}")
    print(f"Accuracy de entrenamiento: {train_accuracy:.3f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")