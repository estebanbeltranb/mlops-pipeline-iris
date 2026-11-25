# Pipeline MLOps reproducible: clasificador de flores Iris

**Actividad 2 – Seminario de Grado II**
Especialización en Ciencia de Datos · Universidad Santo Tomás, Seccional Tunja
Autor: **Esteban Beltrán Barrera** · 2026

---

## Tabla de contenido

1. [Introducción](#1-introducción)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Requisitos previos](#3-requisitos-previos)
4. [Instalación paso a paso](#4-instalación-paso-a-paso)
5. [Ejecución del pipeline](#5-ejecución-del-pipeline)
6. [Metodología](#6-metodología)
7. [Contrato de datos](#7-contrato-de-datos)
8. [Resultados](#8-resultados)
9. [Discusión y trabajo futuro](#9-discusión-y-trabajo-futuro)
10. [Solución de problemas](#10-solución-de-problemas)

---

## 1. Introducción

**Problema.** En la fase experimental de un proyecto de ciencia de datos, el modelo se entrena ejecutando scripts a mano. Este proceso es propenso a errores humanos, difícil de repetir y no garantiza que los datos de entrada sean correctos.

**Objetivo.** Construir un **pipeline único** que, con un solo comando, prepare los datos, valide su calidad, entrene un modelo de clasificación y verifique su funcionamiento, dejando un registro trazable de cada ejecución.

**Caso de estudio.** Se usa el dataset **Iris** (150 flores, 3 especies) para clasificar la especie de una flor a partir de cuatro medidas de sus sépalos y pétalos. Es el mismo caso de la Actividad 1.1, lo que permite mantener la coherencia del proyecto.

---

## 2. Estructura del proyecto

```
Actividad_2_Pipeline/
├── src/
│   ├── prepare.py          # Paso 1: carga los datos y los divide en train/test
│   ├── validate.py         # Paso 2: valida los datos contra el contrato (Python puro)
│   └── train.py            # Paso 3: entrena el modelo y lo registra en MLflow
├── tests/
│   └── test_model.py       # Paso 4: pruebas automáticas con pytest
├── data_contract.json      # Contrato de datos: tipos, rangos y ejemplos válido/inválido
├── run_pipeline.py         # Script de orquestación: ejecuta todo con un solo comando
├── requirements.txt        # Dependencias con versiones fijas
├── .gitignore              # Archivos que no se suben al repositorio
└── README.md               # Este documento
```

Al ejecutar el pipeline se generan automáticamente: `data/processed/` (datos), `models/` (modelo entrenado), `logs/` (registro de ejecución), `mlruns/` y `mlflow.db` (registro de MLflow).

---

## 3. Requisitos previos

| Herramienta | Versión | Uso |
|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.11 o superior | Ejecutar el proyecto |
| [Git](https://git-scm.com/downloads) | Cualquiera reciente | Descargar el repositorio |

Verifique que están instalados:

```bash
python --version
git --version
```

---

## 4. Instalación paso a paso

### 4.1 Clonar el repositorio

```bash
git clone https://github.com/estebanbeltranb/mlops-pipeline-iris.git
cd mlops-pipeline-iris
```

### 4.2 Crear y activar el entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Cuando el entorno está activo, la terminal muestra `(venv)` al inicio de la línea.

### 4.3 Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

## 5. Ejecución del pipeline

Con el entorno virtual activo, ejecute **un solo comando**:

```bash
python run_pipeline.py
```

### Flujo del pipeline

```
python run_pipeline.py
   │
   ├── PASO 1: Preparación     src/prepare.py   → genera data/processed/train.csv y test.csv
   ├── PASO 2: Validación      src/validate.py  → ¿los datos cumplen el contrato?
   │                                                ├── NO → el pipeline SE DETIENE
   │                                                └── SÍ → continúa
   ├── PASO 3: Entrenamiento   src/train.py     → entrena, registra en MLflow y guarda models/model.pkl
   └── PASO 4: Pruebas         pytest tests/    → verifica datos, modelo y contrato
```

### Salida esperada (resumida)

```
[2026-11-25 09:50:30] PIPELINE INICIADO
[2026-11-25 09:50:32] PASO 1/4: Preparación de datos -> OK
[2026-11-25 09:50:32] PASO 2/4: Validación de datos -> OK
[2026-11-25 09:50:39] PASO 3/4: Entrenamiento -> OK
[2026-11-25 09:50:40] PASO 4/4: Pruebas automáticas -> OK
[2026-11-25 09:50:40] PIPELINE COMPLETADO CON ÉXITO

Registro guardado en logs/pipeline.log
```

### Registro de ejecución (logs)

Cada ejecución queda guardada en `logs/pipeline.log` con fecha y hora, indicando qué pasos se completaron, cuáles fallaron y la salida de cada uno. El archivo conserva el historial de todas las ejecuciones.

### Ejecutar los pasos por separado

| Paso | Comando |
|---|---|
| Preparar datos | `python src/prepare.py` |
| Validar datos | `python src/validate.py` |
| Entrenar | `python src/train.py` |
| Pruebas | `python -m pytest tests/ -v` |

### Ver los resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra <http://127.0.0.1:5000> en el navegador, entre al experimento **Proyecto_Final_Grado** y seleccione la vista **Entrenamiento de modelos** (*Model training*).

---

## 6. Metodología

### 6.1 Datos

| Característica | Valor |
|---|---|
| Fuente | Dataset Iris, incluido en scikit-learn (público) |
| Tipo | Datos estructurados (tabulares) |
| Tamaño | 150 registros, 50 por especie (balanceado) |
| Variables de entrada | 4 numéricas continuas, en centímetros: largo y ancho del sépalo y del pétalo |
| Variable objetivo | `species`: 0 = setosa, 1 = versicolor, 2 = virginica |
| Partición | 80 % entrenamiento (120) / 20 % prueba (30), con `random_state=42` |

### 6.2 Validación de datos

`src/validate.py` revisa cada fila de los datos con las reglas de `data_contract.json`, usando **solo la librería estándar de Python** (`csv`, `json` y `sys`), sin librerías externas:

| Verificación | Regla | Ejemplo de error detectado |
|---|---|---|
| **Tipos de datos** | Las medidas deben ser numéricas y la especie un entero | `sepal length (cm): tipo incorrecto ('abc' no es un número)` |
| **Valores nulos** | Ninguna celda puede estar vacía ni ser `null`/`NaN` | `sepal width (cm): valor nulo` |
| **Rangos lógicos** | Cada medida debe estar entre su mínimo y su máximo | `petal length (cm): fuera de rango (15.0 no está entre 1.0 y 7.0)` |

Si encuentra un error, indica **la fila y la columna** del problema y termina con código `1`, lo que **detiene el pipeline antes de entrenar**.

> **Nota de diseño:** el portal del curso propone Great Expectations para esta validación. Como la actividad exige no usar librerías externas, se implementaron las mismas verificaciones (existencia de columnas, rangos, nulos y valores permitidos de la especie) en Python puro.

### 6.3 Modelado

| Elemento | Valor |
|---|---|
| Algoritmo | Random Forest (`scikit-learn`) |
| `n_estimators` | 50 |
| `max_depth` | 3 |
| `random_state` | 42 |
| Seguimiento | MLflow (experimento `Proyecto_Final_Grado`) |

Se usa la configuración seleccionada en la **Actividad 1.1** (`Run_1_n50_depth3`) por ser la más simple y con menor sobreajuste entre las evaluadas.

### 6.4 Evaluación y pruebas

Se mide el **accuracy** en los datos de prueba y en los de entrenamiento (para detectar sobreajuste). Además, `tests/test_model.py` ejecuta 5 pruebas automáticas con pytest:

| Prueba | Qué verifica |
|---|---|
| `test_data_loading` | Los datos procesados existen, no están vacíos y tienen la columna `species` |
| `test_model_exists` | El modelo entrenado se guardó en `models/model.pkl` |
| `test_model_accuracy` | El modelo alcanza un accuracy mayor a 0.8 en prueba |
| `test_contrato_acepta_entrada_valida` | La validación acepta el ejemplo válido del contrato |
| `test_contrato_rechaza_entrada_invalida` | La validación rechaza el ejemplo inválido y detecta sus 4 errores |

---

## 7. Contrato de datos

`data_contract.json` es el **contrato oficial** que deben cumplir los datos que reciba el modelo, tanto en el entrenamiento como en la **inferencia futura** (por ejemplo, a través de una API).

| Columna | Tipo | Mínimo (cm) | Máximo (cm) | Nulos |
|---|---|---|---|---|
| `sepal length (cm)` | float | 4.0 | 8.0 | No |
| `sepal width (cm)` | float | 2.0 | 4.5 | No |
| `petal length (cm)` | float | 1.0 | 7.0 | No |
| `petal width (cm)` | float | 0.1 | 2.6 | No |
| `species` (solo entrenamiento) | int | — | — | No · valores permitidos: 0, 1, 2 |

**Ejemplo de entrada válida** (aceptada):
```json
{ "sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2 }
```

**Ejemplo de entrada inválida** (rechazada):
```json
{ "sepal length (cm)": "cinco", "sepal width (cm)": null, "petal length (cm)": 15.0 }
```
Contiene cuatro errores: tipo incorrecto, valor nulo, valor fuera de rango y una columna obligatoria ausente (`petal width (cm)`).

Las reglas se leen del contrato y no están escritas en el código: si cambian, basta con actualizar el JSON.

### Prueba de robustez

Para comprobar cómo reacciona el pipeline ante datos que no cumplen el contrato, cambie en `data_contract.json` el `maximo` de `sepal length (cm)` de `8.0` a `5.0` y ejecute `python run_pipeline.py`. La validación reportará las filas fuera de rango y el pipeline se detendrá **sin entrenar el modelo**. Restaure el valor original para volver a la configuración normal.

---

## 8. Resultados

| Métrica | Valor |
|---|---|
| Registros validados | 150 de 150 (120 entrenamiento + 30 prueba), sin errores |
| Accuracy en prueba | **1.000** (30 de 30 flores bien clasificadas) |
| Accuracy en entrenamiento | **0.958** |
| Pruebas automáticas | **5 de 5** superadas |
| Ejemplo válido del contrato | Aceptado |
| Ejemplo inválido del contrato | Rechazado con 4 errores |

---

## 9. Discusión y trabajo futuro

**Interpretación.** El modelo clasificó correctamente todas las flores de prueba. Que el accuracy de entrenamiento (0.958) sea ligeramente menor al de prueba indica que el modelo **no memoriza** los datos, lo que es coherente con la elección de árboles poco profundos (`max_depth = 3`).

**Aporte del pipeline.** Centralizar la ejecución en un solo comando reduce el error humano y garantiza que cualquier persona obtenga los mismos resultados. La validación previa evita entrenar con datos corruptos, y el log y MLflow dejan evidencia trazable de cada ejecución.

**Limitaciones.**
- El conjunto de prueba es pequeño (30 flores): un accuracy de 1.0 no garantiza un desempeño perfecto con datos nuevos.
- Los rangos del contrato se definieron a partir de los valores observados en Iris; con otra población de flores podrían requerir ajustes.
- La validación cubre tipos, nulos y rangos, pero no detecta cambios en la distribución de los datos.

**Trabajo futuro.**
- Detectar **Data Drift** (cambios en la distribución de los datos de entrada).
- Integrar **DVC** para el versionado de datos y modelos, como propone el portal del curso.
- Exponer el modelo mediante una **API** que valide cada solicitud con este mismo contrato.
- Automatizar la ejecución del pipeline en cada cambio del repositorio (integración continua).

---

## 10. Solución de problemas

| Problema | Solución |
|---|---|
| `venv\Scripts\activate` muestra un error de scripts deshabilitados | Ejecute `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` y vuelva a activar |
| `ModuleNotFoundError` | Active el entorno virtual y ejecute `pip install -r requirements.txt` |
| `FileNotFoundError` | Ejecute los comandos desde la carpeta raíz del proyecto |
| La validación falla | Revise en la salida (o en `logs/pipeline.log`) la fila y columna del error |
| MLflow no muestra los runs | Seleccione la vista **Entrenamiento de modelos** (*Model training*) en lugar de GenAI |