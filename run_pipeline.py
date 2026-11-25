"""
run_pipeline.py - Script de orquestación del pipeline (Actividad 2)
Seminario de Grado II

Ejecuta TODO el pipeline con UN SOLO COMANDO:

    python run_pipeline.py

Pasos, en orden:
    1. Preparación de datos  -> src/prepare.py
    2. Validación de datos   -> src/validate.py   (si falla, el pipeline se detiene)
    3. Entrenamiento         -> src/train.py
    4. Pruebas automáticas   -> pytest tests/

Cada ejecución queda registrada con fecha y hora en logs/pipeline.log,
indicando qué pasos se completaron y cuáles fallaron (trazabilidad).

Se usa un script de Python en lugar de un Makefile porque funciona igual en
Windows, Mac y Linux sin instalar herramientas adicionales.
"""

import os                       # crear carpetas y leer variables del sistema
import subprocess               # ejecutar otros scripts desde este
import sys                      # sys.executable = el Python del entorno virtual activo
from datetime import datetime   # fecha y hora para el registro (log)

# Lista de pasos del pipeline: (nombre que se muestra, comando a ejecutar)
PASOS = [
    ("Preparación de datos", [sys.executable, "src/prepare.py"]),
    ("Validación de datos",  [sys.executable, "src/validate.py"]),
    ("Entrenamiento",        [sys.executable, "src/train.py"]),
    ("Pruebas automáticas",  [sys.executable, "-m", "pytest", "tests/", "-v"]),
]

RUTA_LOG = "logs/pipeline.log"

# Evita errores al mostrar tildes o símbolos especiales en la terminal de Windows
sys.stdout.reconfigure(errors="replace")


def registrar(log, mensaje):
    """Muestra un mensaje en pantalla y lo guarda en el log con fecha y hora."""
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {mensaje}"
    print(linea)
    log.write(linea + "\n")


def ejecutar_paso(log, numero, nombre, comando):
    """Ejecuta un paso del pipeline. Devuelve True si terminó bien, False si falló."""
    registrar(log, f"PASO {numero}/{len(PASOS)}: {nombre} -> INICIO")

    # Configuración del entorno: textos en UTF-8 y sin el aviso de Git de MLflow
    entorno = dict(os.environ, PYTHONIOENCODING="utf-8", GIT_PYTHON_REFRESH="quiet")

    # Ejecutar el paso y capturar todo lo que imprime
    resultado = subprocess.run(comando, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=entorno)

    # Mostrar la salida del paso en pantalla y guardarla en el log
    salida = (resultado.stdout + resultado.stderr).strip()
    for linea in salida.splitlines():
        print(f"    {linea}")
        log.write(f"    {linea}\n")

    # returncode 0 = éxito; cualquier otro valor = error
    if resultado.returncode == 0:
        registrar(log, f"PASO {numero}/{len(PASOS)}: {nombre} -> OK")
        return True
    registrar(log, f"PASO {numero}/{len(PASOS)}: {nombre} -> FALLÓ (código {resultado.returncode})")
    return False


def main():
    os.makedirs("logs", exist_ok=True)

    # "a" = agregar al final: el log conserva el historial de todas las ejecuciones
    with open(RUTA_LOG, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 70 + "\n")
        registrar(log, "PIPELINE INICIADO")

        for numero, (nombre, comando) in enumerate(PASOS, start=1):
            if not ejecutar_paso(log, numero, nombre, comando):
                registrar(log, "PIPELINE DETENIDO: corrija el error y vuelva a ejecutar.")
                sys.exit(1)  # se detiene en el primer paso que falle

        registrar(log, "PIPELINE COMPLETADO CON ÉXITO")

    print(f"\nRegistro guardado en {RUTA_LOG}")


if __name__ == "__main__":
    main()