"""
validate.py - Paso 2 del pipeline: validación de datos de entrada

Revisa que los datos cumplan las reglas de data_contract.json usando SOLO
Python estándar (sin librerías externas como pandas o great_expectations):

    1. Tipos de datos correctos   -> las medidas deben ser números y la especie un entero.
    2. Ausencia de valores nulos  -> ninguna celda puede estar vacía.
    3. Rangos lógicos             -> cada medida debe estar entre su mínimo y su máximo.

Qué revisa:
    a) Los archivos data/processed/train.csv y data/processed/test.csv
    b) Los dos ejemplos del contrato: el válido debe pasar y el inválido debe fallar.

Si algo falla, termina con código de error 1 para que el pipeline se detenga
y NO entrene un modelo con datos incorrectos.
"""

import csv   # leer archivos CSV          (viene con Python)
import json  # leer el contrato JSON      (viene con Python)
import sys   # terminar con éxito o error (viene con Python)

RUTA_CONTRATO = "data_contract.json"
ARCHIVOS_A_VALIDAR = ["data/processed/train.csv", "data/processed/test.csv"]

# Textos que se consideran "valor nulo", además de una celda vacía
VALORES_NULOS = {"", "null", "none", "nan", "na"}


def es_nulo(valor):
    """True si el valor está vacío o representa un nulo."""
    return valor is None or str(valor).strip().lower() in VALORES_NULOS


def a_numero(valor):
    """Convierte el valor a número decimal. Si no se puede (ej. 'cinco'), devuelve None."""
    if isinstance(valor, bool):  # True/False no son medidas válidas
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def validar_registro(registro, contrato, con_objetivo):
    """
    Valida UNA fila (una flor) contra el contrato.
    con_objetivo=True exige también la columna 'species' (datos de entrenamiento).
    Devuelve la lista de errores encontrados (vacía = fila válida).
    """
    errores = []

    # --- Las 4 medidas de la flor ---
    for columna, regla in contrato["columnas_entrada"].items():
        if columna not in registro:
            errores.append(f"{columna}: columna obligatoria ausente")
            continue

        valor = registro[columna]

        # Verificación 1: valores nulos
        if es_nulo(valor):
            errores.append(f"{columna}: valor nulo")
            continue

        # Verificación 2: tipo de dato (debe ser número)
        numero = a_numero(valor)
        if numero is None:
            errores.append(f"{columna}: tipo incorrecto ('{valor}' no es un número)")
            continue

        # Verificación 3: rango lógico
        if not (regla["minimo"] <= numero <= regla["maximo"]):
            errores.append(f"{columna}: fuera de rango ({numero} no está entre "
                           f"{regla['minimo']} y {regla['maximo']})")

    # --- La especie (solo en datos de entrenamiento) ---
    if con_objetivo:
        for columna, regla in contrato["columna_objetivo"].items():
            valor = registro.get(columna)
            numero = a_numero(valor)
            if es_nulo(valor):
                errores.append(f"{columna}: valor nulo o ausente")
            elif numero is None or numero != int(numero):
                errores.append(f"{columna}: tipo incorrecto ('{valor}' no es un entero)")
            elif int(numero) not in regla["valores_permitidos"]:
                errores.append(f"{columna}: valor no permitido ({int(numero)}); "
                               f"se esperaba uno de {regla['valores_permitidos']}")

    return errores


def validar_archivo(ruta, contrato):
    """Valida todas las filas de un CSV. Devuelve (filas_revisadas, errores)."""
    errores = []
    try:
        with open(ruta, encoding="utf-8", newline="") as archivo:
            lector = csv.DictReader(archivo)
            filas = 0
            for numero_fila, fila in enumerate(lector, start=2):  # fila 1 = encabezado
                filas += 1
                for error in validar_registro(fila, contrato, con_objetivo=True):
                    errores.append(f"fila {numero_fila} -> {error}")
    except FileNotFoundError:
        return 0, [f"no existe el archivo (¿se ejecutó prepare.py?)"]

    if filas == 0:
        errores.append("el archivo no tiene datos")
    return filas, errores


def main():
    with open(RUTA_CONTRATO, encoding="utf-8") as archivo:
        contrato = json.load(archivo)

    print(f"Validación con '{contrato['nombre']}' (v{contrato['version']})")
    hay_fallas = False

    # a) Validar los archivos de datos
    for ruta in ARCHIVOS_A_VALIDAR:
        filas, errores = validar_archivo(ruta, contrato)
        if errores:
            hay_fallas = True
            print(f"[FALLÓ] {ruta}: {len(errores)} error(es)")
            for error in errores[:10]:  # se muestran máximo 10
                print(f"         - {error}")
        else:
            print(f"[OK]    {ruta}: {filas} filas válidas (tipos, nulos y rangos correctos)")

    # b) Probar los ejemplos del contrato
    errores_validos = validar_registro(contrato["ejemplo_entrada_valida"], contrato, con_objetivo=False)
    if errores_validos:
        hay_fallas = True
        print(f"[FALLÓ] El ejemplo válido fue rechazado: {errores_validos}")
    else:
        print("[OK]    Ejemplo de entrada válida: aceptado")

    errores_invalidos = validar_registro(contrato["ejemplo_entrada_invalida"], contrato, con_objetivo=False)
    if errores_invalidos:
        print(f"[OK]    Ejemplo de entrada inválida: rechazado con {len(errores_invalidos)} errores")
        for error in errores_invalidos:
            print(f"         - {error}")
    else:
        hay_fallas = True
        print("[FALLÓ] El ejemplo inválido fue aceptado: la validación no detectó los errores")

    # Resultado final
    if hay_fallas:
        print("RESULTADO: validación FALLIDA. Corrija los datos antes de entrenar.")
        sys.exit(1)  # código 1 = error -> el pipeline se detiene
    print("RESULTADO: validación SUPERADA. Los datos pueden usarse.")


if __name__ == "__main__":
    main()