import numpy as np
import random
from collections import defaultdict

# =========================================================
# 1. FUNCIÓN PRINCIPAL A EVALUAR
# =========================================================
def reporte_especifico_clases(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula la precisión por clase usando matriz de confusión normalizada por fila.
    """
    clases = np.unique(y_true)
    matriz = defaultdict(lambda: defaultdict(int))

    # Construcción de matriz de confusión
    for real, pred in zip(y_true, y_pred):
        matriz[real][pred] += 1

    reporte = {}

    # Calcular precisión por clase (recall por fila)
    for clase in clases:
        total_reales = sum(matriz[clase].values())
        verdaderos_positivos = matriz[clase][clase]

        if total_reales == 0:
            precision = 0
        else:
            precision = verdaderos_positivos / total_reales

        reporte[f"Clase_{clase}"] = round(precision, 3)

    return reporte


# =========================================================
# 2. GENERADOR DE CASOS DE USO ALEATORIOS
# =========================================================
def generar_caso_de_uso_preparar_datos():
    """
    Genera inputs aleatorios y su output esperado para probar la función.
    """

    # 🔹 Clases posibles (3 cultivos)
    clases = [0, 1, 2]

    # 🔹 Tamaño aleatorio
    n = random.randint(30, 100)

    # 🔹 Generar y_true desbalanceado
    y_true = np.random.choice(
        clases,
        size=n,
        p=np.random.dirichlet(np.ones(3))  # distribución aleatoria
    )

    # 🔹 Generar y_pred con cierto error
    y_pred = []
    for val in y_true:
        if random.random() < random.uniform(0.6, 0.9):
            y_pred.append(val)  # predicción correcta
        else:
            y_pred.append(random.choice(clases))  # error

    y_pred = np.array(y_pred)

    # 🔹 Input esperado
    input_data = {
        "y_true": y_true,
        "y_pred": y_pred
    }

    # 🔹 Output esperado usando la función real
    output_data = reporte_especifico_clases(y_true, y_pred)

    return input_data, output_data


# =========================================================
# 3. FUNCIÓN PARA COMPROBAR AUTOMÁTICAMENTE
# =========================================================
def probar_caso():
    input_data, output_esperado = generar_caso_de_uso_preparar_datos()

    resultado = reporte_especifico_clases(
        input_data["y_true"],
        input_data["y_pred"]
    )

    print("========== INPUT ==========")
    print("y_true:", input_data["y_true"])
    print("y_pred:", input_data["y_pred"])

    print("\n======= OUTPUT ESPERADO =======")
    print(output_esperado)

    print("\n======= OUTPUT OBTENIDO =======")
    print(resultado)

    print("\n======= ¿SON IGUALES? =======")
    print(resultado == output_esperado)


# =========================================================
# 4. EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    probar_caso()