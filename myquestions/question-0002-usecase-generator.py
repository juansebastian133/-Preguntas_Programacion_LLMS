import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

# =========================================================
# 1. FUNCIÓN PRINCIPAL (la que pide el problema)
# =========================================================
def procesar_datos_financieros(df: pd.DataFrame, n_componentes: int) -> np.ndarray:
    # 🔹 1. Limpiar datos (rellenar NaN con media)
    df_limpio = df.copy()
    df_limpio = df_limpio.fillna(df_limpio.mean())

    # 🔹 2. Transformación para outliers (PowerTransformer)
    pt = PowerTransformer()
    datos_transformados = pt.fit_transform(df_limpio)

    # 🔹 3. Reducción de dimensionalidad (PCA)
    pca = PCA(n_components=n_componentes)
    datos_finales = pca.fit_transform(datos_transformados)

    return datos_finales


# =========================================================
# 2. GENERADOR DE CASOS DE USO ALEATORIOS
# =========================================================
def generar_caso_de_uso_preparar_datos_2():
    # 🔹 Tamaño aleatorio del dataset
    n_filas = random.randint(20, 100)
    n_columnas = random.randint(3, 8)

    # 🔹 Generar datos con colas largas (simula datos financieros)
    datos = np.random.exponential(scale=50, size=(n_filas, n_columnas))

    # 🔹 Introducir algunos outliers extremos
    for _ in range(random.randint(1, 5)):
        i = random.randint(0, n_filas - 1)
        j = random.randint(0, n_columnas - 1)
        datos[i, j] *= random.randint(10, 50)

    # 🔹 Introducir valores faltantes (NaN)
    for _ in range(random.randint(1, 5)):
        i = random.randint(0, n_filas - 1)
        j = random.randint(0, n_columnas - 1)
        datos[i, j] = np.nan

    # 🔹 Crear DataFrame
    columnas = [f"col_{i}" for i in range(n_columnas)]
    df = pd.DataFrame(datos, columns=columnas)

    # 🔹 Número de componentes válido
    n_componentes = random.randint(1, n_columnas)

    # 🔹 Crear input
    input_data = {
        "df": df,
        "n_componentes": n_componentes
    }

    # 🔹 Generar output esperado
    output_data = procesar_datos_financieros(df, n_componentes)

    return input_data, output_data


# =========================================================
# 3. FUNCIÓN PARA COMPROBAR EL CASO DE USO
# =========================================================
def comprobar_caso_de_uso(input_data, output_esperado):
    df = input_data["df"]
    n_componentes = input_data["n_componentes"]

    resultado = procesar_datos_financieros(df, n_componentes)

    # 🔹 Comparar resultados (con tolerancia numérica)
    if np.allclose(resultado, output_esperado, atol=1e-6):
        print("✅ El caso de uso es CORRECTO")
    else:
        print("❌ El caso de uso es INCORRECTO")

    print("\n--- INPUT ---")
    print(df.head())
    print(f"n_componentes: {n_componentes}")

    print("\n--- OUTPUT ESPERADO (shape) ---")
    print(output_esperado.shape)

    print("\n--- OUTPUT OBTENIDO (shape) ---")
    print(resultado.shape)


# =========================================================
# 4. EJECUCIÓN DE PRUEBA
# =========================================================
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_preparar_datos_2()
    comprobar_caso_de_uso(input_data, output_data)
