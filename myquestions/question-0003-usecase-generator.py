import numpy as np
import pandas as pd
import random
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# =========================================================
# FUNCIÓN PRINCIPAL (la que te piden)
# =========================================================
def entrenar_pronostico_lasso(df, target_col, alpha_lasso):
    # Crear variable de lag (ventas del día anterior)
    df = df.copy()
    df["ventas_ayer"] = df[target_col].shift(1)

    # Eliminar valores nulos
    df = df.dropna()

    # Separar variables
    X = df[["ventas_ayer"]]
    y = df[target_col]

    # Entrenar modelo Lasso
    modelo = Lasso(alpha=alpha_lasso)
    modelo.fit(X, y)

    # Calcular R^2
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)

    return modelo, r2


# =========================================================
# GENERADOR DE CASOS DE USO (lo que te piden en el ejercicio)
# =========================================================
def generar_caso_de_uso_preparar_datos():
    # 1. Generar tamaño aleatorio
    n = random.randint(30, 100)

    # 2. Generar serie de ventas simulada (tipo serie temporal)
    ventas = np.cumsum(np.random.randn(n) * 10 + 50)

    df = pd.DataFrame({
        "ventas": ventas
    })

    # 3. Parámetros aleatorios
    target_col = "ventas"
    alpha_lasso = round(random.uniform(0.01, 1.0), 2)

    # 4. Crear input
    input_data = {
        "df": df,
        "target_col": target_col,
        "alpha_lasso": alpha_lasso
    }

    # 5. Generar output esperado usando la función real
    modelo, r2 = entrenar_pronostico_lasso(df, target_col, alpha_lasso)

    output_data = {
        "coeficiente": modelo.coef_.tolist(),
        "intercepto": modelo.intercept_,
        "r2": r2
    }

    return input_data, output_data


# =========================================================
# FUNCIÓN PARA COMPROBAR EL CASO DE USO
# =========================================================
def comprobar_caso_de_uso(input_data, output_esperado):
    modelo, r2 = entrenar_pronostico_lasso(
        input_data["df"],
        input_data["target_col"],
        input_data["alpha_lasso"]
    )

    print("===== RESULTADO ESPERADO =====")
    print(output_esperado)

    print("\n===== RESULTADO OBTENIDO =====")
    print({
        "coeficiente": modelo.coef_.tolist(),
        "intercepto": modelo.intercept_,
        "r2": r2
    })


# =========================================================
# EJECUCIÓN DE PRUEBA
# =========================================================
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_preparar_datos()

    print("===== INPUT GENERADO =====")
    print("Alpha:", input_data["alpha_lasso"])
    print("Primeras filas del DataFrame:")
    print(input_data["df"].head())

    print("\n===== OUTPUT ESPERADO =====")
    print(output_data)

    print("\n===== VALIDACIÓN =====")
    comprobar_caso_de_uso(input_data, output_data)