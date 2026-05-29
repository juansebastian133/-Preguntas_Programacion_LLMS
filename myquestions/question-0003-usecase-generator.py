import numpy as np
import pandas as pd
import random
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def entrenar_pronostico_lasso(df, target_col, alpha_lasso):
    df = df.copy()
    df["ventas_ayer"] = df[target_col].shift(1)
    df = df.dropna()
    X = df[["ventas_ayer"]]
    y = df[target_col]
    modelo = Lasso(alpha=alpha_lasso)
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)
    return modelo, r2

# =========================================================
# GENERADOR DE CASOS DE USO
# =========================================================
def generar_caso_de_uso_entrenar_pronostico_lasso():
    n = random.randint(30, 100)
    ventas = np.cumsum(np.random.randn(n) * 10 + 50)
    df = pd.DataFrame({"ventas": ventas})
    target_col = "ventas"
    alpha_lasso = round(random.uniform(0.01, 1.0), 2)

    input_data = {
        "df": df,
        "target_col": target_col,
        "alpha_lasso": alpha_lasso
    }

    modelo, r2 = entrenar_pronostico_lasso(df, target_col, alpha_lasso)
    output_data = {
        "coeficiente": modelo.coef_.tolist(),
        "intercepto": modelo.intercept_,
        "r2": r2
    }

    return input_data, output_data

# =========================================================
# EJECUCIÓN DE PRUEBA
# =========================================================
if _name_ == "_main_":
    input_data, output_data = generar_caso_de_uso_entrenar_pronostico_lasso()
    print("===== INPUT GENERADO =====")
    print("Alpha:", input_data["alpha_lasso"])
    print("Primeras filas del DataFrame:")
    print(input_data["df"].head())
    print("\n===== OUTPUT ESPERADO =====")
    print(output_data)
