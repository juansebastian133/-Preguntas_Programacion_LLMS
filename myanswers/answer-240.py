from sklearn.inspection import permutation_importance

def calcular_importancia_permutacion(modelo, X_val, y_val):
    resultado = permutation_importance(
        modelo,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42
    )

    return resultado.importances_mean
