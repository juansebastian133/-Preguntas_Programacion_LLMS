import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def optimizar_svm_grid(X, y, lista_c, cv_folds):
    modelo = SVC()

    param_grid = {'C': lista_c}

    grid = GridSearchCV(modelo, param_grid, cv=cv_folds)
    grid.fit(X, y)

    resultado = {
        'mejor_c': grid.best_params_['C'],
        'mejor_score': grid.best_score_,
        'modelo_final': grid.best_estimator_
    }

    return resultado

import numpy as np
import random
from sklearn.datasets import make_classification

def generar_caso_de_uso_preparar_datos_1():
    # 🔹 1. Generar dataset aleatorio
    n_muestras = random.randint(50, 200)
    n_features = random.randint(5, 20)

    X, y = make_classification(
        n_samples=n_muestras,
        n_features=n_features,
        n_informative=random.randint(2, n_features),
        n_classes=2
    )

    # 🔹 2. Generar lista de C aleatoria
    lista_c = [round(random.uniform(0.1, 10), 2) for _ in range(random.randint(3, 6))]

    # 🔹 3. Generar número de folds
    cv_folds = random.randint(3, 5)

    # 🔹 4. Crear input
    input_data = {
        'X': X,
        'y': y,
        'lista_c': lista_c,
        'cv_folds': cv_folds
    }

    # 🔹 5. Ejecutar la función real
    output_data = optimizar_svm_grid(X, y, lista_c, cv_folds)

    return input_data, output_data

inp, out = generar_caso_de_uso_preparar_datos_1()

print("INPUT:")
print(inp)

print("\nOUTPUT:")
print(out)
