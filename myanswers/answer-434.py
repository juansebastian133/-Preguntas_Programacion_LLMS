import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

def pipeline_seleccion_modelo(X, y, k):
    # Escalar variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Seleccionar las k mejores variables
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    # Obtener nombres de las variables seleccionadas
    selected_indices = selector.get_support(indices=True)
    selected_features = [X.columns[i] for i in selected_indices]

    # Entrenar modelo
    model = RandomForestClassifier()
    model.fit(X_selected, y)

    # Importancias
    importancias = model.feature_importances_

    return {
        "variables_seleccionadas": selected_features,
        "importancias": importancias
    }
