from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

def pipeline_seleccion_modelo(X, y, k):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    selected_indices = selector.get_support(indices=True)
    selected_features = [X.columns[i] for i in selected_indices]

    model = RandomForestClassifier()
    model.fit(X_selected, y)

    return {
        "variables_seleccionadas": selected_features,
        "importancias": model.feature_importances_
    }
