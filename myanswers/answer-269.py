import pandas as pd
import numpy as np

def extract_hrv_metrics(rr_intervals_df, threshold_ms=2000):
    # 1. Filtrar artefactos fisiológicos
    df_filtered = rr_intervals_df[
        (rr_intervals_df["RR_ms"] >= 300) &
        (rr_intervals_df["RR_ms"] <= threshold_ms)
    ].copy()

    # 2. Diferencias sucesivas usando .diff()
    diffs = df_filtered["RR_ms"].diff().dropna()

    # 3. Calcular RMSSD
    rmssd = np.sqrt(np.mean(diffs ** 2))

    # 4. Calcular SDNN
    sdnn = np.std(df_filtered["RR_ms"], ddof=0)

    # 5. Retornar resultados
    return {
        "RMSSD": float(rmssd),
        "SDNN": float(sdnn),
        "valid_samples_count": int(len(df_filtered))
    }
