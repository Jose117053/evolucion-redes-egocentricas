"""
E1: PCA sobre promedios bibliométricos directos.

Objetivo:
  Construir PC1_raw a partir de PCA(mean(X_biblio) por año) y
  compararlo contra PC1_graphsage para medir cuánta información
  aporta realmente la GNN.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


# PCA sobre promedios bibliométricos

def pca_sobre_promedios(df_features: pd.DataFrame,
                        anchor_col: str = "Percent of documents"):
    """
    Aplica PCA de 1 componente sobre los promedios anuales bibliométricos.

    Parámetros
    ----------
    df_features : pd.DataFrame
        DataFrame indexado por año con una columna por feature
        bibliométrica (ya promediada por año, sin ego).
    anchor_col : str
        Columna usada para alinear el signo del PC1 resultante.
        Se alinea para que PC1 tenga correlación positiva con esta
        variable (misma convención que el pipeline GraphSAGE).

    Retorna
    -------
    df_pc1_raw : pd.DataFrame
        Columnas ['year', 'PC1_raw'].
    info : dict
        ev1: varianza explicada por PC1,
        loading: vector de pesos de PC1,
        feature_names: nombres de las columnas usadas.
    """
    years = df_features.index.values
    feature_names = df_features.columns.tolist()
    X = df_features.values.astype(float)

    # Estandarizar antes de PCA para igualar escalas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    pc1_values = pca.fit_transform(X_scaled).flatten()

    ev1 = pca.explained_variance_ratio_[0]
    loading = pca.components_[0]

    # Alinear signo con la columna ancla
    if anchor_col in feature_names:
        r_anchor, _ = pearsonr(pc1_values, df_features[anchor_col].values)
        if r_anchor < 0:
            pc1_values = -pc1_values
            loading = -loading
            print(f"[E1] Se invirtió PC1_raw para alinearlo con '{anchor_col}'. "
                  f"Corr original: {r_anchor:.4f}")
        else:
            print(f"[E1] PC1_raw ya alineado con '{anchor_col}'. "
                  f"Corr: {r_anchor:.4f}")
    else:
        print(f"[E1] Columna ancla '{anchor_col}' no encontrada. "
              "No se realiza alineación de signo.")

    df_pc1_raw = pd.DataFrame({
        "year": years,
        "PC1_raw": pc1_values
    })

    info = {
        "ev1": ev1,
        "loading": loading,
        "feature_names": feature_names,
    }

    return df_pc1_raw, info


# Comparar PC1_raw con PC1_graphsage  

def comparar_pc1(df_pc1_raw: pd.DataFrame,
                 df_pc1_sage: pd.DataFrame,
                 sage_col: str = "y_original"):
    """
    Calcula la correlación de Pearson entre PC1_raw (baseline)
    y PC1_graphsage (pipeline GNN).

    Parámetros
    ----------
    df_pc1_raw : pd.DataFrame
        Columnas ['year', 'PC1_raw'].
    df_pc1_sage : pd.DataFrame
        DataFrame con al menos ['year', sage_col].
    sage_col : str
        Nombre de la columna que contiene PC1 de GraphSAGE.

    Retorna
    -------
    resultado : dict
        rho: correlación de Pearson,
        p_value: p-value de la correlación,
        interpretacion: texto con la interpretación del resultado.
    """
    # Unir por año
    df_merged = pd.merge(df_pc1_raw, df_pc1_sage[["year", sage_col]],
                         on="year", how="inner")

    rho, p_value = pearsonr(df_merged["PC1_raw"].values,
                            df_merged[sage_col].values)

    # Interpretación automática
    abs_rho = abs(rho)
    if abs_rho > 0.90:
        interpretacion = (
            f"MALO: ρ = {rho:.4f} (|ρ| > 0.90). "
            "GraphSAGE básicamente reproduce lo que un PCA directo "
            "sobre promedios ya captura. La GNN es probablemente redundante."
        )
    elif abs_rho > 0.70:
        interpretacion = (
            f"REGULAR: ρ = {rho:.4f} (0.70 < |ρ| < 0.90). "
            "Hay superposición parcial, pero GraphSAGE podría capturar "
            "algo extra. Se necesitan más pruebas (E2, E3, baselines topológicos)."
        )
    else:
        interpretacion = (
            f"FAVORABLE: ρ = {rho:.4f} (|ρ| < 0.70). "
            "GraphSAGE captura información topológica genuina que los "
            "promedios bibliométricos no contienen."
        )

    resultado = {
        "rho": rho,
        "p_value": p_value,
        "interpretacion": interpretacion,
    }

    return resultado, df_merged


# Reporte completo del experimento E1

def ejecutar_e1(df_features: pd.DataFrame,
                df_pc1_sage: pd.DataFrame,
                anchor_col: str = "Percent of documents",
                sage_col: str = "y_original"):
    """
    Ejecuta el experimento E1 completo:
      1. PCA sobre promedios bibliométricos → PC1_raw
      2. Compara PC1_raw vs PC1_graphsage
      3. Imprime reporte detallado

    Parámetros
    ----------
    df_features : pd.DataFrame
        Promedios anuales (indexado por año, sin ego).
    df_pc1_sage : pd.DataFrame
        DataFrame con ['year', sage_col] del pipeline GraphSAGE.
    anchor_col : str
        Variable para alinear el signo de ambos PC1.
    sage_col : str
        Nombre de la columna de PC1 en df_pc1_sage.

    Retorna
    -------
    reporte : dict
        pc1_raw_info: info del PCA (ev1, loading, feature_names),
        comparacion: resultado de la comparación (rho, p_value, interpretacion),
        df_pc1_raw: DataFrame con ['year', 'PC1_raw'],
        df_merged: DataFrame con ambas PC1 unidas por año.
    """
    print("=" * 65)
    print("  EXPERIMENTO E1: PCA sobre promedios bibliométricos directos")
    print("=" * 65)
    print()

    # PCA sobre promedios
    print("─" * 65)
    print("  Paso 1: PCA(mean(X_biblio) por año) → PC1_raw")
    print("─" * 65)
    df_pc1_raw, pc1_info = pca_sobre_promedios(df_features, anchor_col)

    print(f"\n  Varianza explicada por PC1_raw (EV1): {pc1_info['ev1']:.4f}")
    print(f"  Features usadas: {pc1_info['feature_names']}")
    print(f"  Loadings de PC1_raw:")
    for name, w in zip(pc1_info["feature_names"], pc1_info["loading"]):
        print(f"    {name:35s}: {w:+.4f}")
    print()

    # Comparación
    print("─" * 65)
    print("  Paso 2: corr(PC1_raw, PC1_graphsage)")
    print("─" * 65)
    comp, df_merged = comparar_pc1(df_pc1_raw, df_pc1_sage, sage_col)

    print(f"\n  Correlación de Pearson (ρ): {comp['rho']:.4f}")
    print(f"  p-value:                    {comp['p_value']:.6f}")
    print()
    print(f"  → {comp['interpretacion']}")
    print()
    print("=" * 65)

    return {
        "pc1_raw_info": pc1_info,
        "comparacion": comp,
        "df_pc1_raw": df_pc1_raw,
        "df_merged": df_merged,
    }
