"""
Visualización para los experimentos baseline.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_comparacion_pc1(df_merged, sage_col="y_original"):
    """
    Grafica PC1_raw vs PC1_graphsage en dos subplots:
      1. Series temporales superpuestas (normalizadas 0-1)
      2. Scatter plot con línea de identidad

    Parámetros
    ----------
    df_merged : pd.DataFrame
        DataFrame con ['year', 'PC1_raw', sage_col].
    sage_col : str
        Nombre de la columna de PC1 de GraphSAGE.
    """

    years = df_merged["year"].values
    pc1_raw = df_merged["PC1_raw"].values
    pc1_sage = df_merged[sage_col].values

    rho, _ = pearsonr(pc1_raw, pc1_sage)

    # Normalizar a 0-1 para comparar visualmente
    def norm01(x):
        mn, mx = x.min(), x.max()
        if mx - mn == 0:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    pc1_raw_n = norm01(pc1_raw)
    pc1_sage_n = norm01(pc1_sage)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Series temporales
    ax = axes[0]
    ax.plot(years, pc1_raw_n, "o-", label="PC1_raw (PCA promedios)", color="#2196F3", linewidth=2)
    ax.plot(years, pc1_sage_n, "s--", label="PC1_graphsage", color="#FF5722", linewidth=2)
    ax.set_xlabel("Año", fontsize=12)
    ax.set_ylabel("PC1 normalizado (0–1)", fontsize=12)
    ax.set_title("E1: Series temporales de PC1", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Scatter
    ax = axes[1]
    ax.scatter(pc1_raw_n, pc1_sage_n, c=years, cmap="viridis", s=50, edgecolors="k", zorder=3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Identidad")
    ax.set_xlabel("PC1_raw (norm.)", fontsize=12)
    ax.set_ylabel("PC1_graphsage (norm.)", fontsize=12)
    ax.set_title(f"E1: Scatter  (ρ = {rho:.3f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(axes[1].collections[0], ax=ax, label="Año")

    plt.tight_layout()
    plt.show()

    return fig
