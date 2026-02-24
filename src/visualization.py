import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper: normalización 0-1 (para comparar tendencias)
def minmax_01(series: pd.Series):
    x = series.astype(float).to_numpy()
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return pd.Series([np.nan]*len(series), index=series.index)
    xmin, xmax = np.nanmin(x[mask]), np.nanmax(x[mask])
    if np.isclose(xmin, xmax):
        return pd.Series([0.0]*len(series), index=series.index)
    y = (x - xmin) / (xmax - xmin)
    return pd.Series(y, index=series.index)

# Series temporales

def plot_timeseries_matplotlib(df, cols, title):
    plt.figure(figsize=(12, 6))
    plotted_any = False
    for c in cols:
        if c not in df.columns:
            print(f"[WARN] Columna no encontrada: {c}")
            continue
        plt.plot(df["year"], df[c], marker="o", label=c)
        plotted_any = True

    plt.title(title)
    plt.xlabel("Año")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_separated_metrics(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Arriba: Riqueza (q=0)
    if "hill_q0" in df.columns:
        ax1.plot(df["year"], df["hill_q0"], marker="o", label="Categorías presentes (q=0)")
    else:
        ax1.text(0.5, 0.5, "Falta hill_q0", transform=ax1.transAxes, ha="center")
    ax1.set_ylabel("Cantidad de categorías")
    ax1.set_title("Cobertura / riqueza (Hill q=0)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Abajo: Dominancia
    if "top1_share" in df.columns:
        ax2.plot(df["year"], df["top1_share"], marker="s", label="Dominancia (Top 1 Share)")
        ax2.set_ylim(0, 1.1)
    else:
        ax2.text(0.5, 0.5, "Falta top1_share", transform=ax2.transAxes, ha="center")
    ax2.set_ylabel("Proporción del líder (0-1)")
    ax2.set_title("Dominancia: ¿qué tan acaparado está el sistema?")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel("Año")

    plt.tight_layout()
    plt.show()

def plot_corr_matrix(corr, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_stack_shares(df, year_col, cols, title):
    """
    Plot de proporciones (stacked area) para ver composición temporal.
    cols: lista de columnas que suman aprox 1.
    """
    dfp = df[[year_col] + cols].copy().sort_values(year_col)
    plt.figure(figsize=(12, 6))
    plt.stackplot(dfp[year_col], [dfp[c].fillna(0.0) for c in cols], labels=cols)
    plt.title(title)
    plt.xlabel("Año")
    plt.ylabel("Proporción")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()