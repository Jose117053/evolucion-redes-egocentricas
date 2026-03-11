import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm

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
    Plot de proporciones para ver composición temporal.
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

def plot_dispersion_ols(df, x_col, y_col):
    """Grafica la dispersión con la línea de tendencia OLS interactiva."""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        hover_data=['year'],
        title=f"Impacto de {x_col} en {y_col}",
        trendline='ols',# dibuja la recta automáticamente
        trendline_color_override='red',
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.show()

def plot_matriz_features(df, features_list, target_col='sage_pc1'):
    """Genera una matriz 3x3 comparando features candidatas contra el target."""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    axes = axes.flatten()

    for i, feature in enumerate(features_list):
        if feature in df.columns:
            sns.regplot( # sns.regplot dibuja los puntos y a la linea automaticamente
                data=df, 
                x=feature, 
                y=target_col, 
                ax=axes[i],
                scatter_kws={'alpha': 0.5, 'color': 'royalblue'}, #Los puntos
                line_kws={'color': 'red', 'linewidth': 2} #La linea
            )
            axes[i].set_title(f"{feature} vs {target_col}")
            axes[i].set_ylabel(target_col)

    plt.tight_layout()
    plt.show()

def plot_panel_residuos(predicciones, residuos):
    """Muestra el panel triple: Residuos vs Fitted, Histograma y Q-Q Plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Linealidad y Homocedasticidad
    # queremos que los puntos deben verse como una nube aleatoria sin forma de embudo o curva.
    sns.scatterplot(x=predicciones, y=residuos, ax=axes[0], color='blue', alpha=0.7)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuos vs Predicciones (Homocedasticidad)')
    axes[0].set_xlabel('Valores Predichos')
    axes[0].set_ylabel('Residuos')

    # normalidad: Histograma de los residuos
    # deberia mostrar una campana de gauss centrada en 0
    sns.histplot(residuos, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Histograma de los Residuos (Normalidad)')
    axes[1].set_xlabel('Residuo')

    # Normalidad Exacta: Gráfico Q-Q (Quantile-Quantile)
    # los puntos azules deben seguir la línea roja en diagonal
    sm.qqplot(residuos, line='45', fit=True, ax=axes[2], color='purple')
    axes[2].set_title('Gráfico Q-Q (Normalidad)')

    plt.tight_layout()
    plt.show()

def plot_acf_residuos(df_all, residuos):
    """Grafica la función de autocorrelación (ACF) de los residuos."""
    df_res = pd.DataFrame({
        "year": df_all["year"].astype(int).to_numpy(),
        "resid": residuos
    }).sort_values("year")
    # ACF de residuos (para ver si queda autocorrelación)
    fig = plt.figure(figsize=(10, 3))
    sm.graphics.tsa.plot_acf(df_res["resid"], lags=10, ax=plt.gca())
    plt.title("ACF de residuos (hasta 10 lags)")
    plt.tight_layout()
    plt.show()

def plot_feature_stack_shares(df_cats, feature_name, title_prefix="Composición temporal"):
    """
    Limpia el nombre de la feature, busca sus columnas ponderadas (wdocs) 
    y grafica el área apilada.
    """
    safe_cit = feature_name.replace(" ", "_").replace(".", "").replace(",", "").replace("%", "pct")
    cols_cit_w = [c for c in df_cats.columns if c.startswith(f"wdocs_{safe_cit}_")]
    
    if len(cols_cit_w) > 0:
        plot_stack_shares(
            df_cats,
            year_col="year",
            cols=cols_cit_w,
            title=f"{title_prefix} (ponderado por Documents,) - {feature_name}"
        )
    else:
        print(f"No se encontraron columnas para la feature: {feature_name}")