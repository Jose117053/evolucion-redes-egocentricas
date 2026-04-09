import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import src.global_config as global_cfg

def safe_probs(masses: np.ndarray):
    """
    Convierte masas >=0 en distribución p_i.
    Regresa None si no se puede (suma 0 o NaN).
    Prepara los datos para ser usados en las formulas de diversidad: shanon, simpson, etc.
    """
    #Las masaas son la cantidad de algo, por ejemplo el numero de documentos
    masses = np.asarray(masses, dtype=float)
    masses = np.where(np.isfinite(masses), masses, 0.0)
    masses = np.where(masses < 0, 0.0, masses)

    total = masses.sum()
    if total <= 0:
        return None
    return masses / total #Se devuelve la distribucion, EJ: [20,30,50] entonces regresa: [0.2,0.3,0.5]

#Algo importante a considerar es qued hill necesita una distribucion p que sume 1
#Es importante el cómo es que yo defino esto, en mi caso de momento es con "Documents,"
#Hill numbers requiere categorias. DE MOMENTO (porque aun no estoy seguro) uso a los clusters. 
#De lo contrario, ¿cuáles serían las categorias?, aparte, mis nodos unicamente cuentan con un solo valor categorico: numero de cluster.

#La distribucion es en base a una masa, features como ave references, o ave citations no convienen porque son promedios.
def hill_numbers(p: np.ndarray):
    """
    Calcula Hill numbers para q=0,1,2.
    p debe ser distribución (suma 1).
    Devuelve (D0, D1, D2).
    """
    if p is None:
        return (np.nan, np.nan, np.nan)

    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # evitar log(0), pues no está definido
    if p.size == 0:
        return (np.nan, np.nan, np.nan)

    # q=0: riqueza (número de categorías con masa >0)
    D0 = float(p.size)

    # q=1: exp(Shannon)
    H = -float(np.sum(p * np.log(p)))
    D1 = float(np.exp(H))

    # q=2: inverso de Simpson
    D2 = float(1.0 / np.sum(p ** 2))

    return (D0, D1, D2)

def group_masses_by_category(categories: np.ndarray, masses: np.ndarray):
    """
    Agrupa y suma las masas según su categoria (macro,  meso,  micro).
    Suma la aportacion individual de cada ego a la categoría a la que pertenece
    Regresa: cat,mass : cantidad de categorias únicas y la suma total de masa por cada una
    """
    categories = np.asarray(categories, dtype=int)
    masses = np.asarray(masses, dtype=float)

    # limpiar masas
    masses = np.where(np.isfinite(masses), masses, 0.0)
    masses = np.where(masses < 0, 0.0, masses)

    #Cats es unicamente las categorias presentes en la red egocentrica de ese año. No se toman en cuenta el resto de informacion de taxonomia
    #Por ejemplo puedo tener 200  mesotopicos pero  solo 30 alters, y solo 14 mesotopicos, unicamente se toman en cuenta los 14
    
    df = pd.DataFrame({"cat": categories, "mass": masses}) #Simplemente se hace dataframe para poder hacer operaciones sql
    df = df[df["cat"] >= 0] # excluir -1 (no mapeados), es un "por si acaso"
    grp = df.groupby("cat", as_index=False)["mass"].sum().sort_values("cat") #Agrupa todos los que pertenecen al mismo cluster, y suma las masas de los integrantes, al final los odena
    return grp["cat"].to_numpy(), grp["mass"].to_numpy()

def alters_feature_stats(X: np.ndarray):
    """
    Calcula centro y dispersión de una nube de puntos (alters x features).
    Regresa dict con métricas globales + arrays de mean/var por feature.
    """
    X = np.asarray(X, dtype=float) #Pasar a arreglo de numpy
    X = np.where(np.isfinite(X), X, np.nan) #Limpieza de datos: donde haya nan, o infinito se reemplaza por la media
    mu = np.nanmean(X, axis=0) #Media de cada columna (cada)
    var = np.nanvar(X, axis=0) # var por feature 

    # cov y trace (si N>=2)
    if X.shape[0] >= 2:
        # para cov: reemplazar NaN por mean de columna para no romper el calculo, es una estrategia clasicca de estadistica
        X_filled = np.where(np.isnan(X), mu, X)
        cov = np.cov(X_filled, rowvar=False)
        trace_cov = float(np.trace(cov))
    else:
        trace_cov = np.nan

    # distancia promedio al centro
    X_filled = np.where(np.isnan(X), mu, X)
    dists = np.linalg.norm(X_filled - mu, axis=1) #linalg es la distancia enlinea recta al alter con respecto a la  media, asis=1 hace que se haga por cada fila
    mean_dist = float(np.mean(dists))
    median_dist = float(np.median(dists))

    return {
        "mu": mu,
        "var": var,
        "trace_cov": trace_cov,
        "mean_dist_to_mu": mean_dist,
        "median_dist_to_mu": median_dist,
    }


def run_stage2(snapshots: dict, feature_names: list, documents_col_name="Documents,"):
    """
    Construye un DataFrame con métricas mesotópicas por año:
    - mean_* por feature
    - var_* por feature
    - trace_cov, mean_dist_to_mu, median_dist_to_mu
    - Hill numbers por categoría con masa Documents

    documents_col_name lo dejo como "Documents,"
    """
    # ubicar índice de la columna Documents,
    # feature_names debe corresponder a FEATURES en el mismo orden
    if documents_col_name not in feature_names:
        print(f"[AVISO] '{documents_col_name}' no está en feature_names. Hill por Documents saldrá NaN.")
        documents_idx = None
    else:
        documents_idx = feature_names.index(documents_col_name)

    rows = [] #aqui se guardaran diccionarios: una fila por año

    for year in sorted(snapshots.keys()):
        snap = snapshots[year]
        X = snap["X_alters"]
        cats = snap["categories_alters"]
       # strengths = snap["strengths_to_ego"]

        stats = alters_feature_stats(X)
        mu = stats["mu"]
        var = stats["var"]

        row = {
            "year": year,
            "ego_id": snap["ego_id"],
            "n_alters": int(X.shape[0]),
            "missing_in_csv": int(snap.get("missing_in_csv", 0)),
            "trace_cov": stats["trace_cov"],
            "mean_dist_to_mu": stats["mean_dist_to_mu"],
            "median_dist_to_mu": stats["median_dist_to_mu"],
        }

        # mean_* y var_* por feature
        for j, fname in enumerate(feature_names):
            row[f"mean_{fname}"] = float(mu[j]) if np.isfinite(mu[j]) else np.nan
            row[f"var_{fname}"] = float(var[j]) if np.isfinite(var[j]) else np.nan

        # Hill por categorías CSV (macro/meso/micro) usando masa Documents
        if documents_idx is not None:
            docs = X[:, documents_idx]
            _, mass_by_cat = group_masses_by_category(cats, docs)
            p = safe_probs(mass_by_cat)
            D0, D1, D2 = hill_numbers(p)
            row["hill_q0"] = D0
            row["hill_q1"] = D1
            row["hill_q2"] = D2
            row["top1_share"] = float(np.max(p)) if p is not None else np.nan
        else:
            row["hill_q0"] = row["hill_q1"] = row["hill_q2"] = np.nan
            row["top1_share"] = np.nan

        rows.append(row)

    df_stage2 = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return df_stage2

def _clean_1d(x: np.ndarray):
    """Limpia NaN/Inf y deja solo finitos."""
    x = np.asarray(x, dtype=float).reshape(-1) #hacce la matriz sea plana, una sola fila
    x = x[np.isfinite(x)]
    return x

def global_quantile_edges(values_1d: np.ndarray, probs: list):
    """
    Calcula umbrales globales por cuantiles.
    Regresa array de edges
    Maneja casos donde hay muchos valores iguales Ej: [0,0,1] -> [0,1].


    """
    v = _clean_1d(values_1d)
    if v.size == 0:
        return None

    edges = np.quantile(v, probs)

    # Esto es para eliminar secciones repetidas,  por ejemplo si el 80% de  los papers tiene
    #0 citas  entonces la  lista de cortes sería [0,0,5], por lo que lo reducimos a [0,5]
    edges_unique = np.unique(edges)
    if edges_unique.size < edges.size:
        edges = edges_unique

    # Si al final no hay cortes (todo igual), regresamos None
    if edges.size == 0:
        return None

    return edges

def assign_bins(x: np.ndarray, edges: np.ndarray):
    """
    Asigna bins usando edges tipo cuantiles.
    Retorna bins en {0..K} donde K = len(edges) (si edges = [e1,e2,e3] => 4 bins).
    Regla:
      bin 0: x <= e1
      bin 1: e1 < x <= e2
      ...
      bin K: x > eK
    Si edges es None -> todo -1

    Regresa una lista del mismo tamaño del recibido, en lugar de valores,
    serán etiquetas ( a la cual pertenecen: Q1,Q2, Q3 o Q4)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    bins = np.full(x.shape, -1, dtype=int)

    if edges is None:
        return bins

    # Para NaN/Inf dejamos -1
    mask = np.isfinite(x)
    xf = x[mask]

    # np.searchsorted con side='right' implementa (<= edge) correctamente
    # regresa [0..len(edges)]
    b = np.searchsorted(edges, xf, side="left")
    bins[mask] = b
    return bins

'''

Unweighted: cada nodo aporta un valor de uno sin importar su perfil bibliometrico
Weighted: ponderado en base a la cantidad de documentos.

Por ejemplo para unweighted si tengo 5 papers en Q1 y 5 en Q4, Q1 es el 50%
para weighted si tengo 5 papers en Q1 pero aportan mas, entonces Q1 es el 90%
'''

def shares_from_bins(bins: np.ndarray, weights: np.ndarray, n_bins: int):
    """
    Calcula proporciones por bin (0..n_bins-1).
    bins: array de enteros (puede traer -1 para missing)
    weights: pesos por alter (si None: usa 1s)
    """
    bins = np.asarray(bins, dtype=int)
    valid = bins >= 0
    bins_v = bins[valid]

    if weights is None:
        w = np.ones(bins_v.shape[0], dtype=float) #Al no ser ponderado, todos valen uno
    else:
        w_all = np.asarray(weights, dtype=float).reshape(-1)
        w_all = np.where(np.isfinite(w_all), w_all, 0.0)
        w_all = np.where(w_all < 0, 0.0, w_all)
        w = w_all[valid]

    total = w.sum()
    if total <= 0:
        return np.array([np.nan]*n_bins, dtype=float)

    out = np.zeros(n_bins, dtype=float)
    for k in range(n_bins):
        out[k] = w[bins_v == k].sum() / total
    return out


def find_dist_cols(df: pd.DataFrame, prefix: str, feature: str, cats: list):
    """
    Busca columnas {prefix}_{feature}_{cat} y devuelve dict cat->col solo si existen.

    {
    "baja": "unw_Ave_citations_baja",
    "media": "unw_Ave_citations_media",
    ...
    }

    """
    out = {}
    for cat in cats:
        col = f"{prefix}_{feature}_{cat}"
        if col in df.columns:
            out[cat] = col
    return out

def dominant_label_from_cols(df: pd.DataFrame, cols_dict: dict):
    """
    cols_dict: {cat: colname}
    Devuelve Serie con etiqueta dominante por fila.
    Dada la distribución por categorías (proporciones), elegir la categoría dominante por fila (por año)
    """
    if len(cols_dict) < 2:
        return pd.Series([np.nan] * len(df), index=df.index)

    mat = df[list(cols_dict.values())].to_numpy(dtype=float)
    mat = np.where(np.isfinite(mat), mat, -np.inf)

    winners = np.argmax(mat, axis=1)
    #print(mat)
    cats_list = list(cols_dict.keys())

    # si toda la fila es -inf (todo NaN), devolver NaN
    out = []
    values = []
    for r, k in enumerate(winners):
        if np.isfinite(mat[r, k]):
            out.append(cats_list[k])
            values.append(float(mat[r,k]))
        else:
            out.append(np.nan)
            values.append(np.nan)
    
    
    return pd.Series(out, index=df.index), pd.Series(values, index=df.index)

def add_dominants(df: pd.DataFrame, feature_specs: dict):
    df_out = df.copy()

    for feature, spec in feature_specs.items():
        cats = spec["cats"]

        # unw
        cols_unw = find_dist_cols(df_out, "unw", feature, cats)
        df_out[f"unw_{feature}"], df_out[f"unw_{feature}_value"] = dominant_label_from_cols(df_out, cols_unw)

        # wdocs
        cols_w = find_dist_cols(df_out, "wdocs", feature, cats)
        df_out[f"wdocs_{feature}"], df_out[f"wdocs_{feature}_value"] = dominant_label_from_cols(df_out, cols_w)
       # print(dominant_label_from_cols(df_out, cols_unw))

    return df_out

def compute_thresholds(snapshots, feature_names, cat_specs, mode="global", window_size=5):
    """
    Regresa: thresholds_by_year[year][feature] = edges
    - mode="global": un solo set de edges para todos los años
    - mode="window": edges calculados por ventanas de años (bloques)
    """
    years = sorted(snapshots.keys())
    thresholds_by_year = {y: {} for y in years}

    def collect_vals(year_list, feat):
        j = feature_names.index(feat)
        all_vals = []
        for y in year_list:
            X = snapshots[y]["X_alters"]
            if X.size == 0:
                continue
            all_vals.append(X[:, j])
        if not all_vals:
            return None
        return np.concatenate(all_vals, axis=0)

    if mode == "global":
        for feat, spec in cat_specs.items():
            if feat not in feature_names:
                for y in years:
                    thresholds_by_year[y][feat] = None
                continue

            vals = collect_vals(years, feat)
            edges = global_quantile_edges(vals, spec["probs"]) if vals is not None else None
            for y in years:
                thresholds_by_year[y][feat] = edges

            if edges is None:
                print(f"[INFO] {feat}: edges=None (distribución degenerada o vacía).")
            else:
                print(f"[OK] {feat}: edges globales = {edges}")


        return thresholds_by_year

    if mode == "window":
        # Particiona años en bloques consecutivos de tamaño window_size
        for i in range(0, len(years), window_size):
            window_years = years[i:i+window_size]

            for feat, spec in cat_specs.items():
                if feat not in feature_names:
                    for y in window_years:
                        thresholds_by_year[y][feat] = None
                    continue

                vals = collect_vals(window_years, feat)
                edges = global_quantile_edges(vals, spec["probs"]) if vals is not None else None

                for y in window_years:
                    thresholds_by_year[y][feat] = edges

                if edges is None:
                    print(f"[INFO] {feat}: edges=None (distribución degenerada o vacía).")
                else:
                    print(f"[OK] {feat}: edges globales = {edges}")

        return thresholds_by_year

    raise ValueError("mode debe ser 'global' o 'window'")


'''
generate_anual_proportions:

            En lugar de [1 cita, 50 citas, 2 citas], ahora tenemos [0, 3, 0] (Bajo, Alto, Bajo)
            Se asigna el bin en base a estos valores.

            Ejemplo completo:

            si los intervalos globales son [2.0, 10.0, 20.0] las etiquetas son [bajo, medio, medio_alto, alto]
            Tenemos  3 alters: Causal Inference con 1 documento, Rock Mechanics con 1 documento y 
            Reservoir Operation con  8 documentos

            Assign_bins se encarga de asignar bins:
            Causal Inference ->  menor que 2.0 -> Bin 0
            Rock mechanics ->  menor que 2.0 -> Bin 0
            Reservoir Operation ->  mayor que 2.0 y menor que 10.0 -> Bin 1
            Bins -> [0,0,1]

            Share_from_bins Caso unweighted
            Total de alters: 3
            Bin 0: 2 alters 2/3 ->0.66 (66%)
            Bin 1: 1 alter 1/3 ->0.33 (33%)
            Bin 2: 0 alters -> 0.0
            Bin 3: 0 alters -> 0.0

            Share_from_bins Caso weighted
            Total de documents: 10
            Bin 0: 1+1=2 -> 2/10 -> 0.20(20%)
            Bin 1: 8 -> 8/10 -> 0.80(80%)
            Bin 2: 0 -> 0.0
            Bin 3: 0 -> 0.0

            Ultimo paso, guardar:
            unw_citations_bajo:0.66
            w_citations_bajo:0.20
            unw_citations_medio:0.33
            w_citations_medio:0.80
            el resto de variaciones para citations es 0.0

            Al final todos los weighted suman 1 y todos los unweighted suman 1
'''
def generate_anual_proportions(snapshots, feature_names, cat_specs, thresholds_dict, idx_docs):
    """
    Asigna bins y calcula proporciones (weighted y unweighted) por año.
    Devuelve un DataFrame (df_cats) detallado.
    Funciona tanto con umbrales globales como con umbrales por ventana de tiempo.
    """
    rows_cat = []
    
    # Detectamos automáticamente si el diccionario es por año o es global
    # Si la primera llave es un número (un año), sabemos que es modo ventana.
    primer_llave = list(thresholds_dict.keys())[0]
    is_window_mode = isinstance(primer_llave, int)

    for year in sorted(snapshots.keys()):
        snap = snapshots[year]
        X = snap["X_alters"]

        if X.size == 0:
            continue

        w_docs = None
        if idx_docs is not None:
            w_docs = X[:, idx_docs]

        row = {"year": year}

        for feat, spec in cat_specs.items():
            if feat not in feature_names:
                continue

            j = feature_names.index(feat)
            
            if is_window_mode:
                edges = thresholds_dict.get(year, {}).get(feat) # Si estamos en modo ventana, buscamos el año específico
            else:
                edges = thresholds_dict.get(feat) # Si estamos en modo global, extraemos el feature directo

            bins = assign_bins(X[:, j], edges)
            # si edges colapsó (por ejemplo 1 edge en vez de 3), el número de bins reales cambia
            # Definimos n_bins_real = len(edges)+1 si edges existe, sino spec["n_bins"]
            if edges is None:
                n_bins_real = spec["n_bins"]
                labels = spec["labels"]
            else:
                n_bins_real = len(edges) + 1 # si colapsó y hay menos bins que labels, ajustamos labels automáticamente
                labels = spec["labels"][:n_bins_real] 

            s_unw = shares_from_bins(bins, weights=None, n_bins=n_bins_real)
            s_w = shares_from_bins(bins, weights=w_docs, n_bins=n_bins_real)

            safe_feat = feat.replace(" ", "_").replace(".", "").replace(",", "").replace("%", "pct")

            # Guardar columnas con nombres estables
            # share_unw_citations_baja, share_w_citations_alta, etc.
            for k, lab in enumerate(labels):
                row[f"unw_{safe_feat}_{lab}"] = float(s_unw[k]) if np.isfinite(s_unw[k]) else np.nan
                row[f"wdocs_{safe_feat}_{lab}"] = float(s_w[k]) if np.isfinite(s_w[k]) else np.nan

        rows_cat.append(row)

    df_cats = pd.DataFrame(rows_cat).sort_values("year").reset_index(drop=True)
    return df_cats

def cargar_y_unir_datos(level, output_dir="./Output"):
    """Carga las métricas y los embeddings, y los une por año."""
    df_stage2 = pd.read_csv(os.path.join(output_dir, f"metricas_{level}.csv"))
    df_sage = pd.read_csv(os.path.join(output_dir, "graphsage_pca1.csv"))
    
    df_all = pd.merge(df_stage2, df_sage, on="year", how="inner").sort_values("year").reset_index(drop=True) #aqui unimos los 2 data frames
    return df_all

def ajustar_modelo_ols(df, y_col, x_cols, usar_hac=False, maxlags=3):
    """
    Ajusta un modelo de regresión lineal OLS.
    Si usar_hac es True, aplica errores robustos (HAC).
    """
    y = df[y_col].astype(float)
    X = df[x_cols].astype(float)
    X_sm = sm.add_constant(X)
    
    if usar_hac:
        modelo = sm.OLS(y, X_sm).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    else:
        modelo = sm.OLS(y, X_sm).fit()
        
    return modelo

def cargar_y_unir_datos(level, output_dir="./Output"):
    """Carga las métricas y los embeddings, y los une por año."""
    df_stage2 = pd.read_csv(os.path.join(output_dir, f"metricas_{level}.csv"))
    df_sage = pd.read_csv(os.path.join(output_dir, "graphsage_pca1.csv"))
    
    df_all = pd.merge(df_stage2, df_sage, on="year", how="inner").sort_values("year").reset_index(drop=True)
    return df_all

def ajustar_modelo_ols(df, y_col, x_cols, usar_hac=False, maxlags=3):
    """
    Ajusta un modelo de regresión lineal OLS.
    Si usar_hac es True, aplica errores robustos (HAC).
    """
    y = df[y_col].astype(float)
    X = df[x_cols].astype(float)
    X_sm = sm.add_constant(X)
    
    if usar_hac:
        modelo = sm.OLS(y, X_sm).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    else:
        modelo = sm.OLS(y, X_sm).fit()
        
    return modelo

def ejecutar_diagnostico_residuos(modelo):
    """Calcula e imprime las pruebas de Durbin-Watson, Shapiro-Wilk y Breusch-Pagan."""
    residuos = modelo.resid
    
    print("-" * 60)
    print("Diagnostico de residuos")
    print("-" * 60)
    
    # INDEPENDENCIA (Durbin-Watson)
    # Rango de 0 a 4. El valor ideal es 2 (independencia
    # Menos de 1.5 es mala señal (autocorrelación positiva)

    dw_valor = durbin_watson(residuos)
    print(f"Prueba de Durbin-Watson (Independencia): {dw_valor:.3f}")

    # NORMALIDAD (Shapiro-Wilk)
    # H0: Los residuos son normales.
    shapiro_test, shapiro_p = stats.shapiro(residuos)
    print(f"Prueba de Shapiro-Wilk (Normalidad): p-value = {shapiro_p:.4f}")

    # HOMOCEDASTICIDAD (Breusch-Pagan)
    # H0: La varianza es constante.
    bp_test = het_breuschpagan(residuos, modelo.model.exog)
    print(f"Prueba de Breusch-Pagan (Homocedasticidad): p-value = {bp_test[1]:.4f}")
    
    return residuos, modelo.fittedvalues


def procesar_etiquetas_y_valores(df_cats, cat_specs):
    """
    Limpia los nombres de las features, calcula los dominantes y 
    separa el dataframe en uno de etiquetas y otro de valores numéricos.
    """
    df = df_cats.copy()
    
    #Lo unico que hace es extraer los "labels" definidos en la constante CAT_SPECS de config.py
    specs_limpios = {}
    for llave_sucia, valores in cat_specs.items():
        llave_limpia = llave_sucia.replace(".", "").replace(" ", "_")
        specs_limpios[llave_limpia] = {"cats": valores["labels"]}

    # aregar columnas dominantes
    df2 = add_dominants(df, specs_limpios)

    # Etiquetas en forma de palabras
    label_cols = ["year"]
    for feature in specs_limpios.keys():
        label_cols += [f"unw_{feature}", f"wdocs_{feature}"]
    df_labels = df2[label_cols].sort_values("year").reset_index(drop=True)

    #Etiquetas en forma de numeros
    value_cols = ["year"]
    for feature in specs_limpios.keys():
        value_cols += [f"unw_{feature}_value", f"wdocs_{feature}_value"]
    df_values = df2[value_cols].sort_values("year").reset_index(drop=True)

    return df_labels, df_values


# ─────────────────────────────────────────────────────────────────────
# MODULARIDAD ONTOLÓGICA (antes en modularity.py)
# Reutiliza hill_numbers, safe_probs y group_masses_by_category
# ─────────────────────────────────────────────────────────────────────

def resumen_modularidad_temporal(snapshots: dict,
                                  documents_col_name: str = "Documents,"):
    """
    Calcula métricas de diversidad ontológica por año usando Hill Numbers.
    Reutiliza las funciones existentes hill_numbers, safe_probs y
    group_masses_by_category.

    Parámetros
    ----------
    snapshots : dict[year] -> parsed snapshot (de build_yearly_index)
        Cada snapshot tiene 'X_alters' y 'categories_alters'.
    documents_col_name : str
        Nombre de la columna de masa (Documents,) en FEATURES.

    Retorna
    -------
    pd.DataFrame con columnas:
        [year, n_alters, n_clusters, D0, D1, D2, entropia, hhi,
         cluster_dom_id, cluster_dom_prop]
    """
    feature_names = [key for _, key in global_cfg.FEATURES]
    doc_idx = feature_names.index(documents_col_name)

    registros = []

    for year in sorted(snapshots.keys()):
        snap = snapshots[year]
        X = snap["X_alters"]
        cats = snap["categories_alters"]

        n_alters = X.shape[0]
        masses = X[:, doc_idx]

        # Agrupar masas por categoría (reutiliza función existente)
        cat_ids, cat_masses = group_masses_by_category(cats, masses)

        # Hill numbers (reutiliza función existente)
        p = safe_probs(cat_masses)
        D0, D1, D2 = hill_numbers(p)

        # Entropía de Shannon y HHI derivados de Hill
        if p is not None:
            p_arr = np.asarray(p, dtype=float)
            p_arr = p_arr[p_arr > 0]
            entropia = float(-np.sum(p_arr * np.log2(p_arr)))
            hhi = float(np.sum(p_arr ** 2))
        else:
            entropia = 0.0
            hhi = 1.0

        # Cluster dominante
        if len(cat_ids) > 0:
            max_idx = np.argmax(cat_masses)
            dom_id = int(cat_ids[max_idx])
            dom_prop = float(cat_masses[max_idx] / cat_masses.sum())
        else:
            dom_id = -1
            dom_prop = 0.0

        registros.append({
            'year': year,
            'n_alters': n_alters,
            'n_clusters': int(D0) if not np.isnan(D0) else 0,
            'D0': round(D0, 4) if not np.isnan(D0) else 0,
            'D1': round(D1, 4) if not np.isnan(D1) else 0,
            'D2': round(D2, 4) if not np.isnan(D2) else 0,
            'entropia': round(entropia, 4),
            'hhi': round(hhi, 4),
            'cluster_dom_id': dom_id,
            'cluster_dom_prop': round(dom_prop, 4)
        })

    return pd.DataFrame(registros)


def perfil_por_cluster(snapshot: dict, feature_names: list = None):
    """
    Calcula el perfil bibliométrico promedio agrupado por cluster ontológico
    para un snapshot dado.

    Parámetros
    ----------
    snapshot : dict
        Un snapshot de build_yearly_index (tiene X_alters, categories_alters).
    feature_names : list
        Nombres de las features. Si None, se toman de global_config.

    Retorna
    -------
    pd.DataFrame con columnas [cluster, n_alters, feat1_mean, feat1_std, ...]
    """
    if feature_names is None:
        feature_names = [key for _, key in global_cfg.FEATURES]

    X = snapshot["X_alters"]
    cats = snapshot["categories_alters"]

    # Agrupar por categoría
    unique_cats = sorted(set(cats[cats >= 0]))
    registros = []

    for cat in unique_cats:
        mask = cats == cat
        X_cat = X[mask]
        registro = {'cluster': int(cat), 'n_alters': int(mask.sum())}

        for j, feat in enumerate(feature_names):
            vals = X_cat[:, j]
            registro[f'{feat}_mean'] = float(np.nanmean(vals))
            registro[f'{feat}_std'] = float(np.nanstd(vals))

        registros.append(registro)

    return pd.DataFrame(registros)


def enriquecer_con_labels(df_perfil: pd.DataFrame,
                           taxonomy_map: dict,
                           nivel: str = "macro") -> pd.DataFrame:
    """
    Añade etiquetas legibles al DataFrame de perfiles por cluster
    usando el taxonomy_map de load_taxonomy_mapping().

    Parámetros
    ----------
    df_perfil : pd.DataFrame
        DataFrame con columna 'cluster' (IDs numéricos).
    taxonomy_map : dict
        Mapping de load_taxonomy_mapping().
    nivel : str
        'macro', 'meso' o 'micro'.
    """
    # Construir mapping de id -> label desde el taxonomy_map
    label_map = {}
    for code, info in taxonomy_map.items():
        cat_id = str(info[f"{nivel}_id"])
        if cat_id not in label_map:
            label_map[cat_id] = info[f"{nivel}_label"]

    df_perfil = df_perfil.copy()
    df_perfil['cluster_label'] = df_perfil['cluster'].astype(str).map(label_map)

    return df_perfil


def correlacionar_modularidad(df_modularidad: pd.DataFrame,
                               pc1_series: np.ndarray,
                               pc1_years: np.ndarray,
                               pc1_label: str = "PC1") -> pd.DataFrame:
    """
    Correlaciona las métricas de modularidad contra una serie de PC1.

    Parámetros
    ----------
    df_modularidad : DataFrame de resumen_modularidad_temporal()
    pc1_series : array con valores de PC1 (uno por año)
    pc1_years : array con los años correspondientes
    pc1_label : etiqueta (ej. "PC1_raw", "PC1_graphsage")

    Retorna
    -------
    DataFrame con correlaciones de Pearson y p-values.
    Imprime un reporte diagnóstico.
    """
    from scipy.stats import pearsonr

    df_mod = df_modularidad.copy()
    df_pc1 = pd.DataFrame({'year': pc1_years, pc1_label: pc1_series})
    df_merged = pd.merge(df_mod, df_pc1, on='year', how='inner')

    metricas = ['entropia', 'hhi', 'n_clusters', 'cluster_dom_prop', 'n_alters',
                'D0', 'D1', 'D2']
    resultados = []

    print("=" * 65)
    print(f"  CORRELACIÓN: Modularidad vs {pc1_label}")
    print("=" * 65)
    print(f"  Años alineados: {len(df_merged)}")
    print("-" * 65)

    for metrica in metricas:
        if metrica not in df_merged.columns:
            continue

        r, p = pearsonr(df_merged[metrica], df_merged[pc1_label])
        resultados.append({
            'metrica': metrica,
            'pearson_r': round(r, 4),
            'p_value': round(p, 6),
            'r_squared': round(r**2, 4),
            'significativa': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        })
        print(f"  {metrica:20s}:  ρ = {r:+.4f}  (p = {p:.4f})  R² = {r**2:.4f}  {resultados[-1]['significativa']}")

    print("-" * 65)

    # Diagnóstico principal: entropía vs PC1
    r_entropia = [r for r in resultados if r['metrica'] == 'entropia']
    if r_entropia:
        rho = abs(r_entropia[0]['pearson_r'])
        if rho > 0.70:
            print(f"  → 🔴 ALTA CORRELACIÓN (|ρ| = {rho:.2f}): {pc1_label} ya captura")
            print(f"    la diversidad disciplinaria. El Eje 2 probablemente sea redundante.")
        elif rho > 0.30:
            print(f"  → 🟡 CORRELACIÓN MODERADA (|ρ| = {rho:.2f}): Comparten información")
            print(f"    parcial. La ontología aporta info complementaria → LUZ VERDE para Eje 2.")
        else:
            print(f"  → 🟢 BAJA CORRELACIÓN (|ρ| = {rho:.2f}): Son dimensiones independientes.")
            print(f"    La ontología aporta info totalmente nueva → LUZ VERDE FUERTE para Eje 2.")

    print("=" * 65)

    return pd.DataFrame(resultados)