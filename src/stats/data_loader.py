import os
import json
import numpy as np
import pandas as pd
import src.global_config as global_cfg
from src.ontology import is_ego_node, load_taxonomy_mapping

'''
Calcula toda la información del ego-network en base al nivel taxonomico recibido: el id del ego
y una matriz de dimensión N x 8, con N el numero de alters de la red, y el 8
porque hay 8 features

Dependiendo el nivel se busca la informacion de los alters, por  ejemplo para el nivel meso
se guardará el número asociado a ese mesotopico: en 4.61.1460, se  extrae el 61.
 '''
def parse_microradial(raw_json, taxonomy_map: dict, level: str):

    if level not in ("macro", "meso", "micro"):
        raise ValueError("level debe ser: 'macro', 'meso' o 'micro'")
    items = raw_json["network"]["items"]
    links = raw_json["network"]["links"]

    # Encontrar ego, se añade una pequeña excepcion (Gracias a esto se detectó una anomalia en el año 1990 de los datos de SOM)
    ego_indices = [i for i, n in enumerate(items) if is_ego_node(n)]
    if len(ego_indices) == 0:
        raise ValueError(f"No se encontró ego en {level}.")

    def ego_score(i):
        n = items[i]
        nid = str(n.get("id", ""))
        is_mct_dot = 1 if nid.startswith("MCT.") else 0
        pdoc = float(n.get("scores", {}).get("Percent of documents", 0.0))
        docs = float(n.get("weights", {}).get("Documents,", 0.0))
        return (is_mct_dot, pdoc, docs)

    ego_idx = max(ego_indices, key=ego_score)
    ego_id = items[ego_idx]["id"]

    # máscara de alters, lo que se hace es crear un array de puros 1's (True) donde el unico 0 (False) es el ego
    alter_mask = np.ones(len(items), dtype=bool)
    alter_mask[ego_idx] = False

    # matriz X de alters (N x 8) N es el numero de alters de la red, 8 features
    X = []
    categories = []
    missing_in_csv = 0
   
    for i, node in enumerate(items):
        if not alter_mask[i]:
            continue
        row = []
        #Scope es la primera identificacion para acceder a los features. Ej: weights o scores
        #Key ya es como tal la propiedad a extraer. Ej: Percent of documents, aver autorship, etc.
        for scope, key in global_cfg.FEATURES:
            row.append(node.get(scope, {}).get(key, 0))
        X.append(row)

        code = str(node.get("id")) # ej "4.61.1460"
        info = taxonomy_map.get(code)

        if info is None: #-1 por si no se encontró
            categories.append(-1) 
            missing_in_csv += 1
        else:
            categories.append(int(info[f"{level}_id"]))

    #Simplemente los pasamos a arreglos de numpy para poderlos manipular facilmente
    X = np.asarray(X, dtype=float)
    categories = np.asarray(categories, dtype=int)

    return {
        "ego_id": ego_id,
        "X_alters": X,
        "categories_alters": categories,
        "n_items_total": len(items),
        "n_links_total": len(links),
        "missing_in_csv": missing_in_csv
    }

def validate_snapshot(parsed, year):
    warnings = {}

    X = parsed["X_alters"]
    if X.ndim != 2:
        raise ValueError(f"[{year}] X_alters no es 2D (shape={X.shape}).")
    if X.shape[1] != len(global_cfg.FEATURES):
        raise ValueError(f"[{year}] X_alters tiene {X.shape[1]} features, esperaba {len(global_cfg.FEATURES)}.")
    if X.shape[0] == 0:
        raise ValueError(f"[{year}] No hay alters (X_alters vacío).")

    #isnan y isinf crean una mascara del tamaño de la matriz X para detectar nans o infs
    #Si hay alguno se reporta
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    if nan_count > 0 or inf_count > 0:
        warnings["nan_inf"] = {"nan": nan_count, "inf": inf_count}

    if parsed.get("missing_in_csv", 0) > 0:
        warnings["missing_in_csv"] = parsed["missing_in_csv"]

    return warnings

def build_yearly_index(data_folder, start_year, end_year,  taxonomy_map, level):
    """
    Construye los snapshots de cada año (diccionario) con la información definida, esto
    se  hace dependiendo el nivel con el que se trabaja.

    Para cada año:
    - Carga el JSON de la red.
    - Lo procesa a nivel taxonómico (macro/meso/micro) usando parse_microradial.
    - Genera métricas resumen (n_alters, n_categories, missing, etc).
    - Captura errores y warnings

    Devuelve:
      - snapshots: dict[year] ->  snapshot procesado
      - df_index: DataFrame con resumen anual
      - df_issues: DataFrame con errores/warnings por año
    """
    print("Nivel taxonómico: ", level)

    snapshots = {}
    index_rows = []
    issue_rows = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_folder, f"network_{year}.json")

        if not os.path.exists(file_path):
            issue_rows.append({
                "year": year,
                "type": "missing_file",
                "detail": f"Archivo no encontrado: {file_path}"
            })
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            parsed = parse_microradial(raw_json, taxonomy_map=taxonomy_map, level=level)
            warnings = validate_snapshot(parsed, year)


            snapshots[year] = parsed

            # Resumen anual
            X = parsed["X_alters"]
            cats = parsed["categories_alters"]

            # quitar -1 (no mapeados) para contar categorías válidas
            cats_valid = cats[cats >= 0]
            index_rows.append({
                "year": year,
                "ego_id": parsed["ego_id"],
                "n_items_total": parsed["n_items_total"],
                "n_links_total": parsed["n_links_total"],
                "n_alters": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_categories_unique": int(len(set(cats_valid.tolist()))),
                "missing_in_csv": int(parsed.get("missing_in_csv", 0)),
                "level_used": level
            })

            # Guardar warnings si existen
            if warnings:
                issue_rows.append({
                    "year": year,
                    "type": "warning",
                    "detail": json.dumps(warnings, ensure_ascii=False)
                })

            
            print(f"[OK] {year} | alters={X.shape[0]} | cats_unique={len(set(cats_valid.tolist()))} | missing_csv={parsed.get('missing_in_csv',0)}")

        except Exception as e:
            issue_rows.append({
                "year": year,
                "type": "error",
                "detail": str(e)
            })
            print(f"[ERROR] {year}: {e}")

    df_index = pd.DataFrame(index_rows).sort_values("year").reset_index(drop=True)
    df_issues = pd.DataFrame(issue_rows, columns=["year", "type", "detail"])
    if not df_issues.empty:
        df_issues = df_issues.sort_values(["year", "type"]).reset_index(drop=True)

    return snapshots, df_index, df_issues

