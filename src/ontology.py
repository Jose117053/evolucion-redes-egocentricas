"""
src/ontology.py
===============
Módulo centralizado para funciones compartidas de ontología y detección
de nodos ego. Evita duplicación entre los submódulos gnn/ y stats/.

Funciones principales:
  - is_ego_node: Detecta si un nodo es el Ego de la red egocéntrica.
  - load_taxonomy_mapping: Carga el CSV de taxonomía y construye un
    mapping completo {micro_code → {macro_id, meso_id, micro_id, ...}}.
  - load_ontology_index: Construye un índice {micro_code → índice one-hot}
    para generar vectores one-hot por nivel ontológico.
"""

import pandas as pd
import src.global_config as gcfg


# ─────────────────────────────────────────────────────────────────────
# DETECCIÓN DEL NODO EGO
# ─────────────────────────────────────────────────────────────────────

def is_ego_node(node):
    """
    Determina si un nodo es el Ego de la red egocéntrica.
    Criterio: su id empieza con 'MCT' O su Percent of documents es 100.0.

    Utilizado por:
      - src/gnn/data_loader.py  (get_ego_mask, fit_global_scaler)
      - src/gnn/analysis.py     (cálculos post-embedding)
      - src/stats/data_loader.py (parse_microradial)
    """
    nid = str(node.get("id", ""))
    return (
        nid.startswith("MCT") or
        node.get("scores", {}).get("Percent of documents", 0) == 100.0
    )


# ─────────────────────────────────────────────────────────────────────
# CARGA DE TAXONOMÍA (mapping completo)
# ─────────────────────────────────────────────────────────────────────

_taxonomy_cache = {}  # Cache: csv_path → mapping dict

def load_taxonomy_mapping(csv_path: str = None) -> dict:
    """
    Carga el CSV de taxonomía y retorna un diccionario:
      {micro_code → {macro_id, meso_id, micro_id, macro_label, meso_label, micro_label}}

    El micro_code se extrae de la primera parte antes del espacio de
    la columna micro_label (ej: '4.61.1460' de '4.61.1460 Bayesian Networks').

    Si csv_path es None, usa gcfg.TAXONOMY_CSV.
    """
    if csv_path is None:
        csv_path = gcfg.TAXONOMY_CSV

    if csv_path in _taxonomy_cache:
        return _taxonomy_cache[csv_path]

    df = pd.read_csv(csv_path)
    micro_code = df["micro_label"].astype(str).str.split(" ").str[0].str.strip()

    mapping = {}
    for i, code in enumerate(micro_code):
        mapping[code] = {
            "macro_id": int(df.loc[i, "macro_id"]),
            "meso_id":  int(df.loc[i, "meso_id"]),
            "micro_id": int(df.loc[i, "micro_id"]),
            "macro_label": str(df.loc[i, "macro_label"]),
            "meso_label":  str(df.loc[i, "meso_label"]),
            "micro_label": str(df.loc[i, "micro_label"]),
        }

    _taxonomy_cache[csv_path] = mapping
    return mapping


# ─────────────────────────────────────────────────────────────────────
# ÍNDICE ONE-HOT PARA GNN (por nivel ontológico)
# ─────────────────────────────────────────────────────────────────────

_ontology_index_cache = {}  # Cache: (csv_path, level) → (code_to_idx, num_categories)

def load_ontology_index(csv_path: str = None, level: str = "macro"):
    """
    Carga el CSV de taxonomía y construye:
      - code_to_idx: dict {micro_code → índice one-hot} según el nivel
      - num_categories: número total de categorías en ese nivel

    El micro_code es la clave (ej: '4.61.1335') que coincide con los IDs
    de los nodos en el JSON de la red.

    Niveles:
      'macro' → ~10 categorías (ej: macro_id=4)
      'meso'  → ~278 categorías (ej: meso_id=61)
      'micro' → ~1933 categorías (ej: micro_id=1335)

    Si csv_path es None, usa gcfg.TAXONOMY_CSV.
    """
    if csv_path is None:
        csv_path = gcfg.TAXONOMY_CSV

    cache_key = (csv_path, level)
    if cache_key in _ontology_index_cache:
        return _ontology_index_cache[cache_key]

    df = pd.read_csv(csv_path)
    micro_codes = df["micro_label"].astype(str).str.split(" ").str[0].str.strip()

    col = f"{level}_id"  # macro_id, meso_id o micro_id
    if col not in df.columns:
        raise ValueError(f"Columna '{col}' no encontrada en {csv_path}")

    # Obtener categorías únicas y asignarles un índice
    unique_cats = sorted(df[col].unique())
    cat_value_to_idx = {int(v): i for i, v in enumerate(unique_cats)}
    num_categories = len(unique_cats)

    # Mapeo: micro_code → índice one-hot (basado en su categoría a nivel 'level')
    code_to_idx = {}
    for i, code in enumerate(micro_codes):
        cat_value = int(df.loc[i, col])
        code_to_idx[code] = cat_value_to_idx[cat_value]

    _ontology_index_cache[cache_key] = (code_to_idx, num_categories)
    return code_to_idx, num_categories
