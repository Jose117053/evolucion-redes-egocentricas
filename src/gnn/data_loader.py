import os
import json
import torch
import numpy as np
import pandas as pd
import src.global_config as gcfg
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# DETECCIÓN DEL NODO EGO

def _is_ego_node(node):
    """
    Determina si un nodo es el Ego de la red egocéntrica.
    Criterio: su id empieza con 'MCT' O su Percent of documents es 100.0.
    """
    nid = str(node.get("id", ""))
    return (
        nid.startswith("MCT") or
        node.get("scores", {}).get("Percent of documents", 0) == 100.0
    )

# ─────────────────────────────────────────────────────────────────────
# ONTOLOGÍA COMO FEATURE: carga la taxonomía y construye un índice
# para generar vectores one-hot por nodo.
# ─────────────────────────────────────────────────────────────────────

_ontology_cache = {}  # Cache: (csv_path, level) → (cat_to_idx, num_categories)

def load_ontology_index(csv_path, level="macro"):
    """
    Carga el CSV de taxonomía y construye:
      - cat_to_idx: dict {micro_code → índice one-hot} según el nivel
      - num_categories: número total de categorías en ese nivel

    El micro_code es la clave (ej: '4.61.1335') que coincide con los IDs
    de los nodos en el JSON de la red.

    Niveles:
      'macro' → ~10 categorías (ej: macro_id=4)
      'meso'  → ~278 categorías (ej: meso_id=61)
      'micro' → ~1933 categorías (ej: micro_id=1335)
    """
    cache_key = (csv_path, level)
    if cache_key in _ontology_cache:
        return _ontology_cache[cache_key]

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

    _ontology_cache[cache_key] = (code_to_idx, num_categories)
    return code_to_idx, num_categories


# IN_CHANNELS se calcula dinámicamente según FEATURE_MODE
def get_in_channels(feature_mode, ontology_level="macro"):
    """
    Retorna el número de features de entrada según el modo.

    Modos disponibles:
      'ones'              → 1
      'bibliometric'      → len(FEATURES) (8)
      'ontology_only'     → num_categories del nivel ontológico
      'biblio_and_ontology' → len(FEATURES) + num_categories
    """
    if feature_mode == "bibliometric":
        return len(gcfg.FEATURES)
    elif feature_mode == "ones":
        return 1
    elif feature_mode in ("ontology_only", "biblio_and_ontology"):
        _, num_cats = load_ontology_index(gcfg.TAXONOMY_CSV, ontology_level)
        base = len(gcfg.FEATURES) if feature_mode == "biblio_and_ontology" else 0
        return base + num_cats
    else:
        raise ValueError(
            f"FEATURE_MODE no reconocido: '{feature_mode}'. "
            "Opciones: 'ones', 'bibliometric', 'ontology_only', 'biblio_and_ontology'"
        )

def get_ego_mask(raw_json):
    """
    Devuelve un tensor booleano del tamaño del número de nodos:
      True  → nodo ALTER  (se incluye en pooling y análisis)
      False → nodo EGO    (se excluye de pooling y análisis)

    Uso: aplicar esta máscara DESPUÉS del forward pass de GraphSAGE
    para excluir al ego del embedding agregado, sin quitarlo del
    message-passing (donde su presencia preserva la topología
    egocéntrica real).
    """
    mask = []
    for node in raw_json["network"]["items"]:
        mask.append(not _is_ego_node(node))
    return torch.tensor(mask, dtype=torch.bool)


# ─────────────────────────────────────────────────────────────────────
# SCALER GLOBAL (solo para feature_mode="bibliometric")
# ─────────────────────────────────────────────────────────────────────

def fit_global_scaler(data_folder, start_year, end_year, feature_keys, exclude_ego=True):
    """
    Ajusta un StandardScaler global sobre TODAS las ventanas temporales.
    Solo se usa cuando feature_mode="bibliometric".
    Para "ones" el scaler no es necesario.

    feature_keys: lista de tuplas (scope, key), ej. FEATURES.
    exclude_ego: si True, excluye nodos MCT del ajuste del scaler.
    """
    all_rows = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_folder, f"network_{year}.json")
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            raw_json = json.load(f)

        items = raw_json["network"]["items"]

        for node in items:
            if exclude_ego and _is_ego_node(node):
                continue

            row = []
            for scope, key in feature_keys:
                row.append(node.get(scope, {}).get(key, 0))
            all_rows.append(row)

    all_rows = np.asarray(all_rows, dtype=float)
    scaler = StandardScaler()
    scaler.fit(all_rows)
    return scaler


# ─────────────────────────────────────────────────────────────────────
# CONVERSIÓN JSON → PyG Data
# ─────────────────────────────────────────────────────────────────────

def json_to_pyg_data(json_data, feature_mode="bibliometric",
                     feature_keys=None, scaler=None, **kwargs):
    """
    Convierte un JSON de red egocéntrica a un objeto PyG Data.

    El nodo Ego PERMANECE en el grafo (edge_index y x) para que
    participe en el message-passing de GraphSAGE. La exclusión del
    ego se hace DESPUÉS, durante el pooling, usando get_ego_mask().

    Parámetros:
        feature_mode: str
            "bibliometric" → usa features bibliométricas (sin Documents,)
                             escaladas por el scaler global.
            "ones"         → x = ones(N, 1). Embedding puramente
                             topológico; GraphSAGE solo aprende de la
                             estructura de aristas.

        feature_keys: lista de tuplas (scope, key).
            Solo relevante cuando feature_mode="bibliometric".
            Si es None, se importa FEATURES de global_config.

        scaler: StandardScaler ajustado globalmente.
            Solo relevante cuando feature_mode="bibliometric".
            Si es None y mode es "bibliometric", ajusta uno local.

    Retorna:
        data: torch_geometric.data.Data con x, edge_index, edge_attr.
              edge_attr contiene 'strength' como metadato/compatibilidad
              futura — GraphSAGE no lo usa actualmente.
    """
    items = json_data["network"]["items"]
    links = json_data["network"]["links"]

    # ── Mapa de IDs a índices ──
    id_map = {node["id"]: i for i, node in enumerate(items)}
    num_nodes = len(items)

    # ── Aristas ──
    # strength se conserva como metadato en edge_attr, pero
    # SAGEConv no lo usa (no consume edge_attr por defecto), podria 
    # usarlo indirectamente a través de Documents, pues tienen el mismo valor.
    sources = []
    targets = []
    edge_weights = []

    for link in links:
        u = id_map[link["source_id"]]
        v = id_map[link["target_id"]]
        strength = link["strength"]

        sources.append(u)
        targets.append(v)
        edge_weights.append(strength)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

    # ── Features de nodo según feature_mode ──

    if feature_mode in ("bibliometric", "biblio_and_ontology"):
        if feature_keys is None:
            feature_keys = gcfg.FEATURES

        node_features = []
        for node in items:
            row = []
            for scope, key in feature_keys:
                row.append(node.get(scope, {}).get(key, 0))
            node_features.append(row)

        features_array = np.array(node_features, dtype=float)

        if scaler is None:
            scaler = StandardScaler().fit(features_array)
        features_scaled = scaler.transform(features_array)

        x_biblio = torch.tensor(features_scaled, dtype=torch.float)
    else:
        x_biblio = None

    if feature_mode in ("ontology_only", "biblio_and_ontology"):
        # Cargar el índice ontológico
        ontology_level = kwargs.get("ontology_level", "macro")
        code_to_idx, num_cats = load_ontology_index(
            gcfg.TAXONOMY_CSV, ontology_level
        )

        # Construir one-hot para cada nodo
        onehot_rows = []
        for node in items:
            nid = str(node.get("id", ""))
            vec = np.zeros(num_cats, dtype=float)
            if nid in code_to_idx:
                vec[code_to_idx[nid]] = 1.0
            # Si es ego (MCT...) o no está en el CSV, queda como zeros
            onehot_rows.append(vec)

        x_onto = torch.tensor(np.array(onehot_rows), dtype=torch.float)
    else:
        x_onto = None

    # ── Ensamblar x final ──
    if feature_mode == "ones":
        x = torch.ones(num_nodes, 1, dtype=torch.float)
    elif feature_mode == "bibliometric":
        x = x_biblio
    elif feature_mode == "ontology_only":
        x = x_onto
    elif feature_mode == "biblio_and_ontology":
        x = torch.cat([x_biblio, x_onto], dim=1)
    else:
        raise ValueError(
            f"feature_mode='{feature_mode}' no reconocido. "
            "Opciones: 'ones', 'bibliometric', 'ontology_only', 'biblio_and_ontology'"
        )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
