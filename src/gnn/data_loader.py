import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def fit_global_scaler(data_folder, start_year, end_year, feature_keys, exclude_ego=True):
    """
    feature_keys: lista de tuplas (scope, key) como FEATURES
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
            if exclude_ego and (str(node.get("id","")).startswith("MCT")):
                continue

            row = []
            for scope, key in feature_keys:
                row.append(node.get(scope, {}).get(key, 0))
            all_rows.append(row)

    all_rows = np.asarray(all_rows, dtype=float)
    scaler = StandardScaler()
    scaler.fit(all_rows)
    return scaler

def json_to_pyg_data(json_data, scaler=None):
    items = json_data["network"]["items"]
    links = json_data["network"]["links"]

    id_map = {node["id"]: i for i, node in enumerate(items)}

    node_features = []
    for node in items:
        features = [
            node["weights"].get("WoS Categories", 0),
            node["weights"].get("Document Types",  0),
            node["weights"].get("Documents,", 0),
            node["scores"].get("Ave. citations",0.0),
            node["scores"].get("Ave. authorships",0.0),
            node["scores"].get("Ave. references",0.0),
         node["scores"].get("Percent of documents",0.0),
            node["scores"].get("Percent of documents Int. Coll.",0.0)
        ]
        node_features.append(features)
    
    #x = torch.tensor(node_features, dtype=torch.float)
    features_array = np.array(node_features, dtype=float)

    if scaler is None:
        scaler = StandardScaler().fit(features_array)
    features_scaled = scaler.transform(features_array)

    x = torch.tensor(features_scaled, dtype=torch.float)

    #Aristas
    sources = []
    targets = []
    edge_weights = []
    
    for link in links:
        u_str = link["source_id"]
        v_str = link["target_id"]
        strength = link["strength"]

        u = id_map[u_str]
        v = id_map[v_str]

        sources.append(u)
        targets.append(v)
        edge_weights.append(strength)


    edge_index = torch.tensor([sources, targets], dtype = torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1,1) #al ser de una sola dimensión, lo pasamos a una matriz de x filas por 1 columna


    #El tipo "Data" es exclusivamente para que sage pueda procesar el grafo
    data = Data(x=x, edge_index = edge_index, edge_attr = edge_attr)
    #x son mis caracteristicas
    #edge_index es la topologia (aristas)
    #edge_attr en este caso son únicamente los pesos

    return data

def get_ego_mask(raw_json):
    """
    Devuelve un tensor booleano:
    True  -> nodo ALTER
    False -> nodo EGO (MCT)
    """
    mask = []
    for node in raw_json["network"]["items"]:
        is_ego = (
            node["id"].startswith("MCT") or
            node["scores"].get("Percent of documents", 0) == 100.0
        )
        mask.append(not is_ego)
    return torch.tensor(mask, dtype=torch.bool)
