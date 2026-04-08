"""
Módulo de Modularidad Ontológica
================================
Analiza la estructura modular de las redes egocéntricas usando la
ontología jerárquica de 3 niveles (macro / meso / micro) codificada
en los IDs de los nodos.

Métricas principales:
  - Distribución de clusters por nivel ontológico
  - Entropía de Shannon (diversidad disciplinaria)
  - Índice Herfindahl-Hirschman (concentración disciplinaria)
  - Perfiles bibliométricos agrupados por cluster
"""

import os
import json
import numpy as np
import pandas as pd
import src.global_config as gcfg


# ─────────────────────────────────────────────────────────────────────
# PARSING DE LA ONTOLOGÍA DESDE EL ID DEL NODO
# ─────────────────────────────────────────────────────────────────────

def extraer_ontologia(node_id: str) -> dict:
    """
    Parsea el ID de un nodo y extrae los 3 niveles ontológicos.
    
    Ejemplo: '4.61.1335' → {'macro': '4', 'meso': '4.61', 'micro': '4.61.1335'}
    Ejemplo ego: 'MCT.4.61.1335' → {'macro': '4', 'meso': '4.61', 'micro': '4.61.1335', 'is_ego': True}
    """
    is_ego = node_id.startswith("MCT")
    clean_id = node_id.replace("MCT.", "") if is_ego else node_id
    
    parts = clean_id.split(".")
    if len(parts) >= 3:
        return {
            'macro': parts[0],
            'meso': f"{parts[0]}.{parts[1]}",
            'micro': clean_id,
            'is_ego': is_ego
        }
    else:
        return {
            'macro': parts[0] if parts else 'unknown',
            'meso': clean_id,
            'micro': clean_id,
            'is_ego': is_ego
        }


# ─────────────────────────────────────────────────────────────────────
# DISTRIBUCIÓN DE CLUSTERS
# ─────────────────────────────────────────────────────────────────────

def calcular_distribucion_clusters(json_data: dict, nivel: str = "macro",
                                    excluir_ego: bool = True) -> dict:
    """
    Calcula la distribución (proporción) de alters por nivel ontológico.
    
    Parámetros:
        json_data: dict cargado del JSON de red
        nivel: 'macro', 'meso' o 'micro'
        excluir_ego: si True, excluye nodos MCT del conteo
        
    Retorna:
        dict {categoría: proporción} que suma 1.0
    """
    items = json_data["network"]["items"]
    conteo = {}
    total = 0
    
    for node in items:
        onto = extraer_ontologia(node["id"])
        if excluir_ego and onto['is_ego']:
            continue
        
        cat = onto[nivel]
        conteo[cat] = conteo.get(cat, 0) + 1
        total += 1
    
    if total == 0:
        return {}
    
    return {cat: count / total for cat, count in sorted(conteo.items())}


# ─────────────────────────────────────────────────────────────────────
# MÉTRICAS DE DIVERSIDAD / CONCENTRACIÓN
# ─────────────────────────────────────────────────────────────────────

def calcular_entropia(distribucion: dict) -> float:
    """
    Entropía de Shannon sobre la distribución de clusters.
    
    H = -Σ p_i * log2(p_i)
    
    Interpretación:
      - H = 0: todos los alters en un solo cluster (concentración total)
      - H alto: alters distribuidos uniformemente (interdisciplinaridad máxima)
    """
    proporciones = np.array(list(distribucion.values()))
    proporciones = proporciones[proporciones > 0]  # evitar log(0)
    
    if len(proporciones) == 0:
        return 0.0
    
    return float(-np.sum(proporciones * np.log2(proporciones)))


def calcular_hhi(distribucion: dict) -> float:
    """
    Índice de Herfindahl-Hirschman (HHI) sobre la distribución.
    
    HHI = Σ p_i²
    
    Interpretación:
      - HHI → 1: dominado por un solo cluster (monopolio disciplinario)
      - HHI → 1/N: distribución perfectamente uniforme
    """
    proporciones = np.array(list(distribucion.values()))
    
    if len(proporciones) == 0:
        return 1.0  # caso degenerado
    
    return float(np.sum(proporciones ** 2))


def calcular_n_clusters(distribucion: dict) -> int:
    """Número de clusters únicos presentes."""
    return len(distribucion)


def cluster_dominante(distribucion: dict) -> tuple:
    """
    Retorna (id_cluster, proporción) del cluster con mayor proporción.
    """
    if not distribucion:
        return ('none', 0.0)
    
    max_cat = max(distribucion, key=distribucion.get)
    return (max_cat, distribucion[max_cat])


# ─────────────────────────────────────────────────────────────────────
# PERFILES BIBLIOMÉTRICOS POR CLUSTER
# ─────────────────────────────────────────────────────────────────────

def perfil_por_cluster(json_data: dict, nivel: str = "macro",
                       excluir_ego: bool = True) -> pd.DataFrame:
    """
    Calcula el perfil bibliométrico promedio agrupado por cluster ontológico.
    
    Retorna un DataFrame con columnas:
      [cluster, n_alters, WoS Categories, Document Types, Documents,,
       Ave. citations, Ave. authorships, Ave. references,
       Percent of documents, Percent of documents Int. Coll.]
    """
    items = json_data["network"]["items"]
    weight_keys = ['WoS Categories', 'Document Types', 'Documents,']
    score_keys = ['Ave. citations', 'Ave. authorships', 'Ave. references',
                  'Percent of documents', 'Percent of documents Int. Coll.']
    
    grupos = {}
    
    for node in items:
        onto = extraer_ontologia(node["id"])
        if excluir_ego and onto['is_ego']:
            continue
        
        cat = onto[nivel]
        if cat not in grupos:
            grupos[cat] = []
        
        row = {}
        for k in weight_keys:
            row[k] = node.get("weights", {}).get(k, 0)
        for k in score_keys:
            row[k] = node.get("scores", {}).get(k, 0)
        
        grupos[cat].append(row)
    
    registros = []
    all_keys = weight_keys + score_keys
    
    for cat, nodos in sorted(grupos.items()):
        registro = {'cluster': cat, 'n_alters': len(nodos)}
        for k in all_keys:
            vals = [n[k] for n in nodos]
            registro[f'{k}_mean'] = np.mean(vals)
            registro[f'{k}_std'] = np.std(vals)
        registros.append(registro)
    
    return pd.DataFrame(registros)


# ─────────────────────────────────────────────────────────────────────
# SERIE TEMPORAL DE MODULARIDAD
# ─────────────────────────────────────────────────────────────────────

def resumen_modularidad_temporal(data_folder: str = None,
                                  start_year: int = None,
                                  end_year: int = None,
                                  nivel: str = "macro") -> pd.DataFrame:
    """
    Calcula métricas de modularidad para cada año del rango temporal.
    
    Retorna DataFrame con columnas:
      [year, n_alters, n_clusters, entropia, hhi,
       cluster_dom_id, cluster_dom_prop]
    """
    if data_folder is None:
        data_folder = gcfg.DATA_FOLDER
    if start_year is None:
        start_year = gcfg.START_YEAR
    if end_year is None:
        end_year = gcfg.END_YEAR
    
    registros = []
    
    for year in range(start_year, end_year + 1):
        filepath = os.path.join(data_folder, f"network_{year}.json")
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, "r") as f:
            json_data = json.load(f)
        
        items = json_data["network"]["items"]
        n_alters = sum(1 for n in items 
                       if not extraer_ontologia(n["id"])['is_ego'])
        
        dist = calcular_distribucion_clusters(json_data, nivel=nivel)
        entropia = calcular_entropia(dist)
        hhi = calcular_hhi(dist)
        n_clust = calcular_n_clusters(dist)
        dom_id, dom_prop = cluster_dominante(dist)
        
        registros.append({
            'year': year,
            'n_alters': n_alters,
            'n_clusters': n_clust,
            'entropia': round(entropia, 4),
            'hhi': round(hhi, 4),
            'cluster_dom_id': dom_id,
            'cluster_dom_prop': round(dom_prop, 4)
        })
    
    return pd.DataFrame(registros)


# ─────────────────────────────────────────────────────────────────────
# CARGA DE LA ONTOLOGÍA DESDE CSV
# ─────────────────────────────────────────────────────────────────────

def cargar_ontologia_csv(csv_path: str = None) -> pd.DataFrame:
    """
    Carga citation_topics_2024.csv con la ontología completa.
    
    Retorna DataFrame con columnas:
      [macro_id, macro_label, meso_id, meso_label, micro_id, micro_label]
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(gcfg.DATA_FOLDER), 
                                "citation_topics_2024.csv")
    
    return pd.read_csv(csv_path)


def enriquecer_con_labels(df_perfil: pd.DataFrame, 
                           ontologia_df: pd.DataFrame,
                           nivel: str = "macro") -> pd.DataFrame:
    """
    Añade las etiquetas legibles (e.g. '4 → Physical Sciences & Engineering')
    al DataFrame de perfiles por cluster.
    """
    if nivel == "macro":
        label_map = dict(zip(
            ontologia_df['macro_id'].astype(str),
            ontologia_df['macro_label']
        ))
    elif nivel == "meso":
        label_map = dict(zip(
            ontologia_df['meso_id'].astype(str),
            ontologia_df['meso_label']
        ))
    else:
        label_map = dict(zip(
            ontologia_df['micro_id'].astype(str),
            ontologia_df['micro_label']
        ))
    
    df_perfil = df_perfil.copy()
    df_perfil['cluster_label'] = df_perfil['cluster'].map(label_map)
    
    return df_perfil
