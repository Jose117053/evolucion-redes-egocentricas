"""
src/gnn/ablation.py
===================
Sistema de logging para experimentos de ablación.
Guarda los resultados de cada corrida de GraphSAGE en un CSV
con nombre dinámico según la configuración activa.

Estructura de archivos:
  Output/ablation/{dataset}_{feature_mode}[_{feature_level}]_{pooling_mode}[-{pooling_level}].csv
"""

import os
import pandas as pd
import src.gnn.config as ecfg
import src.global_config as gcfg


def get_dataset_name():
    """Extrae el nombre del dataset desde DATA_FOLDER (ej: 'SOM' de './Data/SOM')."""
    return os.path.basename(os.path.normpath(gcfg.DATA_FOLDER))


def get_ablation_filename():
    """
    Construye el nombre del archivo CSV dinámicamente según la configuración activa.

    Ejemplos:
      - SOM_ones_mean.csv
      - SOM_ones_hierarchical-meso.csv
      - SOM_ontology-macro_mean.csv
      - SOM_biblio-and-ontology-macro_hierarchical-meso.csv
    """
    dataset = get_dataset_name()
    feature_mode = ecfg.FEATURE_MODE

    # Feature part
    if feature_mode in ("ontology_only", "biblio_and_ontology"):
        feature_part = f"{feature_mode.replace('_', '-')}-{ecfg.ONTOLOGY_FEATURE_LEVEL}"
    else:
        feature_part = feature_mode

    # Pooling part
    pooling_mode = ecfg.POOLING_MODE
    if pooling_mode == "hierarchical":
        pooling_part = f"hierarchical-{ecfg.POOLING_LEVEL}"
    else:
        pooling_part = pooling_mode

    filename = f"{dataset}_{feature_part}_{pooling_part}.csv"

    # Crear directorio si no existe
    ablation_dir = os.path.join(gcfg.OUTPUT_DIR, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    return os.path.join(ablation_dir, filename)


def log_run(auc_mean, rho_entropia=None, rho_pc1_raw=None):
    """
    Registra una corrida experimental en el CSV de ablación.

    Parámetros:
        auc_mean: float, AUC promedio de Link Prediction sobre todos los años.
        rho_entropia: float o None, correlación Pearson de PC1_sage vs entropía.
        rho_pc1_raw: float o None, correlación Pearson de PC1_sage vs PC1_raw.

    El número de corrida se autoincrementa a partir de las filas existentes.
    """
    filepath = get_ablation_filename()

    row = {
        "corrida": 1,
        "feature_mode": ecfg.FEATURE_MODE,
        "pooling_mode": ecfg.POOLING_MODE,
        "pooling_level": ecfg.POOLING_LEVEL if ecfg.POOLING_MODE == "hierarchical" else "N/A",
        "ontology_feature_level": ecfg.ONTOLOGY_FEATURE_LEVEL if ecfg.FEATURE_MODE in ("ontology_only", "biblio_and_ontology") else "N/A",
        "auc_mean": round(auc_mean, 4),
        "rho_entropia": round(rho_entropia, 4) if rho_entropia is not None else None,
        "rho_pc1_raw": round(rho_pc1_raw, 4) if rho_pc1_raw is not None else None,
    }

    # Leer CSV existente y autoincrement corrida
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        row["corrida"] = int(df_existing["corrida"].max()) + 1
        df_new = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])

    df_new.to_csv(filepath, index=False)
    print(f"[Ablation] Corrida {row['corrida']} guardada en: {filepath}")
    print(f"           AUC={row['auc_mean']}, ρ_entropia={row['rho_entropia']}, ρ_pc1_raw={row['rho_pc1_raw']}")

    return filepath
