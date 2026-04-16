import os
import json
import torch
import src.gnn.config as ecfg
import src.gnn.model as sage
import src.gnn.data_loader as gnnDataLoader
import src.gnn.train as train
import src.global_config as global_cfg

def run_temporal_graphsage(device):
    """
    Ejecuta el entrenamiento secuencial de GraphSAGE a través de los años,
    generando y guardando los embeddings a nivel de grafo.

    Cambios metodológicos respecto a la versión anterior:
    ─────────────────────────────────────────────────────
    1. FEATURE_MODE: controla las features de entrada (bibliometric/ones/degree).
       - "bibliometric": features bibliométricas SIN Documents, (evita circularidad).
       - "ones": embedding puramente topológico.
       - "degree": grado normalizado como única feature.

    2. EGO EXCLUIDO DEL POOLING: el nodo ego permanece en el grafo y
       participa en el message-passing (para preservar la topología
       egocéntrica real), pero se excluye del embedding agregado final
       para que este represente SOLO la comunidad de alters.

    3. PRESERVAR_HISTORIA: independiente de FEATURE_MODE. Controla si
       el modelo se transfiere entre años (warm-start) o se reinicia.
    """
    feature_mode = ecfg.FEATURE_MODE
    in_channels = gnnDataLoader.get_in_channels(feature_mode, ecfg.ONTOLOGY_FEATURE_LEVEL)

    print(f"Feature mode: {feature_mode} (in_channels={in_channels})")
    if feature_mode in ("ontology_only", "biblio_and_ontology"):
        print(f"Ontology feature level: {ecfg.ONTOLOGY_FEATURE_LEVEL}")
    print(f"Warm-start: {ecfg.PRESERVAR_HISTORIA}")
    print(f"Pooling mode: {ecfg.POOLING_MODE}")

    # ── Modelo inicial (si warm-start) ──
    if ecfg.PRESERVAR_HISTORIA:
        model = sage.GraphSAGEEncoder(
            in_channels, ecfg.HIDDEN_CHANNELS,
            ecfg.OUT_CHANNELS, ecfg.DROPOUT_RATE
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=ecfg.LR_INITIAL,
            weight_decay=ecfg.WEIGHT_DECAY
        )
        print("Modo: con memoria")
    else:
        print("Modo: Sin memoria")

    criterion = torch.nn.BCEWithLogitsLoss()
    historia = []
    auc_per_year = []

    # ── Scaler global: solo se necesita para "bibliometric" ──
    if feature_mode == "bibliometric":
        global_scaler = gnnDataLoader.fit_global_scaler(
            global_cfg.DATA_FOLDER,
            global_cfg.START_YEAR, global_cfg.END_YEAR,
            global_cfg.FEATURES,
            exclude_ego=True
        )
    else:
        global_scaler = None  # "ones" no necesitan scaler

    for year in range(global_cfg.START_YEAR, global_cfg.END_YEAR + 1):
        file_path = os.path.join(global_cfg.DATA_FOLDER, f"network_{year}.json")

        # Verificar si el archivo existe (El de 1990 no está en la plataforma del c3)
        if not os.path.exists(file_path):
            print(f"Saltando {year}: Archivo no encontrado.")
            continue

        try:
            with open(file_path, "r") as f:
                raw_json = json.load(f)

            # ── Construir el grafo PyG con el modo seleccionado ──
            # El ego PERMANECE en el grafo para message-passing.
            data = gnnDataLoader.json_to_pyg_data(
                raw_json,
                feature_mode=feature_mode,
                feature_keys=global_cfg.FEATURES,
                scaler=global_scaler,
                ontology_level=ecfg.ONTOLOGY_FEATURE_LEVEL
            )
            data = data.to(device)

            # ── Máscara de alters (True=alter, False=ego) ──
            # Se usa DESPUÉS del forward pass para excluir al ego
            # del embedding agregado, sin afectar message-passing.
            alter_mask = gnnDataLoader.get_ego_mask(raw_json).to(device)

            if not ecfg.PRESERVAR_HISTORIA:
                # Resetear modelo para cada snapshot
                model = sage.GraphSAGEEncoder(
                    in_channels, ecfg.HIDDEN_CHANNELS,
                    ecfg.OUT_CHANNELS, ecfg.DROPOUT_RATE
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=ecfg.LR_RESET,
                    weight_decay=ecfg.WEIGHT_DECAY
                )

            # Entrenar el modelo con el año actual
            model, year_auc = train.train_linkpred(
                model, data, optimizer, criterion, device,
                epochs=ecfg.EPOCHS_PER_YEAR
            )
            auc_per_year.append(year_auc)

            # ── Embeddings: EXCLUIR ego del readout ──
            model.eval()
            with torch.no_grad():
                z_nodes = model(data.x, data.edge_index)

                # Filtrar: solo embeddings de alters
                # El ego participó en message-passing pero su
                # embedding NO entra en el resumen del grafo.
                z_alters = z_nodes[alter_mask]  # [N_alters, out_dim]

                # ── Pooling según POOLING_MODE ──
                pooling_mode = ecfg.POOLING_MODE

                if pooling_mode == "mean":
                    graph_embedding = z_alters.mean(dim=0)

                elif pooling_mode == "weighted":
                    # Ponderado por Documents,
                    items = raw_json["network"]["items"]
                    docs = []
                    for node in items:
                        nid = str(node.get("id", ""))
                        if nid.startswith("MCT"):
                            continue  # skip ego
                        docs.append(float(node.get("weights", {}).get("Documents,", 1)))
                    weights = torch.tensor(docs, dtype=torch.float32, device=device)
                    weights = weights / weights.sum()  # normalizar
                    graph_embedding = (z_alters * weights.unsqueeze(1)).sum(dim=0)

                elif pooling_mode == "hierarchical":
                    # Promedio dentro de cada cluster ontológico, luego promedio de clusters
                    pooling_level = ecfg.POOLING_LEVEL
                    items = raw_json["network"]["items"]
                    clusters = []
                    for node in items:
                        nid = str(node.get("id", ""))
                        if nid.startswith("MCT"):
                            continue  # skip ego
                        parts = nid.split(".")
                        if pooling_level == "macro":
                            cluster_id = parts[0]
                        elif pooling_level == "meso":
                            cluster_id = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
                        else:  # micro
                            cluster_id = nid
                        clusters.append(cluster_id)

                    unique_clusters = sorted(set(clusters))
                    cluster_embeddings = []
                    for m in unique_clusters:
                        mask = torch.tensor(
                            [1 if clusters[i] == m else 0 for i in range(len(clusters))],
                            dtype=torch.bool, device=device
                        )
                        if mask.sum() > 0:
                            cluster_embeddings.append(z_alters[mask].mean(dim=0))

                    if cluster_embeddings:
                        graph_embedding = torch.stack(cluster_embeddings).mean(dim=0)
                    else:
                        graph_embedding = z_alters.mean(dim=0)  # fallback

                elif pooling_mode == "multi_stat":
                    # Concatenar [mean, max, std] para capturar más información
                    emb_mean = z_alters.mean(dim=0)
                    emb_max = z_alters.max(dim=0).values
                    emb_std = z_alters.std(dim=0) if z_alters.shape[0] > 1 else torch.zeros_like(emb_mean)
                    graph_embedding = torch.cat([emb_mean, emb_max, emb_std])

                else:
                    raise ValueError(f"POOLING_MODE '{pooling_mode}' no reconocido. "
                                     f"Opciones: mean, weighted, hierarchical, multi_stat")

                historia.append({
                    "year": year,
                    "vector": graph_embedding.cpu().numpy()
                })

            print(f"{year} procesado. (alters={alter_mask.sum().item()}, "
                  f"total_nodos={data.num_nodes}, pooling={pooling_mode}, AUC={year_auc:.4f})")

        except Exception as e:
            print(f"Error en {year}: {e}")

    # Calcular AUC promedio
    auc_mean = sum(auc_per_year) / len(auc_per_year) if auc_per_year else 0.0
    print(f"Se generaron {len(historia)} puntos. AUC promedio: {auc_mean:.4f}")
    return historia, auc_mean