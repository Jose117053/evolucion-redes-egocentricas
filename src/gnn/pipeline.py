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
    in_channels = gnnDataLoader.get_in_channels(feature_mode)

    print(f"Feature mode: {feature_mode} (in_channels={in_channels})")
    print(f"Warm-start: {ecfg.PRESERVAR_HISTORIA}")

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
                scaler=global_scaler
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
            model = train.train_linkpred(
                model, data, optimizer, criterion, device,
                epochs=ecfg.EPOCHS_PER_YEAR
            )

            # ── Embeddings: EXCLUIR ego del readout ──
            model.eval()
            with torch.no_grad():
                z_nodes = model(data.x, data.edge_index)

                # Filtrar: solo embeddings de alters
                # El ego participó en message-passing pero su
                # embedding NO entra en el resumen del grafo.
                z_alters = z_nodes[alter_mask]  # [N_alters, out_dim]

                # Mean pooling SOLO sobre alters
                graph_embedding = z_alters.mean(dim=0)  # [out_dim]

                historia.append({
                    "year": year,
                    "vector": graph_embedding.cpu().numpy()
                })

            print(f"{year} procesado. (alters={alter_mask.sum().item()}, "
                  f"total_nodos={data.num_nodes})")

        except Exception as e:
            print(f"Error en {year}: {e}")

    print(f"Se generaron {len(historia)} puntos.")
    return historia