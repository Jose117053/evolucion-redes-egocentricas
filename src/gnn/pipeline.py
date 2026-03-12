import os
import json
import torch
from torch_geometric.nn import global_mean_pool
import src.gnn.config as ecfg
import src.gnn.model as sage
import src.gnn.data_loader as gnnDataLoader
import src.gnn.train as train
import src.global_config as global_cfg

def run_temporal_graphsage(device):
    """
    Ejecuta el entrenamiento secuencial de GraphSAGE a través de los años,
    generando y guardando los embeddings a nivel de grafo.
    """
    if ecfg.PRESERVAR_HISTORIA:
        model = sage.GraphSAGEEncoder(ecfg.IN_CHANNELS, ecfg.HIDDEN_CHANNELS, ecfg.OUT_CHANNELS, ecfg.DROPOUT_RATE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=ecfg.LR_INITIAL, weight_decay=ecfg.WEIGHT_DECAY)
        print("Modo: con memoria")
    else:
        print("Modo: Sin memoria")
        
    criterion = torch.nn.BCEWithLogitsLoss()
    historia = []
    
    global_scaler = gnnDataLoader.fit_global_scaler(global_cfg.DATA_FOLDER, global_cfg.START_YEAR, global_cfg.END_YEAR, global_cfg.FEATURES, exclude_ego=True)

    for year in range(global_cfg.START_YEAR, global_cfg.END_YEAR + 1):
        file_path = os.path.join(global_cfg.DATA_FOLDER, f"network_{year}.json")
        
        # Verificar si el archivo existe (El de 1990 no está en la plataforma del c3)
        if not os.path.exists(file_path):
            print(f"Saltando {year}: Archivo no encontrado.")
            continue
            
        try:
            with open(file_path, "r") as f:
                raw_json = json.load(f)
            
            data = gnnDataLoader.json_to_pyg_data(raw_json, scaler=global_scaler)
            data = data.to(device)

            if not ecfg.PRESERVAR_HISTORIA:
                # reseteamos al modelo anterior y creamos uno nuevo
                model = sage.GraphSAGEEncoder(ecfg.IN_CHANNELS, ecfg.HIDDEN_CHANNELS, ecfg.OUT_CHANNELS, ecfg.DROPOUT_RATE).to(device) 
                # renovar el optimizador para el nuevo modelo
                optimizer = torch.optim.Adam(model.parameters(), lr=ecfg.LR_RESET, weight_decay=ecfg.WEIGHT_DECAY) 

            # Entrenar el modelo con el año actual
            # No se pierde la información del año anterior si no se reseteo el modelo
            model = train.train_linkpred(model, data, optimizer, criterion, device, epochs=ecfg.EPOCHS_PER_YEAR) 
            
            # Embeddings de Nodos
            model.eval()
            with torch.no_grad():
                z_nodes = model(data.x, data.edge_index)
                
                # pooling para hacer un solo embedding de un snapshot
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                graph_embedding = global_mean_pool(z_nodes, batch) # Shape: [1, 16] (o la dimension de salida)
                
                # Guardar el resultado
                historia.append({
                    "year": year,
                    "vector": graph_embedding.cpu().numpy()[0] # Convertir a array plano
                })
                
            print(f"{year} procesado.")
            
        except Exception as e:
            print(f"Error en {year}: {e}")

    print(f"Se generaron {len(historia)} puntos.")
    return historia