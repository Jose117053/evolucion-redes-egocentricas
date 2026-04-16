import torch
import numpy as np
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score


def get_link_logits(z, edge_index):
    # z: [N, d]
    src, dst = edge_index
    # producto punto entre embeddings de los extremos
    # Si dos vectores apuntan a la misma direccion (son similares) entonces tendrán un valor positivo, en caso contrario
    return (z[src] * z[dst]).sum(dim=-1)  # [num_edges]


def split_edges(edge_index, test_ratio=0.2):
    """
    Separa las aristas en train (80%) y test (20%).

    IMPORTANTE: el message-passing de GraphSAGE usa TODAS las aristas
    (para preservar la topología). La separación solo aplica al cálculo
    del loss (train) y la evaluación del AUC (test).

    Retorna:
        train_edge_index: aristas para calcular el loss durante entrenamiento
        test_edge_index: aristas reservadas para evaluar AUC (nunca entrenadas)
    """
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)

    num_test = max(1, int(num_edges * test_ratio))
    num_train = num_edges - num_test

    train_edge_index = edge_index[:, perm[:num_train]]
    test_edge_index = edge_index[:, perm[num_train:]]

    return train_edge_index, test_edge_index


def eval_auc(model, data, test_edge_index, device):
    """
    Evalúa el AUC de Link Prediction usando SOLO las aristas de test
    (que el modelo nunca vio durante el entrenamiento).

    El forward pass usa TODAS las aristas (data.edge_index) para el
    message-passing, pero el AUC se calcula exclusivamente sobre las
    aristas reservadas.

    Retorna:
        float: AUC score (0.5 = azar, 1.0 = perfecto)
    """
    model.eval()
    with torch.no_grad():
        # Message-passing con TODAS las aristas (topología completa)
        z = model(data.x, data.edge_index)

        # Evaluar solo sobre aristas de test
        pos_logits = get_link_logits(z, test_edge_index)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,  # evitar tanto train como test
            num_nodes=data.num_nodes,
            num_neg_samples=test_edge_index.size(1),
            method="sparse"
        )
        neg_logits = get_link_logits(z, neg_edge_index)

        logits = torch.cat([pos_logits, neg_logits], dim=0).cpu().numpy()
        labels = np.concatenate([
            np.ones(pos_logits.size(0)),
            np.zeros(neg_logits.size(0))
        ])

        # Convertir logits a probabilidades con sigmoid
        probs = 1 / (1 + np.exp(-logits))

        return roc_auc_score(labels, probs)


def train_linkpred(model, data, optimizer, criterion, device, epochs=200, log_every=20):
    """
    Entrena Link Prediction con split de aristas 80/20.

    - Message-passing: usa TODAS las aristas (data.edge_index)
    - Loss: se calcula solo sobre aristas de TRAIN (80%)
    - AUC: se evalúa solo sobre aristas de TEST (20%)

    Retorna:
        (model, auc): modelo entrenado y AUC sobre aristas no vistas
    """
    # Separar aristas antes de entrenar
    train_edges, test_edges = split_edges(data.edge_index, test_ratio=0.2)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward pass con TODAS las aristas (message-passing completo)
        z = model(data.x, data.edge_index)

        # Loss solo sobre aristas de TRAIN
        pos_logits = get_link_logits(z, train_edges)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.size(1),
            method="sparse"
        )
        neg_logits = get_link_logits(z, neg_edge_index)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([
            torch.ones(pos_logits.size(0), device=device),
            torch.zeros(neg_logits.size(0), device=device)
        ], dim=0)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # Evaluar AUC sobre aristas de TEST (nunca vistas)
    final_auc = eval_auc(model, data, test_edges, device)

    return model, final_auc

