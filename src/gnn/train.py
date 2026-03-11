import torch
from torch_geometric.utils import negative_sampling


def get_link_logits(z, edge_index):
    # z: [N, d]
    src, dst = edge_index
    # producto punto entre embeddings de los extremos
    # Si dos vectores apuntan a la misma direccion (son similares) entonces tendrán un valor positivo, en caso contrario
    return (z[src] * z[dst]).sum(dim=-1)  # [num_edges]

def train_linkpred(model, data, optimizer, criterion, device, epochs=200, log_every=20):
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        # aristas  positivas (reales)
        pos_edge_index = data.edge_index
        # aristas negativas
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),  # mismo número que positivas
            method="sparse"
        )
        # Logits para positivas y negativas
        pos_logits = get_link_logits(z, pos_edge_index)
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
    return model




