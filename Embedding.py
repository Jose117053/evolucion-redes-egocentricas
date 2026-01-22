import torch
import json
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from minisom import MiniSom
import plotly.express as px
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr, kendalltau

def json_to_pyg_data(json_data):
    items = json_data["network"]["items"]
    links = json_data["network"]["links"]

    id_map = {node["id"]: i for i, node in enumerate(items)}

    node_features = []
    for node in items:
        features = [
            node["weights"].get("WoS Categories", 0),
            node["weights"].get("Document Types",  0),
            node["weights"].get("Documents,",0),
            node["scores"].get("Ave. citations",0.0),
            node["scores"].get("Ave. authorships",0.0),
            node["scores"].get("Ave. references",0.0),
            node["scores"].get("Percent of documents",0.0),
            node["scores"].get("Percent of documents Int. Coll.",0.0)
        ]
        node_features.append(features)
    
    #x = torch.tensor(node_features, dtype=torch.float)
    features_array = np.array(node_features, dtype=float)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
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

with open("./Data/BN/network_2003.json", "r") as f:
    raw_json = json.load(f)

snapshot_2003 = json_to_pyg_data(raw_json)
print(f"Snapshot creado: {snapshot_2003}")
print(f"Nodos: {snapshot_2003.num_nodes}, Features: {snapshot_2003.num_node_features}")

data = snapshot_2003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)
data = data.to(device)

class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x  # embeddings finales
    
in_channels = data.num_node_features   # En nuestro caso son 8 features
hidden_channels = 32
out_channels = 20   # dimensión del embedding final

model = GraphSAGEEncoder(in_channels, hidden_channels, out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()


def get_link_logits(z, edge_index):
    # z: [N, d]
    src, dst = edge_index
    # producto punto entre embeddings de los extremos
    # Si dos vectores apuntan a la misma direccion (son similares) entonces tendrán un valor positivo, en caso contrario
    return (z[src] * z[dst]).sum(dim=-1)  # [num_edges]

def train_linkpred(model, data, epochs=200):
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
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
    return model

trained_model = train_linkpred(model, data, epochs=200)
trained_model.eval()
with torch.no_grad():
    z = trained_model(data.x, data.edge_index)  # [num_nodes, out_channels]

embeddings = z.cpu().numpy()
print("dimension de embeddings:", embeddings.shape)  # (num_nodes, out_channels)
print(embeddings[0])

items = raw_json["network"]["items"]
clusters = [node.get("cluster", -1) for node in items]  # lista de clusters

# Crear el vector batch (todos los nodos tienen la etiqueta 0 porque es un solo grafo)
# z es la matriz de embeddings de todos los nodos, es de la forma [No.Nodos, out_channels]
print(z.shape)
batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

# Hacemos el pooling usando los embeddings de nodos ya entrenados (z)
# El pooling hace un "promedio" en base al tamaño del batch, en nuestro caso
# al poner batch como un tensor llenos de 0's, entonces promedia todos aquellos que tengan como indice 0 (todos los nodos)
#Devuelve una matriz de [1, 16], "comprime" todos los embeddings a uno solo
graph_embedding = global_mean_pool(z, batch)
print("Dimension del embedding del grafo:", graph_embedding.shape)
print(graph_embedding)

PRESERVAR_HISTORIA = True
START_YEAR = 1980
END_YEAR = 2024
DATA_FOLDER = "./Data/SOM"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if PRESERVAR_HISTORIA:
    model = GraphSAGEEncoder(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print("Modo: con memoria")
else:
    print("Modo: Sin memoria")
# lilsta de eembeddings de cada año
historia = []

for year in range(START_YEAR, END_YEAR + 1):
    file_path = os.path.join(DATA_FOLDER, f"network_{year}.json")
    
    # Verificar si el archivo existe 
    # El de 1990 no está en la plataforma del c3
    if not os.path.exists(file_path):
        print(f"Saltando {year}: Archivo no encontrado.")
        continue
        
    try:
        with open(file_path, "r") as f:
            raw_json = json.load(f)
        
        data = json_to_pyg_data(raw_json)
        data = data.to(device)

        if not PRESERVAR_HISTORIA:
            model = GraphSAGEEncoder(in_channels, hidden_channels, out_channels).to(device) #reseteamos al modelo anterior y creamos uno nuevo
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) #renovar el optimizador para el nuevo modelo

        
        # Entrenar el modelo con el año actual
        # No se pierde la información del año anterior si no se reseteo el modelo
        model = train_linkpred(model, data, epochs=50) 
        
        #Embeddings de Nodos
        model.eval()
        with torch.no_grad():
            z_nodes = model(data.x, data.edge_index)
            
            # pooling para hacer un solo embedding de un snapshot
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            graph_embedding = global_mean_pool(z_nodes, batch) # Shape: [1, 16]
            
            # Guardar el resultado
            historia.append({
                "year": year,
                "vector": graph_embedding.cpu().numpy()[0] # Convertir a array plano
            })
            
        print(f"{year} procesado.")
        
    except Exception as e:
        print(f"Error en {year}: {e}")

print(f"Se generaron {len(historia)} puntos.")

print(graph_embedding.shape)


vectores = np.array([item["vector"] for item in historia]) #array que contiene los vectores de cada año
years = [item["year"] for item in historia] #para reetiquetar los puntos de acuerdo a su año

# Reducir a 2D
pca = PCA(n_components=2) #Aplanar las 16 mensiones de cada embedding (out_channels=16) a unicamente 2 (coordenadas x,y)
pca_result = pca.fit_transform(vectores) #hacemos uso del algoritmo de pca

# DataFrame para poder graficarlo por medio de Plotly
df_trayectoria = pd.DataFrame({
    'x': pca_result[:, 0], #Todas las filas, pero solo la columna 0, es decir, el eje x
    'y': pca_result[:, 1], #Todas las filas, pero solo la columna 1, es decir, el eje y
    'year': years,
    'label': [str(y) for y in years] 
})

#Graficar la Trayectoria
fig = px.scatter(df_trayectoria, x='x', y='y', 
                 text='label', 
                 color='year',
                 title="Evolución")

# líneas para conectar los puntos
fig.update_traces(mode='lines+markers+text', textposition='top center')

fig.show()




vectores = np.array([item["vector"] for item in historia])
years = [item["year"] for item in historia]

pca_1d = PCA(n_components=1)
coord_y_original = pca_1d.fit_transform(vectores).flatten()

df_flujo = pd.DataFrame({
    'year': years,
    'y_original': coord_y_original
})

# Grafico de linea
fig = px.line(
    df_flujo, 
    x='year', 
    y='y_original',
    title="Evolucion",
    labels={'y_original': 'PC1', 'year': 'Año'},
    markers=True, # ñuntos de cada año
    template="plotly_white"
)

#fig.update_traces(line_shape='spline', line_width=4) #esto es unicamente para "suavizar las esquinas"
fig.show()

# Correlación de pearson dice qué tan fuerte es la relación lineal entre dos variables
# Cuando la variable A sube, ¿ldf_flujoa variable B también sube proporcionalmente?
trayectoria_y = df_flujo['y_original'].values
years = df_flujo['year'].values

feature_names = [
    "WoS Categories", "Document Types", "Documents,", 
    "Ave. citations", "Ave. authorships", "Ave. references", 
    "Percent of documents", "Percent of documents Int. Coll."
]

# Matriz para guardar el promedio de cada feature por año
# Tamaño: [num_años, num_features]
features_anuales = []

print("Promedios anuales")
#¿Por qué es necesario hacer un promedio?
#Para poder ocupar la correlacion de pearson, debemos de comparar los 45 puntos resultantes del pca con otros 45 puntos.
#(La comparación es 1 a 1 por cada feature, es decir, fijamos un feature y lo comparamos con los 45 tiempos)
#Sin embargo, cada año tiene cierta cantidad de nodos, es necesario obtener un solo valor a partir de todos estos.
#por eso se realiza un promedio

#Para cada año (cada json definido en la carpeta Data_Json)
#Para cada feature (feature_names)
#Añadimos el valor de ese feature de cada nodo a la lista "vals"
#al finalizar de contabilizar este feature, realizamos un promedio y lo agregamos a feats_year
#continuamos con el siguiente feature, continuamos cn el siguiente año hasta terminar
for year in years:
    with open(f"./Data/SOM/network_{year}.json", "r") as f:
        raw_json = json.load(f)

    items = raw_json["network"]["items"]
    feats_year = []
    for feature in feature_names:
        # Extraer el valor de cada nodo
        vals = []
        for node in items:
            val = node.get("weights", {}).get(feature) or node.get("scores", {}).get(feature, 0)
            vals.append(val)
        # Guardar el promedio de este año
        feats_year.append(np.mean(vals))
    
    features_anuales.append(feats_year) #cada fila es un año, y 8 columnas, cada una es el promedio de toda la grafica de un solo feature

df_features = pd.DataFrame(features_anuales, columns=feature_names, index=years)


#Ya tenemos los datos como queremos, podemos proceder a realizar correlacion
#Lista A (Trayectoria generada por pca): 45 valores.
#Lista B (Promedio de cada feature): 45 valores.


correlaciones = {}

print("\n--- Resultados de Correlación con el Eje Y PC1")
for col in df_features.columns:
    # Correlación de Pearson (r),  rango entre [-1,1], sí se acerca a 1 positivo, hay bastante correlación
    r, p_value = pearsonr(trayectoria_y, df_features[col].values) #aqui ya están los 45 años y 45 promedios de cada feature
    correlaciones[col] = r
    print(f"{col}: {r:.4f}")

# Con los valores basta, pero para tener una mejor visualización:
plt.figure(figsize=(10, 6))
sorted_corr = pd.Series(correlaciones).sort_values()# los más importantes primero
colors = ['red' if x < 0 else 'blue' for x in sorted_corr]
sorted_corr.plot(kind='barh', color=colors)
plt.title("Correlación entre Features y Trayectoria")
plt.xlabel("Correlación de Pearson")
plt.axvline(0, color='black', linewidth=0.8)
plt.show()

#Percent of documents resultó  tener una correlación bastante alta, por lo tanto, las gráficas
#deberían ser bastante parecidas (la generada por pca y la que es a partir de los datos crudos)
atributo_a_graficar = "Percent of documents"
#datos que calculamos en el paso anterior (df_features)
df_plot = df_features.reset_index().rename(columns={'index': 'Year'})
fig = px.line(
    df_plot, 
    x='Year', 
    y=atributo_a_graficar,
    title=f"Evolución de: {atributo_a_graficar}",
    markers=True,
    template="plotly_white"
)

# esto es solo para suavisar las esquinas
#fig.update_traces(line_shape='spline', line_width=3)
fig.show()



# Unir las tablas 
# Usamos 'inner' para quedarnos solo con los años que existen en ambos lados
df_comparacion = pd.merge(df_plot, df_flujo, left_on='Year', right_on='year', how='inner')

# Normalización Manual (Escala 0 a 1)
# Para poder comparar directamente las tendencias
#PD es percent of documents

#Por cada atributo definido, se busca al maximo, y a cada valor (fila) es dividido entre ese valor
df_comparacion['PD Crudos'] = df_comparacion['Percent of documents'] / df_comparacion['Percent of documents'].max()
df_comparacion['PD PCA'] = df_comparacion['y_original'] / df_comparacion['y_original'].max()
df_comparacion['Autores'] = df_comparacion['Ave. authorships'] / df_comparacion['Ave. authorships'].max()
# Graficar
fig = px.line(
    df_comparacion, 
    x='Year', 
    y=['PD Crudos', 'PD PCA', 'Autores'], 
    title="Comparación de evoluciónes de los embeddings contra los datos crudos",
    labels={'value': 'Intensidad Relativa (0-1)', 'variable': 'Curva', 'Year': 'Año'},
    template="plotly_white",
    markers=False
)

fig.update_traces(line=dict(width=3)) 
fig.update_traces(patch={"line": {"dash": "dot"}}, selector={"name": "Autores"})
fig.update_traces(patch={"line": {"dash": "dot"}}, selector={"name": "PD Crudos"})
fig.show()