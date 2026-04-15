import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from src.ontology import is_ego_node as _is_ego_node

def calcular_pca_2d(historia):
    """Aplica PCA para reducir a 2 dimensiones (para la trayectoria)."""
    vectores = np.array([item["vector"] for item in historia]) #array que contiene los vectores de cada año
    years = [item["year"] for item in historia] #para reetiquetar los puntos de acuerdo a su año

    pca = PCA(n_components=2) #Aplanar las 16 mensiones de cada embedding (out_channels=16) a unicamente 2 (coordenadas x,y)
    pca_result = pca.fit_transform(vectores) #hacemos uso del algoritmo de pca

    df_trayectoria = pd.DataFrame({ # DataFrame para poder graficarlo por medio de Plotly
        'x': pca_result[:, 0], #Todas las filas, pero solo la columna 0, es decir, el eje x
        'y': pca_result[:, 1],#Todas las filas, pero solo la columna 1, es decir, el eje y
        'year': years,
        'label': [str(y) for y in years] 
    })
    return df_trayectoria

def calcular_pca_1d(historia):
    """Aplica PCA para reducir a 1 dimensión (PC1)."""
    vectores = np.array([item["vector"] for item in historia])
    years = [item["year"] for item in historia]

    pca_1d = PCA(n_components=1)
    coord_y_original = pca_1d.fit_transform(vectores).flatten()

    # ¿Qué porcentaje de la información real está capturando esta línea?
    ev1 = pca_1d.explained_variance_ratio_[0] 
    
    # ¿Hacia dónde está apuntando esta línea en el espacio 8D?
    loading = pca_1d.components_[0]

    df_flujo = pd.DataFrame({
        'year': years,
        'y_original': coord_y_original
    })
    return df_flujo, {"ev1": ev1, "loading": loading}


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
def obtener_promedios_anuales(years, feature_names, data_folder):
    """
    Calcula el promedio de las features originales por cada año desde los JSON.
    
    Excluye al nodo Ego (MCT) del cálculo para que los
    promedios representen solo la comunidad de alters, consistente
    con la exclusión del ego en el pooling de GraphSAGE y en el
    módulo de stats.
    """
    
    features_anuales = []
    print("Calculando promedios anuales de los datos crudos (sin ego)...")
    
    for year in years:
        with open(os.path.join(data_folder, f"network_{year}.json"), "r") as f:
            raw_json = json.load(f)

        items = raw_json["network"]["items"]
        feats_year = []
        for feature in feature_names:
            vals = []
            for node in items:
                # Excluir al ego: su Percent of documents=100
                # distorsionaría todos los promedios
                if _is_ego_node(node):
                    continue
                val = node.get("weights", {}).get(feature) or node.get("scores", {}).get(feature, 0)
                vals.append(val)
            feats_year.append(np.mean(vals))
        
        features_anuales.append(feats_year)

    df_features = pd.DataFrame(features_anuales, columns=feature_names, index=years)
    return df_features

#Ya tenemos los datos como queremos, podemos proceder a realizar correlacion
#Lista A (Trayectoria generada por pca): 45 valores.
#Lista B (Promedio de cada feature): 45 valores.

# Simplementa para que PCA1 sea consistente multiplico por -1 en caso de que percent of documents sea negativo
# a pca no le importa  si es negativo o positivo
def correlacionar_y_alinear(df_flujo, df_features, particular_col="Percent of documents"):
    """
    Alinea el signo del PC1 basándose en la correlación con la métrica principal
    y calcula las correlaciones de Pearson para todas las features.
    """
    r_particular, _ = pearsonr(df_flujo["y_original"].values, df_features[particular_col].values)

    if r_particular < 0:
        df_flujo["y_original"] = -df_flujo["y_original"]
        print(f"Se invirtió el signo de PC1 para alinearlo con '{particular_col}'. Corr original: {r_particular:.4f}")
    else:
        print(f"PC1 ya estaba alineado con '{particular_col}'. Corr: {r_particular:.4f}")

    trayectoria_y = df_flujo["y_original"].values
    correlaciones = {}

    print("\nResultados de Correlación con el Eje Y PC1")
    for col in df_features.columns:
        r, p_value = pearsonr(trayectoria_y, df_features[col].values)
        correlaciones[col] = r
        print(f"{col}: {r:.4f}")
        
    return df_flujo, correlaciones

def preparar_datos_comparacion(df_features, df_flujo):
    """Une los dataframes y normaliza (0 a 1) para la gráfica de comparación."""
    df_plot = df_features.reset_index().rename(columns={'index': 'Year'})
    df_comparacion = pd.merge(df_plot, df_flujo, left_on='Year', right_on='year', how='inner')

    # Normalización Manual (Escala 0 a 1)
    df_comparacion['PD Crudos'] = df_comparacion['Percent of documents'] / df_comparacion['Percent of documents'].max()
    df_comparacion['PD PCA'] = df_comparacion['y_original'] / df_comparacion['y_original'].max()
    df_comparacion['Autores'] = df_comparacion['Ave. authorships'] / df_comparacion['Ave. authorships'].max()
    
    return df_plot, df_comparacion