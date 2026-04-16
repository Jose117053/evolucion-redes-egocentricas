import src.gnn.data_loader as dl

PRESERVAR_HISTORIA = True  # True = Warm-start (transfiere aprendizaje al siguiente año). False = Inicia de cero.

# ─────────────────────────────────────────────────────────────────────
# FEATURE_MODE: controla qué features de nodo recibe GraphSAGE.
#
#   "ones"                → x = ones(N,1). GraphSAGE aprende solo desde
#                            la topología del grafo.
#   "bibliometric"        → 8 features bibliométricas (Documents, citas,
#                            etc.). Escaladas con StandardScaler global.
#   "ontology_only"       → One-hot del nivel ontológico (macro/meso/micro).
#                            GraphSAGE "ve" la disciplina de cada alter.
#   "biblio_and_ontology" → Concatenación de bibliometric + ontology_only.
#                            GraphSAGE ve perfil completo + disciplina.
#
# PRESERVAR_HISTORIA y FEATURE_MODE son independientes: se pueden
# combinar libremente (ej. ones + warm-start, bibliometric + reset(False)).
# ─────────────────────────────────────────────────────────────────────
FEATURE_MODE = "ones"

# Nivel ontológico para los features de nodo (solo aplica a
# "ontology_only" y "biblio_and_ontology"):
#   "macro" → ~10 categorías (one-hot de 10 dims)
#   "meso"  → ~278 categorías (one-hot de 278 dims)
#   "micro" → ~1933 categorías (NO RECOMENDADO: sobreajuste)
ONTOLOGY_FEATURE_LEVEL = "micro"

IN_CHANNELS = dl.get_in_channels(FEATURE_MODE, ONTOLOGY_FEATURE_LEVEL)

# Hiperparámetros de GraphSAGE
HIDDEN_CHANNELS = 32
OUT_CHANNELS = 8
DROPOUT_RATE = 0.1

# Hiperparámetros de entrenamiento
LR_INITIAL = 0.01            # Learning rate para la inicialización
LR_RESET = 0.005             # Learning rate alternativo si PRESERVAR_HISTORIA es False
WEIGHT_DECAY = 5e-4
EPOCHS_INITIAL = 200         # Épocas para probar/estabilizar el primer snapshot
EPOCHS_PER_YEAR = 60         # Épocas de fine-tuning durante la iteración de la serie de tiempo

# Configuración de preprocesamiento
EXCLUDE_EGO_IN_SCALER = True

# Parámetros de análisis estadístico
PCA_COMPONENTS_TRAJ = 2      # Para la gráfica de evolución 2D
PCA_COMPONENTS_PC1 = 1       # Para aislar la varianza principal (Nuestra variable Y)
ALIGN_PC1_ANCHOR = "Percent of documents"  # Variable para alinear el signo (evita que la gráfica se invierta sola)

HAC_LAGS = 3                 # Rezagos para los estimadores robustos de Newey-West

# ─────────────────────────────────────────────────────────────────────
# POOLING_MODE: cómo se agrega el embedding de los alters en un solo
# vector que represente al grafo completo (readout).
#
#   "mean"          → Promedio plano de todos los alters (actual).
#   "weighted"      → Promedio ponderado por Documents,.
#   "hierarchical"  → Promedio primero dentro de cada macro-cluster,
#                      luego promedio de los clusters. Cada disciplina
#                      contribuye por igual sin importar cuántos alters
#                      tenga.
#   "multi_stat"    → Concatena [mean, max, std] de los embeddings.
#                      OJO: el embedding resultante es 3× más grande
#                      (OUT_CHANNELS * 3).
# ─────────────────────────────────────────────────────────────────────
POOLING_MODE = "hierarchical"
POOLING_LEVEL = "meso"  # Nivel ontológico para hierarchical pooling: 'macro', 'meso', 'micro'