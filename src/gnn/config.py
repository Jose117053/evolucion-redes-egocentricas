import src.gnn.data_loader as dl

PRESERVAR_HISTORIA = True  # True = Warm-start (transfiere aprendizaje al siguiente año). False = Inicia de cero.

# ─────────────────────────────────────────────────────────────────────
# FEATURE_MODE: controla qué features de nodo recibe GraphSAGE.
#
#   "bibliometric" → features bibliométricas 
#   "ones"         → x = ones(N,1). GraphSAGE aprende solo desde la
#                     topología del grafo.
#
# PRESERVAR_HISTORIA y FEATURE_MODE son independientes: se pueden
# combinar libremente (ej. ones + warm-start, bibliometric + reset(False)).
# ─────────────────────────────────────────────────────────────────────
FEATURE_MODE = "ones"

IN_CHANNELS = dl.get_in_channels(FEATURE_MODE)

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