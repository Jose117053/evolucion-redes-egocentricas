PRESERVAR_HISTORIA = True  # True = Warm-start (transfiere aprendizaje al siguiente año). False = Inicia de cero.

# Features de los nodos
FEATURES = [
    ("weights", "WoS Categories"),
    ("weights", "Document Types"),
    ("weights", "Documents,"),
    ("scores",  "Ave. citations"),
    ("scores",  "Ave. authorships"),
    ("scores",  "Ave. references"),
    ("scores",  "Percent of documents"),
    ("scores",  "Percent of documents Int. Coll.")
]

# Configuración de preprocesamiento
EXCLUDE_EGO_IN_SCALER = True

# Hiperparámetros de graphsage
IN_CHANNELS = len(FEATURES)
HIDDEN_CHANNELS = 32
OUT_CHANNELS = 8
DROPOUT_RATE = 0.1

# Hiperparámetros de entrenamiento
LR_INITIAL = 0.01            # Learning rate para la inicialización
LR_RESET = 0.005             # Learning rate alternativo si PRESERVAR_HISTORIA es False
WEIGHT_DECAY = 5e-4
EPOCHS_INITIAL = 200         # Épocas para probar/estabilizar el primer snapshot
EPOCHS_PER_YEAR = 60         # Épocas de fine-tuning durante la iteración de la serie de tiempo

# Parámetros de análisis estadístico
PCA_COMPONENTS_TRAJ = 2      # Para la gráfica de evolución 2D
PCA_COMPONENTS_PC1 = 1       # Para aislar la varianza principal (Nuestra variable Y)
ALIGN_PC1_ANCHOR = "Percent of documents"  # Variable para alinear el signo (evita que la gráfica se invierta sola)

HAC_LAGS = 3                 # Rezagos para los estimadores robustos de Newey-West