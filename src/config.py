DATA_FOLDER = "./Data/SOM"
START_YEAR = 1980
END_YEAR = 2024

LEVEL = "meso"
TAXONOMY_CSV = "./Data/citation_topics_2024.csv"

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


DOCS_COL = "Documents,"  # masa para ponderar

CAT_SPECS = {
    "Ave. citations": {
        "n_bins": 4,
        "probs": [0.25, 0.50, 0.75],
        "labels": ["baja", "media", "media_alta", "alta"]
    },
    "Ave. references": {
        "n_bins": 4,
        "probs": [0.25, 0.50, 0.75],
        "labels": ["baja", "media", "media_alta", "alta"]
    },
    "Percent of documents Int. Coll.": {
        "n_bins": 4,
        "probs": [0.25, 0.50, 0.75],
        # este feature a veces queda en 3 bins (baja, media, media_alta) por edges=[0,25]
        # entonces aquí ponemos el set "máximo" y abajo el código ignora las que no existan
        "labels": ["baja", "media", "media_alta", "alta"]
    },
    "Ave. authorships": {
        "n_bins": 3,
        "probs": [1/3, 2/3],
        "labels": ["baja", "media", "alta"]
    }
}

MODO_CUANTILES = "global" # Puede ser "global" o "window"
TAMANO_VENTANA = 5        # Tamaño de intervalos para tomar en cuenta los percentiles