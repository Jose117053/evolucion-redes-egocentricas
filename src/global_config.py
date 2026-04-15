import os

DATA_FOLDER = "./Data/SOM" #Carpeta del que se toman los datos
OUTPUT_DIR = "./Output" #Carpeta en el que se guardaran los .csv
TAXONOMY_CSV = "./Data/citation_topics_2024.csv"  # Ontología jerárquica de tópicos
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 1980
END_YEAR = 2024

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