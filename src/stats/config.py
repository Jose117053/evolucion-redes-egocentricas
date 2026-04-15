LEVEL = "meso"
TAXONOMY_CSV = "./Data/citation_topics_2024.csv"
POOLING_MODE = "hierarchical"

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

CATEGORICAL_FEATURES = { #Variables categoricas ya etiquetados para su analicis
    "citas": {
        "unw_val": "unw_Ave_citations_value", "unw_cat": "unw_Ave_citations",
        "wdocs_val": "wdocs_Ave_citations_value", "wdocs_cat": "wdocs_Ave_citations",
        "titulo": "Citas Promedio"
    },
    "referencias": {
        "unw_val": "unw_Ave_references_value", "unw_cat": "unw_Ave_references",
        "wdocs_val": "wdocs_Ave_references_value", "wdocs_cat": "wdocs_Ave_references",
        "titulo": "Referencias Promedio"
    },
    "porcentaje_docs_int": {
        "unw_val": "unw_Percent_of_documents_Int_Coll_value", "unw_cat": "unw_Percent_of_documents_Int_Coll",
        "wdocs_val": "wdocs_Percent_of_documents_Int_Coll_value", "wdocs_cat": "wdocs_Percent_of_documents_Int_Coll",
        "titulo": "Porcentaje de Documentos de colaboración internacional"
    },
    "autores": {
        "unw_val": "unw_Ave_authorships_value", "unw_cat": "unw_Ave_authorships",
        "wdocs_val": "wdocs_Ave_authorships_value", "wdocs_cat": "wdocs_Ave_authorships",
        "titulo": "Autores Promedio"
    }
}