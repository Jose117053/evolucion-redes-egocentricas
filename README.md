## Entorno y Ejecución

La imagen de Docker utiliza **Python 3.10.slim**, que es la versión recomendada y más estable para las dependencias de PyTorch y PyTorch Geometric utilizadas en este entorno.

### Docker

Tener instalado docker:
- **Windows / macOS**: Instalar [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- **Linux**: Instalar `docker` y el plugin `docker-compose`.

### Uso con Docker Compose

Si se cuenta con `docker-compose` (o `docker compose`) instalado, simplemente abrir una terminal en la raíz de este proyecto y ejecuta:

```bash
docker compose up
```

Esto va a construir la imagen y arrancar el servidor. Cuando la terminal muestre `Jupyter Server is running at http://0.0.0.0:8888`, abrir la página `http://localhost:8888` en el navegador. Los resultados (Csvs) se van a guardar automáticamente en la carpeta local `Output/`

### Uso sin Docker Compose

Si no se cuenta con `docker-compose` instalado por defecto, se puede ejecutar el servicio utilizando comandos nativos de Docker:

```bash
# Construir la imagen (esto se hace una vez o si se cambio algo)
docker build -t sage .

# Ejecutar el contenedor montando la carpeta Output y mapeando al puerto 8888
docker run -p 8888:8888 -v ./Output:/app/Output sage
```
Hace lo mismo que Docker Compose.

---

## Estructura y Orden de Notebooks

Para asegurar la correcta ejecución del pipeline y que falten datos generados, los notebooks deben ejecutarse en el siguiente orden:

1. **`1_modelado_Estadistico.ipynb`**: Genera output para las siguientes 2 partes
2. **`2_modelado_GraphSAGE.ipynb`**: Extracción topológica y topológico-ontológica mediante GraphSAGE no supervisado y Link Prediction. (genera output que se usará en el tercer notebook).
3. **`3_experimentos.ipynb`**: Integración, correlación y métricas finales sobre los output producidos por los notebooks `1` y `2`.

---

## Entorno Local (Alternativa a Docker)

Si en lugar de Docker se prefiere gestionar con algun entorno local (ej., Conda o venv), los pasos serían:
1. Usar Python 3.10 o similar.
2. `pip install -r requirements.txt` o recrear el entorno usando `environment.yml`.

### Soporte para GPU (Solo para Entorno Local)

La imagen de Docker proporcionada está configurada exclusivamente para **CPU** por simplicidad y portabilidad.

Si se opta por dependencias en algun **entorno local** (ej. Conda o venv) y se cuenta con una GPU de Nvidia, se puede aprovechar para acelerar GraphSAGE. El `requirements.txt` por defecto instala PyTorch en su versión CPU. Para habilitar CUDA, desinstalar `torch` y reemplazarlo usando este comando:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
