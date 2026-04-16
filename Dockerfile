# Construir:  docker-compose build
# Ejecutar:   docker-compose up
# Acceder:    http://localhost:8888

FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar algunos paquetes de python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para aprovechar la caché de Docker)
COPY requirements.txt .

# Instalar PyTorch CPU y el resto de dependencias, en caso de querer gpu, leer el readme para instalarlo
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt \
    jupyter

# Copiar el código fuente y datos
COPY src/ ./src/
COPY Data/ ./Data/
COPY Utils/ ./Utils/
COPY *.ipynb ./
COPY *.md ./
COPY *.tex ./

#RUN mkdir -p Output/ablation # Crear carpeta de output, pero ya se hace a traves de python

# Puerto de Jupyter
EXPOSE 8888

# Ejecutar Jupyter Notebook sin token ni contraseña (entorno local)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
    "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
