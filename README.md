Version de python usado: 3.12.9

La versión de torch en el requirements es para el modo cpu, para el modo gpu se debe instalar lo siguiente:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```