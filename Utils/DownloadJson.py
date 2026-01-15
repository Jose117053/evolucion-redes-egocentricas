import os
import requests
import time

START_YEAR = 1980
END_YEAR = 2024
BASE_URL = "https://www.c3.unam.mx/networks/som/ct_network_{}.json"
OUTPUT_FOLDER = "Data_Json_som"

def download_jsons():
    # Crear carpeta de salida si no existe
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Carpeta '{OUTPUT_FOLDER}' creada.")

    print(f"Iniciando descarga desde {START_YEAR} hasta {END_YEAR}\n")

    for year in range(START_YEAR, END_YEAR + 1):
        url = BASE_URL.format(year)
        filename = os.path.join(OUTPUT_FOLDER, f"network_{year}.json")
        
        try:
            print(f"Descargando: {url} ...", end=" ")
            response = requests.get(url, timeout=10)
            if response.status_code == 200: #El codigo 200 es para una descarga exitosa
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print("Guardado")
            else:
                print(f"Error {response.status_code} (No encontrado o servidor caído)")      
            # Pausa entre descargas
            time.sleep(0.5)

        except Exception as e:
            print(f"\n Falló la descarga de {year}: {e}")

    print("\n--- Proceso Finalizado ---")

download_jsons()
