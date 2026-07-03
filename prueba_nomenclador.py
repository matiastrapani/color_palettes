import os
import csv
import json
import numpy as np
from skimage import color

def generar_base_nomenclador(archivo_csv: str = "nomencladores.csv") -> None:
    """
    Lee el archivo CSV de colores, calcula sus coordenadas espaciales CIELAB
    y compila un diccionario unificado en formato JSON sin perder duplicados.
    """
    if not os.path.exists(archivo_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {archivo_csv}")

    # 1. Detectar automáticamente la codificación del archivo para evitar UnicodeDecodeError
    encoding_detectado = "utf-8-sig"
    try:
        with open(archivo_csv, mode="rb") as f_bin:
            bloque = f_bin.read(2048)
            bloque.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback a la codificación típica de Excel en sistemas en español
        encoding_detectado = "latin-1"

    tabla_nomenclador = {}

    # 2. Procesar el archivo CSV con la codificación segura
    with open(archivo_csv, mode="r", encoding=encoding_detectado) as f:
        # Sniffer para detectar si el separador es coma o punto y coma
        dialecto = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        
        lector = csv.DictReader(f, dialect=dialecto)
        
        # Validar las columnas requeridas limpiando espacios ocultos en las cabeceras
        columnas_limpias = [col.strip() for col in lector.fieldnames] if lector.fieldnames else []
        for col in ["Nombre", "R", "G", "B"]:
            if col not in columnas_limpias:
                raise ValueError(f"Falta la columna requerida en el CSV: {col}")

        # 3. Iterar y calcular coordenadas CIELAB
        for fila in lector:
            # Limpiar espacios en blanco de las celdas
            fila_limpia = {k.strip(): v.strip() for k, v in fila.items() if k}
            
            nombre_base = fila_limpia.get("Nombre", "").strip()
            if not nombre_base:
                continue
                
            # Resolver duplicados agregando un índice correlativo para no pisar datos
            nombre = nombre_base
            contador = 1
            while nombre in tabla_nomenclador:
                nombre = f"{nombre_base} {contador}"
                contador += 1
                
            try:
                # Normalizar RGB al rango [0, 1] que exige skimage
                r = float(fila_limpia["R"]) / 255.0
                g = float(fila_limpia["G"]) / 255.0
                b = float(fila_limpia["B"]) / 255.0
                
                # Transformación matemática estricta al espacio perceptualmente uniforme CIELAB
                rgb_array = np.array([r, g, b], dtype=np.float64).reshape((1, 1, 3))
                lab_convertido = color.rgb2lab(rgb_array).flatten()
                
                # Estructurar la entidad de color completa
                tabla_nomenclador[nombre] = {
                    "lab": [float(lab_convertido[0]), float(lab_convertido[1]), float(lab_convertido[2])],
                    "hex": fila_limpia.get("Cod. Hex.", ""),
                    "rgb_original": [int(fila_limpia["R"]), int(fila_limpia["G"]), int(fila_limpia["B"])]
                }
            except (ValueError, KeyError) as e:
                print(f"Advertencia: Saltando fila inválida o corrupta para el color '{nombre_base}': {e}")
                continue

    # 4. Guardar el JSON resultante en el directorio de módulos
    ruta_salida = os.path.join("modulos", "tabla_colores.json")
    os.makedirs("modulos", exist_ok=True)
    
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(tabla_nomenclador, f, ensure_ascii=False, indent=4)
    
    print("--- Compilation exitosa ---")
    print(f"Codificación utilizada: [{encoding_detectado}]")
    print(f"Total de muestras procesadas: {len(tabla_nomenclador)} colores.")
    print(f"Archivo JSON exportado en: {ruta_salida}")

if __name__ == "__main__":
    # Asegurate de que el nombre coincida exactamente con tu archivo CSV real
    generar_base_nomenclador("nomencladores.csv")