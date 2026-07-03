import cv2
import numpy as np
import torch
import gc
import os
from typing import List
from ultralytics.models.sam import SAM3SemanticPredictor

def segmentacion_final_completa():
    # 1. Configuración
    USAR_TILES = False

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="models/sam3.pt",
        half=True,
        device="mps",
        imgsz=1036,
        save=False,
        project=None,
        name=None
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    directorio_entrada = "entrada/isleta"
    filename = "PIÑEYRO_ISLETA_260616_F1971.jpg"
    img_path = os.path.join(directorio_entrada, filename)
    output_path = os.path.join("salida/isleta", f"MASC_{filename}")

    img_orig = cv2.imread(img_path)
    if img_orig is None: return
    h, w = img_orig.shape[:2]
    
    sumar = ["building facade", "pavement", "window", "door", "urban furniture", "sign", "roof", "column", "architecture", "air conditioner", "wall", "molding", "tree", "grass"]
    restar = ["car", "sky", "vehicle"]
    
    # 2. Lógica de Parches (Tiling) para imágenes HD
    tile_size = 1036
    overlap = 50

    mascara_final = np.zeros((h, w), dtype=np.uint8)

    print("🚀 Iniciando procesamiento por parches...")
    paso = tile_size - overlap
    tiles_x = int(np.ceil((w - tile_size) / paso) + 1) if w > tile_size else 1
    tiles_y = int(np.ceil((h - tile_size) / paso) + 1) if h > tile_size else 1
    total_tiles = tiles_x * tiles_y
    tile = 0

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            tile =+ 1
            print(f"paso {tile}/{total_tiles}")
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            patch = img_orig[y:y_end, x:x_end]
            
            # Crear mascara local para este parche
            mascara_patch = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
            
            # Procesar conceptos (suma/resta) dentro del parche
            def procesar_logica(lista, es_suma):
                nonlocal mascara_patch
                for concepto in lista:
                    results = predictor(source=patch, text=[concepto])
                    if results and results[0].masks is not None:
                        for mask in results[0].masks.data:
                            m = cv2.resize(mask.cpu().numpy().astype(np.uint8), (patch.shape[1], patch.shape[0]))
                            if es_suma: mascara_patch = cv2.bitwise_or(mascara_patch, m)
                            else: mascara_patch = cv2.bitwise_and(mascara_patch, cv2.bitwise_not(m))
                    del results; gc.collect(); torch.mps.empty_cache()

            procesar_logica(sumar, True)
            #procesar_logica(restar, False)
            
            # Unir parche a la máscara global
            mascara_final[y:y_end, x:x_end] = cv2.bitwise_or(mascara_final[y:y_end, x:x_end], mascara_patch)




    # 3. Aplicar efecto velado
    fondo_blanco = np.full_like(img_orig, 255, dtype=np.uint8)
    velado = cv2.addWeighted(img_orig, 0.15, fondo_blanco, 0.85, 0)
    
    resultado_final = velado.copy()
    resultado_final[mascara_final > 0] = img_orig[mascara_final > 0]
    
    # 4. Guardar y mostrar
    cv2.imwrite(output_path, resultado_final)
    print(f"✅ Proceso completado. Guardado en: {output_path}")
    cv2.imshow("Resultado HD", cv2.resize(resultado_final, (1200, 800))) # Resize para visualizar en pantalla
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    segmentacion_final_completa()