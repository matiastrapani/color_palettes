import cv2
import numpy as np
import torch
import gc
import os
import time
import logging
from typing import List, Tuple
from ultralytics.models.sam import SAM3SemanticPredictor



logging.getLogger("ultralytics").setLevel(logging.WARNING)
'''
Tile 1: Comienza en 0 y termina en 1036.

Tile 2: Comienza en 986 (porque 0+986) y termina en 1036 + 986 = 2022.

Tile 3: Comienza en 1972 (porque 986+986) y termina en 2022 + 986 = 3008.
'''

class SegmentadorMorfologico:
    def __init__(self, model_path: str = "models/sam3.pt", device: str = "mps", imgsz: int = 1036):
        self.device = device
        self.imgsz = imgsz
        self.predictor = SAM3SemanticPredictor(overrides=dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=model_path,
            half=True,
            device=self.device,
            imgsz=self.imgsz,
            save=False
        ))

    def procesar_concepto_completo(self, img_orig: np.ndarray, concepto: str, usar_tiles: bool) -> np.ndarray:
        h, w = img_orig.shape[:2]
        mascara_concepto = np.zeros((h, w), dtype=np.uint8)
        
        # Definir coordenadas
        if not usar_tiles:
            coords = [(0, 0, h, w)]
        else:
            y_coords = list(range(0, h, self.imgsz - 12))
            x_coords = list(range(0, w, self.imgsz - 12))
            coords = [(y, x, min(y + self.imgsz, h), min(x + self.imgsz, w)) 
                      for y in y_coords for x in x_coords]
        
        total_tiles = len(coords)
        print(f"--- Iniciando concepto: '{concepto}' ({total_tiles} tiles) ---")
        
        for i, (y, x, y_end, x_end) in enumerate(coords, 1):
            # Aquí recuperamos el log detallado que querías
            tile_info = f"Tile {i}/{total_tiles}"
            print(f"  > [{tile_info}] Procesando...", end=" ", flush=True)
            
            patch = img_orig[y:y_end, x:x_end]
            
            # Medir tiempo del tile
            start_t = time.time()
            results = self.predictor(source=patch, text=[concepto])
            
            detecciones = 0
            if results and results[0].masks is not None:
                detecciones = len(results[0].masks.data)
                for mask in results[0].masks.data:
                    m = cv2.resize(mask.cpu().numpy().astype(np.uint8), (x_end - x, y_end - y))
                    mascara_concepto[y:y_end, x:x_end] = cv2.bitwise_or(mascara_concepto[y:y_end, x:x_end], m)
            
            elapsed = time.time() - start_t
            print(f"Detectados: {detecciones} | Tiempo: {elapsed:.2f}s")
            
            del results; gc.collect(); torch.mps.empty_cache()
            
        return mascara_concepto

    def generar_mascaras_y_guardar(self, img_orig: np.ndarray, sumar: List[str], restar: List[str], 
                                  folder: str, file_base: str, save_concepts: bool = False, 
                                  usar_tiles: bool = True) -> np.ndarray:
        
        h, w = img_orig.shape[:2]
        tiempo_inicio_total = time.time() # Inicio del cronómetro global
        mascara_suma = np.zeros(img_orig.shape[:2], dtype=np.uint8)
        
        # Refactorización: Función interna para procesar grupo (suma o resta)
        def procesar_grupo(conceptos, prefijo):
            mascara_grupo = np.zeros(img_orig.shape[:2], dtype=np.uint8)
            for concepto in conceptos:
                m = self.procesar_concepto_completo(img_orig, concepto, usar_tiles)
                if save_concepts:
                    path = os.path.join(folder, f"{prefijo}_{file_base}_{concepto.replace(' ', '_')}.png")
                    cv2.imwrite(path, np.where(m > 0, 255, 0).astype(np.uint8))
                mascara_grupo = cv2.bitwise_or(mascara_grupo, m)
            return mascara_grupo

        if not sumar:
            mascara_suma = np.full((h, w), 1, dtype=np.uint8)
        else:
            mascara_suma = procesar_grupo(sumar, "MASC")
            
        mascara_resta = procesar_grupo(restar, "REST")
            
        tiempo_total = time.time() - tiempo_inicio_total
        print(f"\n⏱️ Tiempo total de procesamiento: {tiempo_total:.2f} segundos.")
        
        return cv2.bitwise_and(mascara_suma, cv2.bitwise_not(mascara_resta))
            
    def aplicar_efecto_velado(self, img: np.ndarray, mascara: np.ndarray, opacidad: float = 0.85) -> np.ndarray:
        # 1. Crear fondo blanco base
        fondo_blanco = np.full_like(img, 255, dtype=np.uint8)
        
        # 2. Crear la versión velada (85% blanco sobre la original)
        # Esto hace que la imagen se vea "lavada" o grisácea
        velado = cv2.addWeighted(img, 1.0 - opacidad, fondo_blanco, opacidad, 0)
        
        # 3. Crear copia para el resultado final
        resultado = velado.copy()
        
        # 4. Debug: ¿La máscara tiene contenido?
        pixeles_detectados = np.sum(mascara > 0)
        print(f"Píxeles segmentados detectados: {pixeles_detectados}")
        
        if pixeles_detectados > 0:
            # 5. Pegar la imagen original solo donde la máscara es > 0
            resultado[mascara > 0] = img[mascara > 0]
        else:
            print("⚠️ Aviso: La máscara está vacía. Revisa los conceptos o el umbral 'conf'.")
            
        return resultado

# Uso del objeto
if __name__ == "__main__":
    segmentador = SegmentadorMorfologico()
    
    # Configuración de rutas
    dir_entrada = "entrada/isleta"
    dir_salida = "salida/isleta"
    dir_masc = "salida/isleta/masc"

    filename = "PIÑEYRO_ISLETA_260616_F1971_2022.jpg"
    file_base = os.path.splitext(filename)[0]
    file_path = os.path.join(dir_entrada, filename)
    
    dir_concepts = f"{dir_masc}/MASC_{file_base}"
    masc_path = os.path.join(dir_masc, f"MASC_{file_base}_2.png")
    result_path = os.path.join(dir_salida, f"RES_{filename}")

    img = cv2.imread(file_path)
    
    # Validación explícita para Pylance
    if img is not None:
        sumar = ["building facade", "pavement", "window", "door", "trash can", "urban furniture", "sign", "roof", "column", "architecture", "air conditioner", "wall", "molding"]
        restar = ["vehicle", "sky"]
        
    
        usar_tiles = False
        save_concepts = False
        save_result = True
        save_masc = True
        show_result = True


        if save_masc:
            os.makedirs(dir_masc, exist_ok=True)
        if save_concepts:     
            os.makedirs(dir_concepts, exist_ok=True)


        mascara = segmentador.generar_mascaras_y_guardar(img, sumar, restar, dir_concepts, file_base, save_concepts, usar_tiles)
        final = segmentador.aplicar_efecto_velado(img, mascara)
        print("✅ Proceso finalizado.")

        if save_result:
            print(f"\nResultado guardado en:\n{result_path}")
            cv2.imwrite(result_path, final)
        if save_masc:
            print(f"\nMascara general guardada en:\n{masc_path}\n")      
            mascara_guardable = (mascara > 0).astype(np.uint8) * 255
            cv2.imwrite(masc_path, mascara_guardable)


        if show_result:
            cv2.imshow("Resultado HD", cv2.resize(final, (1200, 800))) # Resize para visualizar en pantalla
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        print(f"❌ Error: No se pudo cargar la imagen en {file_path}")