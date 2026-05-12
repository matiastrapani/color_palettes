import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

def segmentacion_final_completa():
    # 1. Configuración del Predictor
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="models/sam3.pt",
        half=True,
        device="mps",
        imgsz=1024 
    )
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    img_path = "entrada/piñeyro_001_s.jpg"
    
    # Cargar imagen para dimensiones originales
    img_orig = cv2.imread(img_path)
    if img_orig is not None:
        h, w, _ = img_orig.shape
    
    predictor.set_image(img_path)
    
    # 2. Inferencia semántica múltiple
    conceptos = ["building facade", "tree"]
    results = predictor(text=conceptos)
    
    if results[0].masks is not None:
        # --- PARTE 1: Guardar Imagen Superpuesta (Arcoíris) ---
        # El método .plot() usará colores distintos para 'facade' y 'tree'
        anotada = results[0].plot(
            conf=False,
            labels=True,  # Ponemos True para verificar visualmente la clase
            boxes=False,
            masks=True
        )
        cv2.imwrite("salida/superposicion_color.jpg", anotada)
        
        # --- PARTE 2: Guardar SVG con Capas/Colores ---
        nombres = results[0].names
        clases = results[0].boxes.cls.cpu().numpy()
        poligonos = results[0].masks.xy
        
        # Definimos colores fijos para el SVG para diferenciar capas
        colores_capas = {
            "building facade": "#FF0000", # Rojo
            "tree": "#00FF00"            # Verde
        }

        with open("salida/morfologia_capas.svg", "w") as f:
            f.write(f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">\n')
            f.write('  <rect width="100%" height="100%" fill="none" />\n') # Fondo transparente
            
            for i, poly in enumerate(poligonos):
                nombre_clase = nombres[int(clases[i])]
                color = colores_capas.get(nombre_clase, "#0000FF") # Azul por defecto
                
                # Convertir coordenadas a string
                puntos_str = " ".join([f"{p[0]},{p[1]}" for p in poly])
                
                # Escribimos el polígono con un ID de clase para que actúe como "capa"
                f.write(f'  <polygon id="{nombre_clase.replace(" ", "_")}_{i}" '
                        f'points="{puntos_str}" '
                        f'fill="{color}" fill-opacity="0.3" '
                        f'stroke="{color}" stroke-width="2" />\n')
            
            f.write('</svg>')

        print(f"✅ Proceso completado:")
        print(f"- Visualización: salida/superposicion_color.jpg")
        print(f"- Vectorial: salida/morfologia_capas.svg (Rojo: Fachada, Verde: Árboles)")
    else:
        print("❌ No se detectaron elementos.")

if __name__ == "__main__":
    segmentacion_final_completa()