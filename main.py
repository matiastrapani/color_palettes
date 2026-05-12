import os
import torch
from ultralytics import SAM

def segmentacion_color_total():
    # 1. Cargar SAM 2 Large (Hiera)
    # Si no lo tenés en la carpeta, se descargará automáticamente
    model = SAM("./models/sam2.1_l.pt") 
    device = "mps"
    
    img_path = "entrada/piñeyro_001_s.jpg"
    output_dir = "salida/visualizacion"
    os.makedirs(output_dir, exist_ok=True)

    print("🎨 Generando mapa de colores morfológico...")

    # 2. Inferencia
    # Usamos una grilla interna para activar la detección de toda la fachada
    results = model.predict(
        source=img_path,
        device=device,
        imgsz=1024,
        conf=0.25,
        save=False # No usamos el save genérico para tener más control
    )

    # 3. Renderizado de la imagen combinada
    # .plot() genera un array de numpy con la imagen original + máscaras
    # labels=False y boxes=False para que solo se vea la morfología y el color
    anotada = results[0].plot(
        labels=False, 
        boxes=False, 
        conf=False,
        line_width=1
    )

    # 4. Guardar resultado final
    output_path = os.path.join(output_dir, "fachada_segmentada_total.jpg")
    
    # Importante: results.plot() devuelve BGR, ideal para OpenCV
    import cv2
    cv2.imwrite(output_path, anotada)

    print(f"✅ ¡Listo! Revisá: {output_path}")

    cantidad = len(results[0].masks.xy) if results[0].masks is not None else 0
    print(f"Fachadas encontradas: {cantidad}")

if __name__ == "__main__":
    segmentacion_color_total()