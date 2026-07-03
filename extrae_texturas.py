import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def extraer_textura_photopea_con_mascara(ruta_imagen: str, ruta_mascara: str | None = None, blur_coherencia: float = 0.6, factor_fuerza: float = 1.8):
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encuentra la imagen en '{ruta_imagen}'")
        return

    print("1. Cargando imagen original HD...")
    img_pil = Image.open(ruta_imagen).convert("RGB")
    img_rgb = np.array(img_pil, dtype=np.float32) / 255.0
    alto, ancho, _ = img_rgb.shape

    # --- CARGA Y VALIDACIÓN DE LA MÁSCARA ---
    mascara_2d = None
    if ruta_mascara and os.path.exists(ruta_mascara):
        print("2. Cargando y alineando máscara geométrica...")
        masc_pil = Image.open(ruta_mascara).convert("L")
        # Aseguramos consistencia dimensional estricta usando NEAREST
        if masc_pil.size != (ancho, alto):
            masc_pil = masc_pil.resize((ancho, alto), Image.Resampling.NEAREST)
        # Convertimos a matriz booleana: True para la fachada, False para ignorar
        mascara_2d = np.array(masc_pil) > 127
    else:
        print("2. No se especificó o no se encontró máscara. Se procesa completa.")

    print("3. Filtrado de coherencia y conversión a grises...")
    img_suave = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=blur_coherencia) if blur_coherencia > 0 else img_rgb.copy()
    img_gris = cv2.cvtColor((img_suave * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img_gris_f = img_gris.astype(np.float32) / 255.0

    print("4. Calculando gradientes de textura (Sobel)...")
    sobel_x = cv2.Sobel(img_gris_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gris_f, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitud_gris = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitud_gris = np.clip(magnitud_gris * factor_fuerza, 0.0, 1.0)

    print("5. Invirtiendo a fondo blanco y aplicando máscara...")
    resultado_grises = 1.0 - magnitud_gris

    # SI HAY MÁSCARA: Forzamos a blanco puro (1.0) todo lo que esté fuera (donde es False)
    if mascara_2d is not None:
        resultado_grises[~mascara_2d] = 1.0

    print("6. Guardando resultado final...")
    resultado_uint8 = (resultado_grises * 255).astype(np.uint8)
    cv2.imwrite("resultado_textura_mascarada.png", resultado_uint8)

    print("7. Visualizando comparativa con Matplotlib...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    # Imagen original recortada visualmente por la máscara para control
    img_original_oculta = img_rgb.copy()
    if mascara_2d is not None:
        img_original_oculta[~mascara_2d] = 1.0 # Fondo blanco en la original para comparar bien
        
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Zona Activa Original")
    axes[0].axis("off")
    
    axes[1].imshow(resultado_grises, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"2. Textura Extraída (Filtro Máscara)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir_entrada = "entrada/isleta"
    dir_mascara = "salida/isleta/masc"
    filename = "PIÑEYRO_ISLETA_260616_F1971_2022.jpg"
    masc_filename = "MASC_PIÑEYRO_ISLETA_260616_F1971_2022_.png"
    RUTA_TEST = os.path.join(dir_entrada, filename)
    RUTA_MASC = os.path.join(dir_mascara, masc_filename)

    
    extraer_textura_photopea_con_mascara(
        RUTA_TEST, 
        ruta_mascara=RUTA_MASC if os.path.exists(RUTA_MASC) else None,
        blur_coherencia=0.6, 
        factor_fuerza=1.8
    )