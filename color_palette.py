import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.cluster import KMeans
import pandas as pd
import gc

def analizar_fachada_secuencial(ruta_imagen, var_maxima=18.0):
    # 1. Preparación de datos y limpieza de memoria
    gc.collect()
    img_rgb = io.imread(ruta_imagen).astype(np.float32) / 255.0
    if img_rgb.shape[2] == 4: img_rgb = img_rgb[:, :, :3]
    h, w, _ = img_rgb.shape
    
    img_lab = color.rgb2lab(img_rgb)
    pixeles_lab = img_lab.reshape((-1, 3))
    
    # Coordenadas normalizadas para análisis de dispersión
    yy, xx = np.mgrid[0:h, 0:w]
    pixeles_xy = np.stack([xx / w, yy / h], axis=-1).reshape((-1, 2))
    
    # 2. K-Means Dinámico con límite de seguridad
    k = 2
    max_k = 7
    encontrado = False
    
    while k <= max_k and not encontrado:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(pixeles_lab)
        # Calculamos la varianza promedio de los clusters actuales
        v_actual = np.mean([np.mean(np.var(pixeles_lab[km.labels_ == i], axis=0)) for i in range(k)])
        
        if v_actual < var_maxima:
            encontrado = True
        else:
            k += 1

    # Re-asignamos k al valor final obtenido por el modelo
    labels = km.labels_.reshape(h, w)
    centros = km.cluster_centers_
    num_final_colores = len(centros) # Usamos la longitud real para evitar IndexError

    # 3. Construcción de la Visualización Secuencial
    fig = plt.figure(figsize=(14, 4 * (num_final_colores + 1)))
    gs = fig.add_gridspec(num_final_colores + 1, 3, width_ratios=[1.5, 0.5, 1])

    # --- FILA 0: CABECERA Y PALETA GLOBAL ---
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_rgb)
    ax_orig.set_title("Imagen de Referencia", loc='left', fontweight='bold')
    ax_orig.axis('off')

    ax_paleta = fig.add_subplot(gs[0, 1:])
    ax_paleta.axis('off')
    for i in range(num_final_colores):
        c_rgb = color.lab2rgb(centros[i].reshape(1,1,3)).flatten().clip(0,1)
        # Dibujamos la paleta completa como referencia
        rect = plt.Rectangle((i/num_final_colores, 0.3), 0.7/num_final_colores, 0.4, 
                             facecolor=c_rgb, transform=ax_paleta.transAxes)
        ax_paleta.add_patch(rect)
        ax_paleta.text(i/num_final_colores + 0.35/num_final_colores, 0.2, f"ID {i}", ha='center')
    ax_paleta.set_title("Paleta Global Detectada", fontweight='bold')

    # --- FILAS SIGUIENTES: DESGLOSE POR COLOR ---
    for i in range(num_final_colores):
        mask = (labels == i)
        pts_lab = pixeles_lab[km.labels_ == i]
        pts_xy = pixeles_xy[km.labels_ == i]
        
        c_lab = centros[i]
        c_rgb = color.lab2rgb(c_lab.reshape(1,1,3)).flatten().clip(0, 1)
        
        # A. Máscara Morfológica
        ax_mask = fig.add_subplot(gs[i+1, 0])
        mask_viz = np.zeros_like(img_rgb)
        mask_viz[mask] = c_rgb
        ax_mask.imshow(mask_viz)
        ax_mask.axis('off')
        ax_mask.set_title(f"Morfología Color ID {i}")

        # B. Parche CIELAB
        ax_patch = fig.add_subplot(gs[i+1, 1])
        ax_patch.axis('off')
        ax_patch.add_patch(plt.Rectangle((0.2, 0.3), 0.6, 0.4, facecolor=c_rgb))
        lab_text = f"L*: {c_lab[0]:.1f}\na*: {c_lab[1]:.1f}\nb*: {c_lab[2]:.1f}"
        ax_patch.text(0.5, 0.15, lab_text, ha='center', va='top', fontsize=9, fontweight='bold')

        # C. Métricas de Identidad
        ax_met = fig.add_subplot(gs[i+1, 2])
        ax_met.axis('off')
        abundancia = np.sum(mask) / (h * w) * 100
        varianza = np.mean(np.var(pts_lab, axis=0))
        dispersion = np.mean(np.var(pts_xy, axis=0))
        
        metricas_text = (
            f"• Cobertura: {abundancia:.2f}%\n"
            f"• Varianza: {varianza:.2f}\n"
            f"• Dispersión: {dispersion:.4f}"
        )
        ax_met.text(0, 0.5, metricas_text, ha='left', va='center', fontsize=11, linespacing=1.8)
        ax_met.set_title(f"Análisis Estadístico ID {i}")

    plt.tight_layout()
    plt.show()

# Ejecución
analizar_fachada_secuencial('entrada/isleta/PIÑEYRO_ISLETA_260616_F1937_F1941.jpg')