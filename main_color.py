import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from PIL import Image
from skimage import color
import sys
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


# --- CONFIGURACIÓN DE VISUALIZACIÓN ---
CONFIG = {
    "ver_3d": True,

    "ver_datos_brutos": True,   # Puntos CIELAB originales
    "ver_anclas": True,        # Toggle para anclas
    "ver_median_cut": False,    # Toggle para los 32/500 puntos de paleta
    "ver_dbscan": False,

    "ver_histogramas_2d": False,
    "ver_histogramas_1d": False,

    "ver_gmm": False,
    "ver_paleta_mediana": True,
    "n_paleta_mediana": 500,

    "ver_elipsoides_fijas": True,
    "tol_L": 20.0,   # Rango completo ΔL
    "tol_C": 20.0,   # Rango completo ΔC
    "tol_T": 10.0    # NUEVO: Rango completo ΔT (Tangencial) en unidades CIELAB
}



def agrupar_por_tolerancia_fija(centroides, tol_L, tol_C, tol_T):
    """
    Agrupa centroides usando un sistema cartesiano local (Radial/Tangencial).
    Garantiza que ningún volumen de elipsoide se solape con otro.
    """
    indices_restantes = list(range(len(centroides)))
    centros_finales = []
    labels = np.full(len(centroides), -1)
    cluster_id = 0
    
    while len(indices_restantes) > 0:
        mejores_vecinos = []
        mejor_centro_idx = -1
        
        for idx in indices_restantes:
            L0, a0, b0 = centroides[idx]
            C0 = np.sqrt(a0**2 + b0**2)
            
            # Vectores unitarios del sistema local radial/tangencial
            if C0 > 1e-5:
                uR_a, uR_b = a0 / C0, b0 / C0
                uT_a, uT_b = -b0 / C0, a0 / C0
            else:
                uR_a, uR_b = 1.0, 0.0
                uT_a, uT_b = 0.0, 1.0
            
            # --- FILTRO DE NO-SOLAPAMIENTO GEOMÉTRICO ---
            colision = False
            for centro_g in centros_finales:
                L_g, a_g, b_g = centro_g
                C_g = np.sqrt(a_g**2 + b_g**2)
                if C_g > 1e-5:
                    uR_ag, uR_bg = a_g / C_g, b_g / C_g
                    uT_ag, uT_bg = -b_g / C_g, a_g / C_g
                else:
                    uR_ag, uR_bg = 1.0, 0.0
                    uT_ag, uT_bg = 0.0, 1.0
                
                # Distancia normalizada desde la perspectiva del centro ya guardado
                dL_g = (L0 - L_g) / (tol_L / 2)
                dC_g = ((a0 - a_g) * uR_ag + (b0 - b_g) * uR_bg) / (tol_C / 2)
                dT_g = ((a0 - a_g) * uT_ag + (b0 - b_g) * uT_bg) / (tol_T / 2)
                
                # Si la distancia es menor a 2.0 (radios sumados), los volúmenes colisionarían
                if (dL_g**2 + dC_g**2 + dT_g**2) < 4.0: 
                    colision = True
                    break
            if colision:
                continue # No puede ser centro porque su elipsoide se solaparía
            
            # Buscar vecinos para el candidato válido
            vecinos_candidatos = []
            for cand_idx in indices_restantes:
                L1, a1, b1 = centroides[cand_idx]
                
                dL = (L1 - L0) / (tol_L / 2)
                da, db = a1 - a0, b1 - b0
                dC = (da * uR_a + db * uR_b) / (tol_C / 2)
                dT = (da * uT_a + db * uT_b) / (tol_T / 2)
                
                if (dL**2 + dC**2 + dT**2) <= 1.0:
                    vecinos_candidatos.append(cand_idx)
                    
            if len(vecinos_candidatos) > len(mejores_vecinos):
                mejores_vecinos = vecinos_candidatos
                mejor_centro_idx = idx
                
        if mejor_centro_idx == -1 or len(mejores_vecinos) == 0:
            break
            
        labels[mejores_vecinos] = cluster_id
        centros_finales.append(centroides[mejor_centro_idx])
        cluster_id += 1
        
        for v in mejores_vecinos:
            indices_restantes.remove(v)
            
    print(f"Familias rígidas (sin solapamiento): {cluster_id}")
    return labels, np.array(centros_finales)

def dibujar_elipsoide_radial(ax, centro_lab, tol_L, tol_C, tol_T, color):
    """
    Dibuja la elipse usando el radio tangencial lineal absoluto (tol_T / 2).
    """
    L0, a0, b0 = centro_lab
    C0 = np.sqrt(a0**2 + b0**2)
    h0 = np.arctan2(b0, a0)
    
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_esf = np.cos(u) * np.sin(v) # Eje Tangencial
    y_esf = np.sin(u) * np.sin(v) # Eje Radial
    z_esf = np.cos(v)             # Eje L
    
    r_L = tol_L / 2
    r_C = tol_C / 2
    r_T = tol_T / 2 # Tamaño constante e independiente del croma central
    
    # Rotación e interpolación limpia en el espacio ab
    a_rot = x_esf * r_T * (-np.sin(h0)) + y_esf * r_C * np.cos(h0) + a0
    b_rot = x_esf * r_T * np.cos(h0) + y_esf * r_C * np.sin(h0) + b0
    L_rot = z_esf * r_L + L0
    
    ax.plot_surface(a_rot, b_rot, L_rot, color=color, alpha=0.20, edgecolor='none')

#Gaussian Mixture Models (GMM)
def encontrar_elipsoides_optimas(data, max_components=20):
    bics = []
    models = []
    
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(data)
        bics.append(gmm.bic(data))
        models.append(gmm)
        
    # El modelo con el BIC más bajo es el que mejor describe la imagen 
    # sin ser innecesariamente complejo
    best_idx = np.argmin(bics)
    return models[best_idx]

def agrupar_colores_gmm_automatico(lab_pixels, max_components=20):
    # 1. Preparar datos (CIELCh -> [L, C, h_sin, h_cos])
    L = lab_pixels[:, 0]
    C = np.sqrt(lab_pixels[:, 1]**2 + lab_pixels[:, 2]**2)
    h_rad = np.radians(np.degrees(np.arctan2(lab_pixels[:, 2], lab_pixels[:, 1])) % 360)
    
    data = np.stack([L, C, np.sin(h_rad), np.cos(h_rad)], axis=1)
    
    # 2. Búsqueda de mejor BIC
    bics = []
    models = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(data)
        bics.append(gmm.bic(data))
        models.append(gmm)
        
    best_gmm = models[np.argmin(bics)]
    labels = best_gmm.predict(data)
    
    print(f"GMM seleccionó automáticamente {len(best_gmm.means_)} componentes (familias).")
    
    # Retornamos medias (centroides) y covarianzas (forma de la elipsoide)
    return labels, best_gmm.means_, best_gmm.covariances_

def dibujar_elipsoide(ax, centro, cov, color, n_std=2.0):
    """
    Dibuja una elipsoide 3D basada en media y covarianza.
    n_std: Número de desviaciones estándar para el radio de la elipsoide.
    """
    # Descomposición de valores y vectores propios
    vals, vecs = np.linalg.eigh(cov)
    
    # Ordenar y escalar
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 20)
    x, y, z = np.outer(np.cos(theta), np.sin(phi)), np.outer(np.sin(theta), np.sin(phi)), np.outer(np.ones(theta.size), np.cos(phi))
    
    # Aplicar escala y rotación
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(vecs, [x[i, j]*np.sqrt(vals[0])*n_std, 
                                                       y[i, j]*np.sqrt(vals[1])*n_std, 
                                                       z[i, j]*np.sqrt(vals[2])*n_std]) + centro

    ax.plot_surface(x, y, z, color=color, alpha=0.3)

# median cut
def obtener_paleta_mediana(lab_pixels, n_colors=500):
    """
    Reduce los colores a un máximo de n_colors mediante Median Cut.
    """
    # Empezamos con una sola caja que contiene todos los píxeles
    cajas = [lab_pixels]
    
    # Dividimos cajas hasta alcanzar n_colors
    while len(cajas) < n_colors:
        # Buscamos la caja que tenga mayor rango (más dispersión)
        # Esto nos asegura dividir primero donde hay más variedad de color
        rangos = [np.ptp(caja, axis=0) for caja in cajas]
        max_rangos = [np.max(r) for r in rangos]
        idx_caja_a_dividir = np.argmax(max_rangos)
        
        caja_a_dividir = cajas.pop(idx_caja_a_dividir)
        
        if len(caja_a_dividir) <= 1:
            # Si no se puede dividir más, la devolvemos a la lista
            cajas.append(caja_a_dividir)
            break
            
        # Elegimos el eje de mayor dispersión dentro de esa caja específica
        axis = np.argmax(rangos[idx_caja_a_dividir])
        
        # Partimos por la mediana
        caja_ordenada = caja_a_dividir[caja_a_dividir[:, axis].argsort()]
        mid = len(caja_ordenada) // 2
        
        cajas.append(caja_ordenada[:mid])
        cajas.append(caja_ordenada[mid:])
        
    # El color representativo de cada caja es su promedio
    paleta = [np.mean(caja, axis=0) for caja in cajas]
    return np.array(paleta)

#Anclas
def detectar_anclas(lab_pixels, bins=30, umbral_porcentaje=0.001, sigma=1.0):
    """
    Detecta colores ancla basados en la densidad del espacio CIELAB.
    
    :param lab_pixels: Array de (N, 3) con valores L, a, b.
    :param bins: Resolución del histograma 3D (30x30x30).
    :param umbral_porcentaje: 0.005 equivale al 0.5%.
    :param sigma: Fuerza del suavizado (aumentar para agrupar más colores cercanos).
    :return: Lista de anclas (coordenadas L, a, b).
    """
    # 1. Definir los límites del espacio LAB
    # L: 0-100, a: -100 a 100, b: -100 a 100
    rango = ((0, 100), (-100, 100), (-100, 100))
    
    # 2. Crear histograma 3D
    hist, edges = np.histogramdd(lab_pixels, bins=bins, range=rango)
    
    # 3. Suavizado gaussiano para unir áreas de color dispersas (esencial para precisión)
    if sigma > 0:
        hist = gaussian_filter(hist, sigma=sigma)
    
    # 4. Calcular umbral absoluto (número de píxeles)
    total_pixeles = len(lab_pixels)
    umbral_absoluto = total_pixeles * umbral_porcentaje
    
    # 5. Encontrar picos (bins que superan el umbral)
    anclas = []
    
    # Buscamos índices donde la densidad es mayor al umbral
    indices_picos = np.argwhere(hist > umbral_absoluto)
    
    for idx in indices_picos:
        # Recuperar el valor central del bin (L, a, b)
        l_idx, a_idx, b_idx = idx
        
        L = (edges[0][l_idx] + edges[0][l_idx+1]) / 2
        a = (edges[1][a_idx] + edges[1][a_idx+1]) / 2
        b = (edges[2][b_idx] + edges[2][b_idx+1]) / 2
        
        anclas.append([L, a, b])

    print(f'anclas: {len(anclas)}')
    return np.array(anclas)

#DBSCAN y HDBSCAN
def agrupar_colores_dbscan(lab_pixels, pesos={'L': 1.00, 'C': 1.0, 'h': 2.0}, eps=2.0):
    """
    Agrupa colores usando DBSCAN en CIELCh corrigiendo la discontinuidad del Tono.
    """
    # 1. Convertir a CIELCh
    L = lab_pixels[:, 0]
    C = np.sqrt(lab_pixels[:, 1]**2 + lab_pixels[:, 2]**2)
    h = np.degrees(np.arctan2(lab_pixels[:, 2], lab_pixels[:, 1])) % 360
    
    # 2. Convertir Tono a coordenadas polares (sin/cos)
    # Esto elimina la discontinuidad 0/360
    h_rad = np.radians(h)
    h_sin = np.sin(h_rad)
    h_cos = np.cos(h_rad)
    
    # 3. Normalizar y aplicar pesos
    # Ahora usamos 4 dimensiones para los datos: [L, C, h_sin, h_cos]
    # Aplicamos el peso 'h' a ambos componentes del seno y coseno
    data = np.stack([
        L * pesos['L'], 
        C * pesos['C'], 
        h_sin * pesos['h'], 
        h_cos * pesos['h']
    ], axis=1)
    
    # 4. Aplicar DBSCAN
    db = DBSCAN(eps=eps, min_samples=3).fit(data)
    
    # 5. Retornar labels
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Familias de color detectadas: {n_clusters}")
    
    return db.labels_, n_clusters

def agrupar_colores_hdbscan(lab_pixels, pesos={'L': 0.5, 'C': 1.0, 'h': 2.0}, min_cluster_size=4):
    # 1. Convertir a CIELCh
    L = lab_pixels[:, 0]
    C = np.sqrt(lab_pixels[:, 1]**2 + lab_pixels[:, 2]**2)
    h = np.degrees(np.arctan2(lab_pixels[:, 2], lab_pixels[:, 1])) % 360
    
    # 2. Conversión a polares para evitar el salto 0/360
    h_rad = np.radians(h)
    
    # 3. Normalización y pesos
    # Usamos 4 dimensiones: [L, C, h_sin, h_cos]
    data = np.stack([
        L * pesos['L'], 
        C * pesos['C'], 
        np.sin(h_rad) * pesos['h'], 
        np.cos(h_rad) * pesos['h']
    ], axis=1)
    
    # 4. Aplicar HDBSCAN
    # min_cluster_size controla qué tan pequeña puede ser una familia
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(data)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Familias de color detectadas con HDBSCAN: {n_clusters}")
    
    return labels, n_clusters

def on_pick(event):
    # Verificamos si el objeto clickeado contiene la información del centroide
    if hasattr(event.artist, 'centro_info'):
        cluster_id, centro_lab, rgb_color = event.artist.centro_info
        
        # Crear una nueva ventana emergente independiente
        fig_popup = plt.figure(figsize=(4, 4))
        ax_popup = fig_popup.add_subplot(111)
        
        # Mostrar el color sólido del centroide
        ax_popup.imshow([[np.clip(rgb_color, 0, 1)]])
        ax_popup.axis('off')
        
        # Agregar título e información LAB
        fig_popup.suptitle(f"Familia #{cluster_id}", fontsize=13, fontweight='bold')
        texto_lab = f"L*: {centro_lab[0]:.2f}\na*: {centro_lab[1]:.2f}\nb*: {centro_lab[2]:.2f}"
        ax_popup.text(0.5, -0.15, texto_lab, transform=ax_popup.transAxes, 
                     ha='center', va='top', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        fig_popup.tight_layout()
        fig_popup.show() # Importante: .show() abre la ventana sin bloquear el hilo principal

def plot_3d(ax, lab_pixels, rgb_plot, anclas, centroides=None, labels=None, gmm_means=None, gmm_covs=None, popup=None, img_original=None, h_mues=0, w_mues=0, lab_total=None, rgb_total_real=None):
    if not CONFIG["ver_3d"]: return
    if CONFIG["ver_datos_brutos"]:
        ax.scatter(lab_pixels[:, 1], lab_pixels[:, 2], lab_pixels[:, 0], 
                   c=rgb_plot, s=5, alpha=0.01, label='Datos Brutos')

    if CONFIG["ver_anclas"]:
        if len(anclas) > 0:
            ax.scatter(anclas[:, 1], anclas[:, 2], anclas[:, 0], c='yellow', s=50, marker='.', edgecolors='black', label='Anclas', zorder=10)

    
    if CONFIG["ver_dbscan"] and centroides is not None and labels is not None:
        is_noise = (labels == -1)
        # Clusters
        ax.scatter(centroides[~is_noise, 1], centroides[~is_noise, 2], centroides[~is_noise, 0], 
                   c=labels[~is_noise], cmap='tab20', s=150, marker='o', label='Familias DBSCAN', zorder=50)
        # Ruido
        if False:
            ax.scatter(centroides[is_noise, 1], centroides[is_noise, 2], centroides[is_noise, 0], 
                    c='red', s=30, marker='x', label='Ruido DBSCAN', alpha=0.5)

    # 4. Capa Median Cut (Solo se muestra si dbscan no está activo o si lo quieres ver de fondo)
    if CONFIG["ver_median_cut"] and centroides is not None:
        ax.scatter(centroides[:, 1], centroides[:, 2], centroides[:, 0], 
                   c='red', s=40, marker='.', label='Median Cut', alpha=0.6)


    if CONFIG.get("ver_gmm") and gmm_means is not None and gmm_covs is not None:
        for i in range(len(gmm_means)):
            # 1. Extraer componentes (L, C, h_sin, h_cos)
            l, c, hs, hc = gmm_means[i]
            
            # 2. Convertir de nuevo a CIELAB (L, a, b)
            h_rad = np.arctan2(hs, hc)
            a = c * np.cos(h_rad)
            b = c * np.sin(h_rad)
            lab_color = np.array([l, a, b])
            
            # 3. Convertir a RGB para graficar
            c_promedio = color.lab2rgb(lab_color.reshape(1, 1, 3)).flatten()
            
            # 4. Dibujar usando los índices originales [L, a, b]
            # Nota: Asegúrate de mapear según tu gráfico: 
            # ax.scatter(a, b, L) -> means[i, 1], means[i, 2], means[i, 0]
            ax.scatter(a, b, l, c=[c_promedio], s=100, marker='o', edgecolors='black', zorder=100)
            
            # 5. Dibujar la elipsoide (necesitas la matriz de covarianza en LAB)
            # Como GMM está en espacio (L, C, h_sin, h_cos), para simplificar 
            # dibujaremos la elipsoide proyectada en el plano a, b, L
            cov_sub = gmm_covs[i][np.ix_([0, 1, 2], [0, 1, 2])] # Simplificación
            dibujar_elipsoide(ax, np.array([a, b, l]), cov_sub, color=c_promedio)

    # Reemplaza el bloque "if CONFIG.get('ver_gmm')..." por este:
    if CONFIG.get("ver_elipsoides_fijas") and gmm_means is not None:
        xs = gmm_means[:, 1]
        ys = gmm_means[:, 2]
        zs = gmm_means[:, 0]
        colores = [np.clip(color.lab2rgb(c.reshape(1, 1, 3)).flatten(), 0, 1) for c in gmm_means]
        
        sc_centros = ax.scatter(xs, ys, zs, c=colores, s=120, marker='o', edgecolors='black', zorder=100)
        sc_centros.centros_lab = gmm_means
        sc_centros.colores_rgb = colores
        
        for i, centro in enumerate(gmm_means):
            dibujar_elipsoide_radial(ax, centro, CONFIG["tol_L"], CONFIG["tol_C"], CONFIG["tol_T"], color=colores[i])
            
        if popup is not None:
            conectar_interaccion_cielab(ax, sc_centros, popup, img_original, lab_pixels, rgb_total_real, anclas, h_mues, w_mues, lab_total)

    ax.set_xlabel('a* (Verde - Rojo +)')
    ax.set_xlim(-100, 100)
    
    ax.set_ylabel('b* (Azul - Amarillo +)')
    ax.set_ylim(-100, 100)
    
    ax.set_zlabel('L* (Luminosidad)')
    ax.set_zlim(0, 100)

class VentanaPopupColor:
    """Manejador persistente para mostrar el análisis detallado de la familia en macOS."""
    def __init__(self):
        self.fig = None
        self.ax_imagen_origen = None
        self.ax_segmentada = None
        self.ax_color_centro = None
        self.ax_color_promedio = None
        self.ax_color_pico = None
        self.activa = False

    def inicializar_al_principio(self):
        """Inicializa la ventana con una grilla de 2x3 al inicio."""
        self.fig = plt.figure(figsize=(14, 8))
        
        self.ax_imagen_origen = self.fig.add_subplot(2, 3, 1)
        self.ax_segmentada = self.fig.add_subplot(2, 3, 2)
        
        self.ax_color_centro = self.fig.add_subplot(2, 3, 4)
        self.ax_color_promedio = self.fig.add_subplot(2, 3, 5)
        self.ax_color_pico = self.fig.add_subplot(2, 3, 6)
        
        for ax in [self.ax_imagen_origen, self.ax_segmentada, self.ax_color_centro, self.ax_color_promedio, self.ax_color_pico]:
            ax.axis('off')
            
        self.ax_imagen_origen.text(0.5, 0.5, "Seleccione una familia\nen el gráfico 3D", 
                                   ha='center', va='center', color='gray', fontsize=12)
        
        self.fig.tight_layout()
        self.activa = True
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, event):
        self.activa = False

    def actualizar(self, idx, centro_lab, rgb_centro, img_original, lab_pixels, rgb_completo, indices_pixeles_familia, anclas, h_mues, w_mues):
        if not self.activa or self.fig is None:
            return
            
        if (self.ax_imagen_origen is None or self.ax_segmentada is None or 
            self.ax_color_centro is None or self.ax_color_promedio is None or self.ax_color_pico is None):
            return

        # --- 1. FILTRADO DE PÍXELES EXCLUSIVOS DE LA FAMILIA (RESOLUCIÓN ORIGINAL) ---
        pixeles_f_lab = lab_pixels[indices_pixeles_familia]

        # --- 2. CÁLCULO DE COLORES Y SUS VALORES LAB ---
        lab_centro_txt = f"L*: {centro_lab[0]:.1f}\na*: {centro_lab[1]:.1f}\nb*: {centro_lab[2]:.1f}"

        if len(pixeles_f_lab) > 0:
            promedio_lab = np.mean(pixeles_f_lab, axis=0)
            rgb_promedio = np.clip(color.lab2rgb(promedio_lab.reshape(1, 1, 3)).flatten(), 0, 1)
            lab_promedio_txt = f"L*: {promedio_lab[0]:.1f}\na*: {promedio_lab[1]:.1f}\nb*: {promedio_lab[2]:.1f}"
        else:
            rgb_promedio = rgb_centro
            lab_promedio_txt = lab_centro_txt

        if len(anclas) > 0 and len(pixeles_f_lab) > 0:
            distancias_anclas = np.linalg.norm(anclas - centro_lab, axis=1)
            idx_ancla_pico = np.argmin(distancias_anclas)
            lab_pico = anclas[idx_ancla_pico]
            rgb_pico = np.clip(color.lab2rgb(lab_pico.reshape(1, 1, 3)).flatten(), 0, 1)
            lab_pico_txt = f"L*: {lab_pico[0]:.1f}\na*: {lab_pico[1]:.1f}\nb*: {lab_pico[2]:.1f}"
        else:
            rgb_pico = rgb_centro
            lab_pico_txt = lab_centro_txt

        # --- 3. RECONSTRUCCIÓN GEOMÉTRICA A RESOLUCIÓN ORIGINAL ---
        img_np = np.array(img_original)
        
        # Inicializar el lienzo con el alto y ancho de la resolución real (h_mues y w_mues)
        mancha_perfecta = np.ones((h_mues, w_mues, 3), dtype=np.uint8) * 255
        
        # Crear la máscara plana del tamaño total de píxeles reales
        mascara_plana = np.zeros(h_mues * w_mues, dtype=bool)
        mascara_plana[indices_pixeles_familia] = True
        
        # Redimensionar la máscara a la matriz espacial gigante
        mascara_2d = mascara_plana.reshape(h_mues, w_mues)
        
        # Extraer los colores sRGB originales (0-255) correspondientes a la elipsoide
        pixeles_originales_255 = (rgb_completo[indices_pixeles_familia] * 255).astype(np.uint8)
        
        # Mapear los píxeles en su posición espacial exacta sin deformaciones ni desajustes
        mancha_perfecta[mascara_2d] = pixeles_originales_255

        # --- 4. RENDERIZADO DE PANELES ---
        self.ax_imagen_origen.clear()
        self.ax_imagen_origen.imshow(img_np)
        self.ax_imagen_origen.set_title("Imagen Original", fontsize=11, fontweight='bold')
        self.ax_imagen_origen.axis('off')

        self.ax_segmentada.clear()
        self.ax_segmentada.imshow(mancha_perfecta)
        self.ax_segmentada.set_title(f"Mancha de la Familia #{idx}\n(Resolución Real s/ Blanco)", fontsize=11, fontweight='bold')
        self.ax_segmentada.axis('off')

        # Panel Centroide
        self.ax_color_centro.clear()
        self.ax_color_centro.imshow([[np.clip(rgb_centro, 0, 1)]])
        self.ax_color_centro.axis('off')
        self.ax_color_centro.set_title("1. Centroide Geométrico\n(Media Teórica)", fontsize=10)
        self.ax_color_centro.text(0.5, -0.15, lab_centro_txt, transform=self.ax_color_centro.transAxes, 
                                  ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Panel Promedio
        self.ax_color_promedio.clear()
        self.ax_color_promedio.imshow([[np.clip(rgb_promedio, 0, 1)]])
        self.ax_color_promedio.axis('off')
        self.ax_color_promedio.set_title("2. Promedio Real\nde la Familia", fontsize=10)
        self.ax_color_promedio.text(0.5, -0.15, lab_promedio_txt, transform=self.ax_color_promedio.transAxes, 
                                    ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Panel Pico
        self.ax_color_pico.clear()
        self.ax_color_pico.imshow([[np.clip(rgb_pico, 0, 1)]])
        self.ax_color_pico.axis('off')
        self.ax_color_pico.set_title("3. Pico de Densidad\n(Ancla de la Familia)", fontsize=10)
        self.ax_color_pico.text(0.5, -0.15, lab_pico_txt, transform=self.ax_color_pico.transAxes, 
                                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        self.fig.suptitle(f"ANÁLISIS DE ATRIBUTOS: FAMILIA DE COLOR #{idx}", fontsize=14, fontweight='bold')
        
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.show()

def conectar_interaccion_cielab(ax, sc_centros, popup, img_original, lab_pixels, rgb_completo, anclas, h_mues, w_mues, lab_total):
    fig = ax.get_figure()
    
    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"),
                        zorder=200)
    annot.set_visible(False)
    
    def on_hover(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            cont, ind = sc_centros.contains(event)
            if cont:
                idx = ind["ind"][0]
                centro_lab = sc_centros.centros_lab[idx]
                annot.set_text(f"Familia #{idx}\nL*: {centro_lab[0]:.1f}\na*: {centro_lab[1]:.1f}\nb*: {centro_lab[2]:.1f}")
                annot.xy = (event.xdata, event.ydata)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax and event.button == 1:
            if popup.activa:
                cont, ind = sc_centros.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    centro_lab = sc_centros.centros_lab[idx]
                    rgb_color = sc_centros.colores_rgb[idx]
                    
                    # --- CLUSTERING DINÁMICO POR TOLERANCIAS SOBRE EL TOTAL REAL ---
                    L0, a0, b0 = centro_lab
                    C0 = np.sqrt(a0**2 + b0**2)
                    
                    if C0 > 1e-5:
                        uR_a, uR_b = a0 / C0, b0 / C0
                        uT_a, uT_b = -b0 / C0, a0 / C0
                    else:
                        uR_a, uR_b = 1.0, 0.0
                        uT_a, uT_b = 0.0, 1.0
                    
                    # Usamos lab_total para evaluar cada píxel real de la imagen original
                    dL = (lab_total[:, 0] - L0) / (CONFIG["tol_L"] / 2)
                    da = lab_total[:, 1] - a0
                    db = lab_total[:, 2] - b0
                    dC = (da * uR_a + db * uR_b) / (CONFIG["tol_C"] / 2)
                    dT = (da * uT_a + db * uT_b) / (CONFIG["tol_T"] / 2)
                    
                    # Índices exactos a resolución original que caen dentro de esta familia
                    indices_pixeles_familia = np.where((dL**2 + dC**2 + dT**2) <= 1.0)[0]
                    
                    # DATO CONCRETO SOLICITADO: Cantidad exacta sobre el total
                    cantidad_pixeles_total = len(indices_pixeles_familia)
                    porcentaje = (cantidad_pixeles_total / len(lab_total)) * 100
                    print(f"\n[ANÁLISIS TOTAL] Familia #{idx}:")
                    print(f" -> Cantidad de píxeles reales: {cantidad_pixeles_total}")
                    print(f" -> Cobertura de la imagen: {porcentaje:.2f}%")
                    
                    # Actualizar ventana emergente con datos y dimensiones 100% reales
                    popup.actualizar(idx, centro_lab, rgb_color, img_original, 
                                     lab_total, rgb_completo, indices_pixeles_familia, anclas, h_mues, w_mues)

    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)

def plot_histogramas(lab_pixels, anclas, centroides):
    # Aquí centralizas todos los plots 2D y 1D
    if CONFIG["ver_histogramas_2d"]:
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        
        # Rango fijo para que el eje sea siempre constante
        rango_fijo = (-100, 100, -100, 100)
        
        # Histograma 2D del plano a-b
        # Usamos bins=50 y rango fijo para consistencia
        hist_2d, xedges, yedges = np.histogram2d(
            lab_pixels[:, 1], 
            lab_pixels[:, 2], 
            bins=50, 
            range=[[-100, 100], [-100, 100]]
        )
        
        # Visualización con vmin/vmax para comparar densidades entre ejecuciones
        im = ax2.imshow(
            hist_2d.T, 
            origin='lower', 
            extent=rango_fijo, 
            cmap='viridis', 
            vmin=0, 
            vmax=500, # Ajusta este valor según la densidad real de tus imágenes
            aspect='auto'
        )
        
        # Dibujar las anclas originales sobre el histograma
        if len(anclas) > 0:
            ax2.scatter(
                anclas[:, 1], 
                anclas[:, 2], 
                color='red', 
                marker='x', 
                s=50, 
                label='Anclas'
            )
            
        ax2.set_title("Densidad plano a-b (Comparativo)")
        ax2.set_xlabel('a*')
        ax2.set_ylabel('b*')
        ax2.set_xlim(-100, 100)
        ax2.set_ylim(-100, 100)
        
        plt.colorbar(im, label='Densidad de píxeles')
        plt.legend()
        plt.tight_layout()
    
    if CONFIG["ver_histogramas_1d"]:
        L = lab_pixels[:, 0]
        C = np.sqrt(lab_pixels[:, 1]**2 + lab_pixels[:, 2]**2)
        h = np.degrees(np.arctan2(lab_pixels[:, 2], lab_pixels[:, 1])) % 360

        # 2. Configuración de figura
        fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histograma de Tono (Hue)
        axes[0].hist(h, bins=60, range=(0, 360), color='purple', alpha=0.7)
        axes[0].set_title('Distribución de Tono (Hue)')
        axes[0].set_xlabel('Grados (0-360°)')
        axes[0].set_xlim(0, 360)
        
        # Histograma de Croma (Saturación)
        axes[1].hist(C, bins=30, range=(0, 100), color='orange', alpha=0.7)
        axes[1].set_title('Distribución de Croma')
        axes[1].set_xlabel('Intensidad (0-100)')
        axes[1].set_xlim(0, 100)
        
        # Histograma de Luminosidad
        axes[2].hist(L, bins=30, range=(0, 100), color='gray', alpha=0.7)
        axes[2].set_title('Distribución de Luminosidad')
        axes[2].set_xlabel('Luminosidad (0-100)')
        axes[2].set_xlim(0, 100)
        
        plt.tight_layout()

def main(ruta_imagen):
    # 1. Carga y pre-procesamiento

    img = Image.open(ruta_imagen).convert('RGB')
    img_np = np.array(img)
    
    # Dimensiones reales de la imagen gigante
    h_orig, w_orig = img_np.shape[:2]
    
    # CONVERSIÓN TOTAL: Todo el universo de la imagen a CIELAB con máxima precisión
    rgb_total = img_np.reshape(-1, 3) / 255.0
    lab_total = color.rgb2lab(rgb_total)
    
    img = Image.open(ruta_imagen).convert('RGB')
    img_np = np.array(img)
    
    # Dimensiones reales de la imagen gigante
    h_orig, w_orig = img_np.shape[:2]
    
    # CONVERSIÓN TOTAL: Universo completo en CIELAB para máxima precisión
    rgb_total = img_np.reshape(-1, 3) / 255.0
    lab_total = color.rgb2lab(rgb_total)
    
    # Submuestreo exclusivo para que Matplotlib y el 3D no colapsen
    lab_pixels = lab_total[::50]
    rgb_plot = rgb_total[::50]
    
    # 2. Análisis espacial
    anclas = detectar_anclas(lab_pixels, bins=10)
    centroides = obtener_paleta_mediana(lab_pixels, n_colors=CONFIG["n_paleta_mediana"]) if CONFIG["ver_paleta_mediana"] else None
    
    # Parámetros espaciales pasados a la función (los nombres se mantienen por compatibilidad)
    h_mues, w_mues = h_orig, w_orig
    rgb_completo = rgb_total


    labels = None
    centros_fijos = None
    if centroides is not None and CONFIG.get("ver_elipsoides_fijas"):
        # LLAMADA CORRECTA AL NUEVO MÉTODO GEOMÉTRICO
        labels, centros_fijos = agrupar_por_tolerancia_fija(
            centroides, CONFIG["tol_L"], CONFIG["tol_C"], CONFIG["tol_T"]
        )
        
    # 3. Visualización
    if CONFIG["ver_3d"]:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # NUEVO: Instanciamos e inicializamos la ventana aquí antes de los plots
        popup = VentanaPopupColor()
        popup.inicializar_al_principio()
        
        # Enviamos la referencia armada al plot_3d
        plot_3d(ax, lab_pixels, rgb_total[::50], anclas, centroides, labels, 
                gmm_means=centros_fijos, gmm_covs=None, popup=popup, img_original=img,
                h_mues=h_orig, w_mues=w_orig, lab_total=lab_total, rgb_total_real=rgb_total)
                
        ax.set_title(f'Espacio CIELAB (Elipsoides Rígidas Orientadas): {ruta_imagen}')
        fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Llamada a los otros paneles
    plot_histogramas(lab_pixels, anclas, centroides)
    
    plt.show()




if __name__ == "__main__":
    # Cambia 'tu_imagen.jpg' por la ruta de tu archivo
    import os
    dir_salida = "salida/isleta"

    filename = "RES_PIÑEYRO_ISLETA_260616_F1971_2022.jpg"

    img_path = os.path.join(dir_salida, filename)


    if os.path.exists(img_path):
        main(img_path)
    else:
        print(f"Error: No se encontró la imagen en {img_path}")
    
