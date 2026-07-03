import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import label, binary_erosion, binary_dilation
from skimage.measure import perimeter
from dataclasses import dataclass

@dataclass(frozen=True)
class ReporteMorfologicoColor:
    cobertura_inicial: float
    cobertura_estabilizada: float
    retencion_masa: float
    islas_iniciales: int
    islas_estabilizadas: int
    persistencia_trama: float
    pasos_optimos: int              
    grilla_densidad: np.ndarray     # Porcentaje de cobertura de fondo para el mapa de calor
    grilla_categorias: np.ndarray   # Matriz KxKx3 [Acentos, Texturas, Ruidos]

class AnalizadorMorfologicoEspacial:
    def __init__(self, divisiones_grilla: int = 8, max_pasos: int = 6) -> None:
        self.K = divisiones_grilla
        self.max_pasos = max_pasos

    def analizar_mascara(self, mascara_2d: np.ndarray) -> ReporteMorfologicoColor:
        if mascara_2d.ndim != 2:
            raise ValueError("La máscara debe tener exactamente 2 dimensiones.")
            
        alto, ancho = mascara_2d.shape
        pixeles_totales = alto * ancho
        
        # --- ETAPA 1: Análisis Inicial (M1) y Extracción de Ruido ---
        area_inicial = int(np.sum(mascara_2d))
        cobertura_inicial = (area_inicial / pixeles_totales) * 100.0
        
        estructura_8_vecinos = np.ones((3, 3), dtype=int)
        res_label_ini: tuple[np.ndarray, int] = label(mascara_2d, structure=estructura_8_vecinos)  # type: ignore
        mascara_etiquetada_ini, islas_iniciales = res_label_ini
        
        if area_inicial == 0:
            return ReporteMorfologicoColor(0.0, 0.0, 0.0, 0, 0, 0.0, 0, np.zeros((self.K, self.K)), np.zeros((self.K, self.K, 3)))

        # Grillas de acumulación espacial
        grilla_categorias = np.zeros((self.K, self.K, 3), dtype=int)
        grilla_densidad = np.zeros((self.K, self.K), dtype=float)
        
        paso_y = alto / self.K
        paso_x = ancho / self.K

        # Evaluamos y registramos el Ruido Real sobre M1 antes de que se borre
        for i in range(1, islas_iniciales + 1):
            isla_binaria_ini = (mascara_etiquetada_ini == i)
            area_isla_ini = int(np.sum(isla_binaria_ini))
            
            # Si mide menos de 15 píxeles en la imagen original, es ruido de superficie inequívoco
            if area_isla_ini <= 15:
                coordenadas_y, coordenadas_x = np.where(isla_binaria_ini)
                cy = min(int(np.mean(coordenadas_y) / paso_y), self.K - 1)
                cx = min(int(np.mean(coordenadas_x) / paso_x), self.K - 1)
                grilla_categorias[cy, cx, 2] += 1  # Registramos Ruido en M1

        # --- ETAPA 2: Bucle de Codo Intermedio Basado en Estabilización de Islas ---
        pasos_optimos = 1
        mascara_actual = mascara_2d.copy()
        
        # Primera erosión de control
        mascara_actual = binary_erosion(mascara_actual, structure=estructura_8_vecinos)
        _, islas_previas = label(mascara_actual, structure=estructura_8_vecinos)  # type: ignore
        
        for p in range(2, self.max_pasos + 1):
            mascara_sig = binary_erosion(mascara_actual, structure=estructura_8_vecinos)
            _, islas_sig = label(mascara_sig, structure=estructura_8_vecinos)  # type: ignore
            
            # Si el número de islas deja de cambiar drásticamente, encontramos el codo morfológico
            if islas_sig == 0 or abs(islas_previas - islas_sig) <= 1:
                pasos_optimos = p - 1
                break
                
            mascara_actual = mascara_sig
            islas_previas = islas_sig
            pasos_optimos = p

        # Reconstrucción (Estabilización M2)
        mascara_filtrada = mascara_2d.copy()
        for _ in range(pasos_optimos):
            mascara_filtrada = binary_erosion(mascara_filtrada, structure=estructura_8_vecinos)
        for _ in range(pasos_optimos):
            mascara_filtrada = binary_dilation(mascara_filtrada, structure=estructura_8_vecinos)
            
        area_estabilizada = int(np.sum(mascara_filtrada))
        cobertura_estabilizada = (area_estabilizada / pixeles_totales) * 100.0
        retencion_masa = (area_estabilizada / area_inicial) * 100.0

        # --- ETAPA 3: Análisis de Estructuras Limpias (M2) ---
        res_label_est: tuple[np.ndarray, int] = label(mascara_filtrada, structure=estructura_8_vecinos)  # type: ignore
        mascara_etiquetada_est, islas_estabilizadas = res_label_est
        persistencia_trama = (islas_estabilizadas / islas_iniciales) * 100.0 if islas_iniciales > 0 else 0.0

        # Clasificamos Acentuaciones y Texturas Estructurales sobre M2
        for i in range(1, islas_estabilizadas + 1):
            isla_binaria_est = (mascara_etiquetada_est == i)
            area_isla_est = int(np.sum(isla_binaria_est))
            
            coordenadas_y, coordenadas_x = np.where(isla_binaria_est)
            cy = min(int(np.mean(coordenadas_y) / paso_y), self.K - 1)
            cx = min(int(np.mean(coordenadas_x) / paso_x), self.K - 1)
            
            if area_isla_est > 350:
                grilla_categorias[cy, cx, 0] += 1  # Ace (Acento Consistente)
            else:
                grilla_categorias[cy, cx, 1] += 1  # Tex (Textura / Trama Estable)

        # Mapeo de Densidad de Cobertura para el fondo visual del pixelado
        for f in range(self.K):
            y_ini, y_fin = int(f * paso_y), int((f + 1) * paso_y)
            for c in range(self.K):
                x_ini, x_fin = int(c * paso_x), int((c + 1) * paso_x)
                bloque = mascara_filtrada[y_ini:y_fin, x_ini:x_fin]
                if bloque.size > 0:
                    grilla_densidad[f, c] = (np.sum(bloque) / bloque.size) * 100.0

        return ReporteMorfologicoColor(
            cobertura_inicial=cobertura_inicial, cobertura_estabilizada=cobertura_estabilizada,
            retencion_masa=retencion_masa, islas_iniciales=islas_iniciales,
            islas_estabilizadas=islas_estabilizadas, persistencia_trama=persistencia_trama,
            pasos_optimos=pasos_optimos, grilla_densidad=grilla_densidad, grilla_categorias=grilla_categorias
        )

    def renderizar_diagnostico_visual(self, mascara_2d: np.ndarray, titulo_ventana: str = "Diagnóstico") -> None:
        reporte = self.analizar_mascara(mascara_2d)
        estructura_8_vecinos = np.ones((3, 3), dtype=int)
        
        alto, ancho = mascara_2d.shape
        paso_y = alto / self.K
        paso_x = ancho / self.K
        
        # 1. Regenerar procesos intermedios
        m_eros = mascara_2d.copy()
        for _ in range(reporte.pasos_optimos):
            m_eros = binary_erosion(m_eros, structure=estructura_8_vecinos)
        m_estab = m_eros.copy()
        for _ in range(reporte.pasos_optimos):
            m_estab = binary_dilation(m_estab, structure=estructura_8_vecinos)

        # 2. CONSTRUCCIÓN DE LA MÁSCARA CROMÁTICA DE CATEGORÍAS (Panel 3)
        # Creamos un lienzo RGB inicializado en blanco (o fondo oscuro si preferís)
        lienzo_categorias = np.ones((alto, ancho, 3), dtype=float)  # Fondo blanco de base
        
        # Primero pintamos TODO el color original (M1) en un GRIS tenue para el ruido descartado
        # (RGB: 0.75, 0.75, 0.75 da un gris claro limpio)
        lienzo_categorias[mascara_2d] = [0.75, 0.75, 0.75]
        
        # Volvemos a etiquetar M2 para separar e identificar componentes estructurales en el renderizado
        res_label_est: tuple[np.ndarray, int] = label(m_estab, structure=estructura_8_vecinos)  # type: ignore
        mascara_etiquetada_est, islas_estabilizadas = res_label_est
        
        for i in range(1, islas_estabilizadas + 1):
            isla_binaria = (mascara_etiquetada_est == i)
            area_isla = np.sum(isla_binaria)
            
            if area_isla > 350:
                # ACENTOS / PAÑOS NETOS -> Pintamos en AZUL sólido (RGB: 0.11, 0.44, 0.71)
                lienzo_categorias[isla_binaria] = [0.11, 0.44, 0.71]
            else:
                # TEXTURAS / TRAMAS -> Pintamos en NARANJA (RGB: 0.91, 0.49, 0.13)
                lienzo_categorias[isla_binaria] = [0.91, 0.49, 0.13]

        # Impresión limpia en terminal
        print("\n" + "="*85)
        print(f"TABLA DE DESCRIPTORES (Filtro Adaptativo: {reporte.pasos_optimos} pasadas)")
        print("="*85)
        print(f"{'Descriptor Morfológico':<25} | {'Estado Inicial (M1)':<22} | {'Estado Estabilizado (M2)':<25} | {'Indicador (Delta)':<25}")
        print("-"*85)
        print(f"{'Área / Cobertura':<25} | {reporte.cobertura_inicial:<19.2f}% | {reporte.cobertura_estabilizada:<22.2f}% | Retención Masa: {reporte.retencion_masa:.1f}%")
        print(f"{'Fragmentación (Islas)':<25} | {reporte.islas_iniciales:<22} | {reporte.islas_estabilizadas:<25} | Persistencia Trama: {reporte.persistencia_trama:.1f}%")
        print("="*85 + "\n")

        # Configuración de paneles en Matplotlib
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), num=titulo_ventana)
        ((ax_orig, ax_eros), (ax_estab, ax_grilla)) = axes

        # Configuración de los tres paneles de imágenes
        ax_orig.imshow(mascara_2d, cmap='gray', interpolation='nearest')
        ax_orig.set_title(f"1. Máscara Original (M1)\nIslas: {reporte.islas_iniciales}", fontsize=10)
        
        ax_eros.imshow(m_eros, cmap='magma', interpolation='nearest')
        ax_eros.set_title(f"2. Contracción (Erosión x{reporte.pasos_optimos})", fontsize=10)
        
        # PANEL 3: Renderizado Cromático de Categorías Reales
        ax_estab.imshow(lienzo_categorias, interpolation='nearest')
        ax_estab.set_title("3. Diagnóstico Cromático (M2)\n[Gris: Ruido | Naranja: Textura | Azul: Acento]", fontsize=10)

        # Proyectar líneas rojas guía sobre los paneles 1, 2 y 3
        for ax in (ax_orig, ax_eros, ax_estab):
            ax.set_xlim(0, ancho)
            ax.set_ylim(alto, 0)
            for i in range(1, self.K):
                ax.axhline(i * paso_y, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.axvline(i * paso_x, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axis('off')

        # PANEL 4: Grilla de Diagnóstico Proporcional
        ax_grilla.imshow(
            reporte.grilla_densidad, 
            cmap='YlOrRd', 
            interpolation='nearest', 
            origin='upper', 
            vmin=0, 
            vmax=100,
            extent=[0, ancho, alto, 0]
        )
        ax_grilla.set_title(f"4. Grilla de Diagnóstico Proporcional ({self.K}x{self.K})", fontsize=10)
        
        for i in range(1, self.K):
            ax_grilla.axhline(i * paso_y, color='#7f8c8d', linestyle='-', linewidth=0.6)
            ax_grilla.axvline(i * paso_x, color='#7f8c8d', linestyle='-', linewidth=0.6)

        ax_grilla.set_xticks([i * paso_x for i in range(self.K + 1)])
        ax_grilla.set_yticks([i * paso_y for i in range(self.K + 1)])
        ax_grilla.set_xticklabels([f"C{i}" for i in range(self.K + 1)], fontsize=8)
        ax_grilla.set_yticklabels([f"F{i}" for i in range(self.K + 1)], fontsize=8)
        ax_grilla.set_aspect('equal')

        # Superponer los contadores de texto en la grilla proporcional
        for f in range(self.K):
            pos_y_texto = (f + 0.5) * paso_y
            for c in range(self.K):
                pos_x_texto = (c + 0.5) * paso_x
                
                ace = reporte.grilla_categorias[f, c, 0]
                tex = reporte.grilla_categorias[f, c, 1]
                rui = reporte.grilla_categorias[f, c, 2]
                densidad = reporte.grilla_densidad[f, c]
                
                if ace == 0 and tex == 0 and rui == 0:
                    ax_grilla.text(pos_x_texto, pos_y_texto, "-", ha="center", va="center", color="#7f8c8d", fontsize=9)
                else:
                    texto_celda = f"Ace: {ace}\nTex: {tex}\nRui: {rui}"
                    color_borde = "#2c3e50" if densidad > 40 else "#bdc3c7"
                    ax_grilla.text(
                        pos_x_texto, pos_y_texto, texto_celda, 
                        ha="center", va="center", color="#2c3e50", 
                        fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', alpha=0.85, edgecolor=color_borde)
                    )

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import os
    from PIL import Image
    dir_entrada = "./../entrada/isleta/pruebas"
    filename = "PIÑEYRO_ISLETA_260616_F1937_F1941.png"
    ruta_mascara_real = os.path.join(dir_entrada, filename)
    
    if os.path.exists(ruta_mascara_real):
        print(f"-> Analizando imagen real: {ruta_mascara_real}")
        img_pil = Image.open(ruta_mascara_real).convert("L")
        matriz_analisis = np.array(img_pil) > 128
        titulo = "Análisis Morfológico de Fachada Real"
    else:
        # Contingencia sintética si no hay archivo
        matriz_analisis = np.zeros((600, 600), dtype=bool)
        matriz_analisis[200:450, 150:400] = True
        for i in range(5):
            matriz_analisis[100:130, 100 + i*90 : 140 + i*90] = True
        np.random.seed(42)
        matriz_analisis = np.logical_or(matriz_analisis, np.random.rand(600, 600) > 0.998)
        titulo = "Control Simulado Semiadaptativo"

    # Instancia balanceada
    analizador = AnalizadorMorfologicoEspacial(divisiones_grilla=6, max_pasos=6)
    analizador.renderizar_diagnostico_visual(matriz_analisis, titulo)