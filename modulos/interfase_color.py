import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.transforms as transforms
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation
from skimage import color
from typing import Any, cast
from modulos.modelo_color import AgrupadorColor, EstadoImagen, NomencladorColor
from modulos.configuracion import ParametrosAnalisis
from modulos.metricas import MetricasProcesamiento


class VentanaDetalleColor:
    """
    Manejador persistente para mostrar el análisis detallado y la mancha 
    espacial de la familia analizada en resolución nativa.
    """
    def __init__(self) -> None:
        self.fig: Figure | None = None
        self.ax_imagen_origen: Axes | None = None
        self.ax_segmentada: Axes | None = None
        self.ax_todas_las_familias: Axes | None = None  # <--- Eje para el mapa total
        self.ax_color_centro: Axes | None = None
        self.ax_color_promedio: Axes | None = None
        self.ax_color_pico: Axes | None = None
        self.mapa_total_familias: np.ndarray | None = None  # <--- Contenedor del lienzo precargado
        self.cobertura_total_txt: str = ""
        self.activa: bool = False
        self._mascara_precalculada: np.ndarray | None = None

        self.ax_hist_L: Axes | None = None
        self.ax_hist_C: Axes | None = None
        self.ax_hist_T: Axes | None = None

    def inicializar(self) -> None:
        """Inicializa la ventana con una grilla geométrica fija y de tipo estricto."""
        figura = plt.figure(figsize=(12, 9))
        if not isinstance(figura, Figure): return
        self.fig = figura
        
        # 1. MANCHAS (Arriba: y=0.52, alto=0.40) - SIN CAMBIOS
        self.ax_imagen_origen = cast(Axes, self.fig.add_axes((0.05, 0.52, 0.28, 0.40)))
        self.ax_todas_las_familias = cast(Axes, self.fig.add_axes((0.36, 0.52, 0.28, 0.40)))
        self.ax_segmentada = cast(Axes, self.fig.add_axes((0.67, 0.52, 0.28, 0.40)))
        
        # Coordenadas: (x, y, ancho, alto)
        # Ajusta estos valores según el espacio libre en tu figura de 12x9
        self.ax_hist_L = self.fig.add_axes((0.08, 0.45, 0.22, 0.10))
        self.ax_hist_C = self.fig.add_axes((0.39, 0.45, 0.22, 0.10))
        self.ax_hist_T = self.fig.add_axes((0.70, 0.45, 0.22, 0.10))


        #self.ax_color_centro = cast(Axes, self.fig.add_axes((0.08, 0.18, 0.22, 0.10)))
        self.ax_color_centro = cast(Axes, self.fig.add_axes((0.07, 0.2, 0.1, 0.13)))
        #self.ax_color_promedio = cast(Axes, self.fig.add_axes((0.39, 0.18, 0.22, 0.10)))
        #self.ax_color_promedio = cast(Axes, self.fig.add_axes((0.39, 0.15, 0.075, 0.08)))
        #self.ax_color_pico = cast(Axes, self.fig.add_axes((0.70, 0.18, 0.22, 0.10)))
        #self.ax_color_pico = cast(Axes, self.fig.add_axes((0.70, 0.15, 0.075, 0.08)))
        self.ax_color_promedio = cast(Axes, self.fig.add_axes((0.37, 0.24, 0.1, 0.09)))
        self.ax_color_pico = cast(Axes, self.fig.add_axes((0.37, 0.06, 0.1, 0.09)))
        
        ejes: list[Axes] = [
            self.ax_imagen_origen, self.ax_segmentada, self.ax_todas_las_familias,
            self.ax_color_centro, self.ax_color_promedio, self.ax_color_pico
        ]
        
        for ax in ejes:
            ax.axis('off')


        
        # Añade los nuevos ejes a la lista para limpiar
        ejes = [
            self.ax_imagen_origen, self.ax_segmentada, self.ax_todas_las_familias,
            self.ax_color_centro, self.ax_color_promedio, self.ax_color_pico,
            self.ax_hist_L, self.ax_hist_C, self.ax_hist_T
        ]

        self.txt_bienvenida = self.fig.text(
            0.5, 0.5, "Seleccione una familia\nen el gráfico 3D", 
            horizontalalignment='center', 
            verticalalignment='center', 
            color='gray', 
            fontsize=14,
            transform=self.fig.transFigure
        )
        
        self.activa = True
        
        def _al_cerrar_ficha(event: Any) -> None:
            self.activa = False
            for num in plt.get_fignums():
                fig_obj = plt.figure(num)
                if fig_obj != self.fig:
                    plt.close(fig_obj)

        self.fig.canvas.mpl_connect('close_event', _al_cerrar_ficha)

        # Inicialmente ocultos
        for ax in [self.ax_hist_L, self.ax_hist_C, self.ax_hist_T]:
            ax.set_visible(False)
  
    def _on_close(self, event: Any) -> None:
        self.activa = False

    def actualizar(
        self, 
        idx: int, 
        centro_lab: np.ndarray, 
        rgb_centro: np.ndarray, 
        indices_pixeles_familia: np.ndarray, 
        anclas: np.ndarray, 
        estado_contexto: EstadoImagen,
        config: ParametrosAnalisis,
        metricas: MetricasProcesamiento,
        nomenclador: Any | None = None
    ) -> None:
        """Actualiza todos los paneles con la información de la elipsoide seleccionada."""
        if not self.activa or self.fig is None:
            return
        
        if hasattr(self, 'txt_bienvenida') and self.txt_bienvenida:
            self.txt_bienvenida.remove()
            self.txt_bienvenida = None
            
        assert self.ax_imagen_origen is not None
        assert self.ax_segmentada is not None
        assert self.ax_todas_las_familias is not None
        assert self.ax_color_centro is not None
        assert self.ax_color_promedio is not None
        assert self.ax_color_pico is not None
        assert self.ax_hist_L is not None
        assert self.ax_hist_C is not None
        assert self.ax_hist_T is not None

        alto_real, ancho_real = estado_contexto.alto, estado_contexto.ancho
        rgb_completo = estado_contexto.rgb_plano
        tol_L, tol_C, tol_T = config.tolerancia_L, config.tolerancia_C, config.tolerancia_T

        # --- CÁLCULOS ---
        nombre_centro = nomenclador.obtener_nombre(centro_lab) if nomenclador else "Desconocido"
        pixeles_f_lab = estado_contexto.lab_plano[indices_pixeles_familia]
        
        C_centro_norm = np.hypot(centro_lab[1], centro_lab[2])
        dir_a = centro_lab[1] / C_centro_norm if C_centro_norm > 1e-6 else 1.0
        dir_b = centro_lab[2] / C_centro_norm if C_centro_norm > 1e-6 else 0.0
        
        H_val = pixeles_f_lab[:, 1] * (-dir_b) + pixeles_f_lab[:, 2] * dir_a
        C_val = pixeles_f_lab[:, 1] * dir_a + pixeles_f_lab[:, 2] * dir_b
        L_val = pixeles_f_lab[:, 0]

        # Cálculos para muestras de color
        promedio_lab = np.mean(pixeles_f_lab, axis=0) if len(pixeles_f_lab) > 0 else centro_lab
        rgb_promedio = np.clip(color.lab2rgb(promedio_lab.reshape(1, 1, 3)).flatten(), 0, 1)
        nombre_promedio = nomenclador.obtener_nombre(promedio_lab) if nomenclador else "Desconocido"

        if len(anclas) > 0 and len(pixeles_f_lab) > 0:
            lab_pico = anclas[np.argmin(np.linalg.norm(anclas - centro_lab, axis=1))]
            rgb_pico = np.clip(color.lab2rgb(lab_pico.reshape(1, 1, 3)).flatten(), 0, 1)
            nombre_pico = nomenclador.obtener_nombre(lab_pico) if nomenclador else "Desconocido"
        else:
            lab_pico, rgb_pico, nombre_pico = centro_lab, rgb_centro, nombre_centro

        # --- 1. RENDERIZADO DE MANCHAS ---
        mancha_perfecta = np.ones((alto_real, ancho_real, 3), dtype=np.float32)
        if len(indices_pixeles_familia) > 0:
            mancha_perfecta.reshape(-1, 3)[indices_pixeles_familia] = rgb_completo[indices_pixeles_familia]
            
        area_util, total_abs = f_num(metricas.muestras_utiles), f_num(metricas.total_absoluto)
        pix_cub = f_num(metricas.pixeles_cubiertos)
        pct_cob = f_num(metricas.cobertura_util_pct, decimales=2)
        cant = f_num(metricas.familias.get(idx, {}).get("cantidad_pixeles", 0)) if metricas.familias else "0"
        pct_f = f_num(metricas.familias.get(idx, {}).get("porcentaje_cobertura", 0.0), decimales=2) if metricas.familias else "0.00"

        self.ax_imagen_origen.clear()
        self.ax_imagen_origen.imshow(np.nan_to_num(estado_contexto.rgb, nan=1.0))
        self.ax_imagen_origen.set_title(f"Imagen original: {total_abs} px ({metricas.ancho_px} x {metricas.alto_px})\nArea útil: {area_util} px", fontsize=11)
        self.ax_imagen_origen.axis('off')

        self.ax_todas_las_familias.clear()
        if self.mapa_total_familias is not None:
            self.ax_todas_las_familias.imshow(self.mapa_total_familias)
        self.ax_todas_las_familias.set_title(f"Reconstrucción total: {pix_cub} px\nCobertura s/ área útil: {pct_cob}%", fontsize=11)
        self.ax_todas_las_familias.axis('off')

        self.ax_segmentada.clear()
        self.ax_segmentada.imshow(mancha_perfecta)
        self.ax_segmentada.set_title(f"Familia {idx + 1}: {cant} px\nCobertura s/ área útil: {pct_f}%", fontsize=11)
        self.ax_segmentada.axis('off')

        # --- 2. RENDERIZADO DE HISTOGRAMAS ---
        for ax in [self.ax_hist_L, self.ax_hist_C, self.ax_hist_T]: ax.set_visible(True)
        self._dibujar_histograma_1d(self.ax_hist_L, L_val, pixeles_f_lab, tol_L, "Histograma L* (Luminosidad/Valor)", centro_lab, 'L')
        self._dibujar_histograma_1d(self.ax_hist_C, C_val, pixeles_f_lab, tol_C, "Histograma C* (Chroma/Saturación) ", centro_lab, 'C')
        self._dibujar_histograma_1d(self.ax_hist_T, H_val, pixeles_f_lab, tol_T, "Histograma h (Tono)", centro_lab, 'H')

        # --- 3. RENDERIZADO DE MUESTRAS ---
        # Reactivamos los paneles de visualización de color
        for ax in [self.ax_color_centro, self.ax_color_promedio, self.ax_color_pico]:
            ax.set_visible(True)

        # Función auxiliar para calcular LCH y formatear los renglones
        def formatear_texto_color(nombre: str, lab: np.ndarray) -> str:
            L, a, b = lab
            C = np.hypot(a, b)
            h = np.degrees(np.arctan2(b, a)) % 360
            return f"Color: {nombre}\nCIELAB:\nL*: {L:.2f},  a*: {a:.2f},  b*: {b:.2f}\nCIELCh:\nL*: {L:.2f},  C*: {C:.2f},  h: {h:.1f}°"

        

        # 1. Centroide
        self.ax_color_centro.clear()
        self.ax_color_centro.imshow([[np.clip(rgb_centro, 0, 1)]])
        self.ax_color_centro.axis('off')
        self.ax_color_centro.set_title("Color de la familia\nCentroide geométrico", fontsize=11, ha='left', loc='left', x=0)
        self.ax_color_centro.text(1.15, 0.65, formatear_texto_color(nombre_centro, centro_lab), 
                                  transform=self.ax_color_centro.transAxes, ha='left', va='center', 
                                  fontsize=9)

        # 2. Promedio
        self.ax_color_promedio.clear()
        self.ax_color_promedio.imshow([[np.clip(rgb_promedio, 0, 1)]])
        self.ax_color_promedio.axis('off')
        self.ax_color_promedio.set_title("Promedio\nde las muestras", fontsize=11, ha='left', loc='left', x=0)
        self.ax_color_promedio.text(1.15, 0.5, formatear_texto_color(nombre_promedio, promedio_lab), 
                                    transform=self.ax_color_promedio.transAxes, ha='left', va='center', 
                                    fontsize=9)

        # 3. Pico
        self.ax_color_pico.clear()
        self.ax_color_pico.imshow([[np.clip(rgb_pico, 0, 1)]])
        self.ax_color_pico.axis('off')
        self.ax_color_pico.set_title("Pico de densidad\nde las muestras", fontsize=11, ha='left', loc='left', x=0)
        self.ax_color_pico.text(1.15, 0.5, formatear_texto_color(nombre_pico, lab_pico), 
                                transform=self.ax_color_pico.transAxes, ha='left', va='center', 
                                fontsize=9)

        self.fig.suptitle(f"FAMILIA #{idx + 1}: {nombre_centro}", fontsize=11, fontweight='normal')
        self.fig.canvas.draw_idle()
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.show()

    def _dibujar_histograma_1d(
        self, ax: Axes, valores_abs: np.ndarray, pixeles_f_lab: np.ndarray, 
        tolerancia: float, titulo: str, centro_lab: np.ndarray, atributo: str
    ) -> None:
        ax.clear()
        L0, A0, B0 = centro_lab
        C0 = np.hypot(A0, B0)
        
        if atributo == 'L':
            centro_val = L0
        elif atributo == 'C':
            centro_val = C0
        else: # 'H' o 'T'
            centro_val = 0.0
            
        radio_visual = (tolerancia / 2.0) * 1.2  # Equivalente a tolerancia / 1.2
        x_min, x_max = centro_val - radio_visual, centro_val + radio_visual
        
        # 1. Usamos 21 bins para que haya una barra central exacta en centro_val
        num_bins = 21
        bins_edges = np.linspace(x_min, x_max, num_bins + 1)
        indices = np.digitize(valores_abs, bins_edges) - 1
        ax.figure.subplots_adjust(bottom=0.3)
        
        # 2. Dibujo de barras
        for i in range(num_bins):
            # El centro del bin es el promedio de los bordes
            bin_medio = (bins_edges[i] + bins_edges[i+1]) / 2
            desvio = bin_medio - centro_val
            
            rgb_bin = self._obtener_color_teorico(desvio, atributo, centro_lab)
            
            mask = indices == i
            if np.sum(mask) > 0:
                ax.bar(bin_medio, np.sum(mask), width=(bins_edges[i+1]-bins_edges[i]), 
                       color=rgb_bin, edgecolor=None, linewidth=0.5)

        # 3. Referencias (Se mantiene igual que antes)
        color_centroide = self._obtener_color_teorico(0.0, atributo, centro_lab)
        ax.plot([x_min, x_max], [0, 0], color="black", 
                linestyle='-', linewidth=0.80, transform=ax.get_xaxis_transform(), 
                clip_on=False, zorder=0)
        

        puntos_a_dibujar = [x_min, centro_val, x_max]
        if atributo == 'C':
            if x_min <= 0.0 <= x_max:
                ax.plot([0.0, 0.0], [0.0, -0.7], color='black', linestyle='-', 
                        alpha=1.0, linewidth=1.0, transform=ax.get_xaxis_transform(), 
                        clip_on=False, zorder=0.00)
                puntos_a_dibujar.append(0.0)


        for px in sorted(list(set(puntos_a_dibujar))):
            desvio_ref = px - centro_val
            #if atributo != 'L' or px != centro_val:
            ax.plot([px], [0], marker='o', markersize=8, 
                    color=self._obtener_color_teorico(desvio_ref, atributo, centro_lab),
                    transform=ax.get_xaxis_transform(), clip_on=False, markeredgecolor="black", zorder=5.1)

        # 2. Posicionamiento: centro en X, debajo de los otros marcadores
        # Usamos una altura menor (-0.55) para que queden debajo de la línea y marcadores
        # Offset fijo de 8 puntos (hacia cada lado)
        markersize = 12
        dx_px = markersize / (72.0 * 2)
        # La posición Y fija en -0.35 usando el transformador del eje X
        y_trans = -0.35
        if atributo == "H":
            y_trans = -0.65
        
        # Transformación para el cuadrado izquierdo (offset negativo)
        trans_min = ax.get_xaxis_transform() + transforms.ScaledTranslation(-dx_px, y_trans, ax.figure.dpi_scale_trans)
        # Transformación para el cuadrado derecho (offset positivo)
        trans_max = ax.get_xaxis_transform() + transforms.ScaledTranslation(dx_px, y_trans, ax.figure.dpi_scale_trans)
        
        color_min = self._obtener_color_teorico(x_min - centro_val, atributo, centro_lab)
        color_max = self._obtener_color_teorico(x_max - centro_val, atributo, centro_lab)
        
        # Dibujamos ambos cuadrados en el centro del eje (centro_val)
        # La transformación se encarga de moverlos a los lados
        ax.plot([centro_val], [0], marker='s', markersize=markersize, 
                color=color_min, transform=trans_min, 
                clip_on=False, markeredgecolor="white")
        
        ax.plot([centro_val], [0], marker='s', markersize=markersize, 
                color=color_max, transform=trans_max, 
                clip_on=False, markeredgecolor="white")

        # Eje angular secundario para Tono (T o H)
        if atributo == "H":
            C0 = np.hypot(A0, B0)
            if C0 > 0.1: 
                angle_centroid = np.degrees(np.arctan2(B0, A0)) % 360
                
                # 2. Función forward normalizada
                def forward(linear_dev):
                    return (angle_centroid + np.degrees(linear_dev / C0)) % 360
                
                # 3. Función inversa (calculamos la diferencia respecto al centroide)
                def inverse(angle_deg):
                    # Calculamos el desplazamiento angular mínimo para evitar saltos en 0/360
                    diff = (angle_deg - angle_centroid + 180) % 360 - 180
                    return np.radians(diff) * C0
                
                # 1. Dejamos el eje angular en la posición principal (abajo, sin offset)
                sec_ax = ax.secondary_xaxis('bottom', functions=(forward, inverse))
                sec_ax.spines['bottom'].set_position(('outward', 0))
                sec_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}°"))
                
                # 2. Movemos el eje lineal a la posición secundaria (abajo, con offset)
                ax.spines['bottom'].set_position(('outward', 20))
                
                # Mantenemos los estilos originales
                #sec_ax.tick_params(axis='x', labelsize=7)
                #ax.tick_params(axis='x', labelsize=7)


        ax.set_title(titulo, fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.axvline(centro_val, color='black', linestyle='--', linewidth=0.8)
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)

        # Ajuste de etiquetas (Se mantiene igual, ahora funcionará mejor con los 21 bins)
        ax.figure.canvas.draw()

        #Ajuste del 0 en Chroma
        if atributo == 'C' and (x_min <= 0.0 <= x_max):
            dx = 3/72. 
            offset = transforms.ScaledTranslation(dx, 0, ax.figure.dpi_scale_trans)
            for label in ax.get_xticklabels():
                texto = label.get_text().replace('-', '-').replace('-', '-')
                try:
                    if float(texto) == 0.0:
                        label.set_transform(label.get_transform() + offset)
                        label.set_horizontalalignment('left')
                except ValueError:
                    continue

    def _calcular_color_para_bin(
        self, 
        val_medio: float, 
        atributo: str, 
        centro_lab: np.ndarray, 
        x_min: float, 
        x_max: float
    ) -> np.ndarray:
        """Calcula el color RGB consistente con el sistema local del agrupador."""
        L0, A0, B0 = centro_lab
        # Obtenemos los mismos vectores que usa el agrupador para definir la elipsoide
        uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centro_lab)
        
        # El desvío respecto al centroide es el valor actual menos el centro (o 0 si es relativo)
        desvio = val_medio
        
        if atributo == 'H':
            # Proyección sobre el vector tangente (uT)
            # Esto es lo que define el plano de tono en tu modelo
            lab_color = [L0, A0 + desvio * uT_a, B0 + desvio * uT_b]
            
        elif atributo == 'C':
            # Proyección sobre el vector radial (uR)
            lab_color = [L0, A0 + desvio * uR_a, B0 + desvio * uR_b]
            
        else: # Atributo 'L'
            # Desplazamiento lineal en el eje L
            lab_color = [L0 + desvio, A0, B0]
            
        # Retorno con clipping para asegurar rango RGB
        return np.clip(color.lab2rgb(np.array(lab_color).reshape(1, 1, 3)).flatten(), 0, 1)

    def _obtener_color_teorico(self, desvio: float, atributo: str, centro: np.ndarray) -> np.ndarray:
        L0, A0, B0 = centro
        uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centro)
        
        if atributo == 'L':
            lab = [L0 + desvio, A0, B0]
        elif atributo == 'C':
            lab = [L0, A0 + desvio * uR_a, B0 + desvio * uR_b]
        elif atributo == 'H':  # o 'T'
            lab = [L0, A0 + desvio * uT_a, B0 + desvio * uT_b]
            
        rgb = color.lab2rgb(np.array(lab).reshape(1, 1, 3)).flatten()
        return np.clip(rgb, 0, 1)

class VisualizadorEspacioColor:
    """
    Encapsula el entorno gráfico 3D de Matplotlib, renderiza los datos
    espaciales y gestiona las colisiones de eventos de mouse del usuario.
    """
    def __init__(self, config: ParametrosAnalisis, metricas: MetricasProcesamiento) -> None:
        self.config = config
        self.tolerancias = {"L": config.tolerancia_L, "C": config.tolerancia_C, "T": config.tolerancia_T}

        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.anotacion: Annotation | None = None
        self.sc_centros: PathCollection | None = None
        self.sc_anclas: PathCollection | None = None
        self.sc_paleta_mediana: PathCollection | None = None
        self.sc_muestras: PathCollection | None = None
        self.ax_reticula: Axes | None = None
        self.lista_elipsoides: list[Any] = []
        self.check_visualizacion: Any | None = None
        self.datos_centros_lab: np.ndarray = np.empty((0, 3))
        self.datos_colores_rgb: list[np.ndarray] = []
        self.nomenclador: NomencladorColor | None = None
        self._texto_titulo_paleta: Any | None = None

        self._ventana_detalle = None
        self._agrupador = None
        self._mapeo_celdas = {}
        self.lab_total = None
        self.anclas = None
        self._contexto = None
        self._nomenclador = None

        self._cid_hover = None
        self._cid_click = None
        self._cid_hover_reticula = None
        self._cid_click_reticula = None

        self.calculador = metricas

    def _setup_callbacks(self) -> None:
        """Paso 1: Este método conecta TODOS los eventos una sola vez."""
        if not self.fig or not self.fig.canvas:
            return

        # 1. Eventos de mouse principales
        self._cid_hover = self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self._cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # 2. Eventos globales de la ventana
        def cerrar_aplicacion_limpia(event: Any) -> None:
            import matplotlib.pyplot as plt
            for num in plt.get_fignums():
                fig_obj = plt.figure(num)
                if fig_obj != self.fig: 
                    plt.close(fig_obj)

        def on_leave_figure(event: Any) -> None:
            if not self.fig: return
            refrescar = False
            
            # Ocultar cartel 3D si estaba visible
            if self.anotacion and self.anotacion.get_visible():
                self.anotacion.set_visible(False)
                refrescar = True
                
            # Ocultar cartel de retícula si estaba visible
            if hasattr(self, 'anotacion_reticula') and self.anotacion_reticula.get_visible():
                self.anotacion_reticula.set_visible(False)
                refrescar = True
                
            if refrescar:
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('close_event', cerrar_aplicacion_limpia)
        self.fig.canvas.mpl_connect('figure_leave_event', on_leave_figure)

    def _on_click(self, event: Any) -> None:
        if (self.agrupador is None or 
            self._ventana_detalle is None or 
            self.lab_total is None or
            event.button != 1):
            return

        idx = -1

        if event.inaxes == self.ax and self.sc_centros is not None:
            cont, ind = self.sc_centros.contains(event)
            if cont:
                indices_candidatos = ind.get("ind", [])
                if len(indices_candidatos) > 0:
                    idx = int(indices_candidatos[0])

        elif event.inaxes == self.ax_reticula and event.xdata is not None and event.ydata is not None:
            c_click, f_click = int(np.round(event.xdata)), int(np.round(event.ydata))
            idx = self._mapeo_celdas.get((c_click, f_click), -1)

        if idx != -1 and self.estado_original is not None:
            centro_lab = self.datos_centros_lab[idx]
            rgb_color = self.datos_colores_rgb[idx]
            indices_pixeles_familia = self.familias_indices[idx]
            
            # Obtenemos los datos precalculados de forma directa y limpia
            familia_data = getattr(self.calculador, 'familias', {})
            datos_mancha = familia_data.get(idx)
            
            if datos_mancha:
                print(f"\n[ANÁLISIS TOTAL HD] Familia #{idx + 1}:")
                print(f" -> Cantidad de píxeles reales: {datos_mancha['cantidad_pixeles']}")
                print(f" -> Cobertura real sobre fachada nativa: {datos_mancha['porcentaje_cobertura']:.2f}%")

            anclas_envio = self.anclas if self.anclas is not None else np.empty((0, 3))

            # Pasamos las métricas unificadas donde la ventana podrá leer todo el contexto
            self._ventana_detalle.actualizar(
                idx=idx,
                centro_lab=centro_lab,
                rgb_centro=rgb_color,
                anclas=anclas_envio,
                indices_pixeles_familia=indices_pixeles_familia,
                estado_contexto=self.estado_original,
                metricas=self.calculador,
                config=self.config,
                nomenclador=self.nomenclador
            )

    def _on_hover(self, event: Any) -> None:
        if self.fig is None or self.anotacion is None or not hasattr(self, 'anotacion_reticula') or self.ax is None or self.ax_reticula is None:
            return

        idx = -1
        pos = (0, 0)
        anotacion_activa = None
        
        if event.inaxes == self.ax and self.sc_centros is not None:
            cont, ind = self.sc_centros.contains(event)
            if cont and len(ind.get("ind", [])) > 0:
                idx = int(ind["ind"][0])
                pos = (event.xdata, event.ydata)
                anotacion_activa = self.anotacion

        elif event.inaxes == self.ax_reticula and hasattr(self, '_mapeo_celdas'):
            c, f = int(np.round(event.xdata)), int(np.round(event.ydata))
            idx = self._mapeo_celdas.get((c, f), -1)
            pos = (c, f)
            anotacion_activa = self.anotacion_reticula

        self.anotacion.set_visible(False)
        self.anotacion_reticula.set_visible(False)

        if idx != -1 and anotacion_activa is not None:
            centro_lab = self.datos_centros_lab[idx]
            rgb_actual = self.datos_colores_rgb[idx]
            # CORRECCIÓN: Cartel emergente descriptivo en base 1
            nombre = self.nomenclador.obtener_nombre(centro_lab) if self.nomenclador else f"Familia #{idx + 1}"
            
            anotacion_activa.set_text(
                f"Familia #{idx + 1}\n{nombre}\n"
                f"L*: {centro_lab[0]:.1f} a*: {centro_lab[1]:.1f} b*: {centro_lab[2]:.1f}"
            )
            anotacion_activa.xy = pos
            bbox = anotacion_activa.get_bbox_patch()
            if bbox is not None:
                bbox.set_facecolor(rgb_actual.tolist())
            anotacion_activa.set_color("black" if centro_lab[0] > 40 else "white")
            anotacion_activa.set_visible(True)
        
        self.fig.canvas.draw_idle()

    def _dibujar_elipsoide_radial(self, centro_lab: np.ndarray, color_rgb: np.ndarray) -> None:
        if self.ax is None: return
        ax3d = cast(Axes3D, self.ax)

        L0, a0, b0 = centro_lab
        C0 = np.sqrt(a0**2 + b0**2)
        h0 = np.arctan2(b0, a0)
        

        uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centro_lab)
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_esf = np.cos(u) * np.sin(v)
        y_esf = np.sin(u) * np.sin(v)
        z_esf = np.cos(v)
        
        r_L = self.tolerancias["L"] / 2
        r_C = self.tolerancias["C"] / 2
        r_T = self.tolerancias["T"] / 2
        
        # Expresión geométrica directa usando la proyección unificada
        a_rot = x_esf * r_T * uT_a + y_esf * r_C * uR_a + a0
        b_rot = x_esf * r_T * uT_b + y_esf * r_C * uR_b + b0
        L_rot = z_esf * r_L + L0
        
        elipsoide = ax3d.plot_surface(a_rot, b_rot, L_rot, color=color_rgb, alpha=0.15, linewidth=0, shade=True)
        self.lista_elipsoides.append(elipsoide)

    def configurar_titulo(self, ruta_imagen: str) -> None:
        """
        Permite actualizar de forma externa el título del gráfico con el nombre 
        del archivo procesado y las estadísticas del agrupamiento.
        """
        if self.ax is None or self.fig is None:
            return
            
        # Extraemos únicamente el nombre del archivo de la ruta completa
        nombre_archivo = os.path.basename(ruta_imagen)
        
        # Estructuramos el título multirrenglón solicitado
        titulo_formateado = (
            "Espacio CIELAB\n"
            f"Archivo: {nombre_archivo}"
        )
        
        # Aplicamos el título al eje y forzamos el redibujado de la interfaz
        self.ax.set_title(titulo_formateado, fontsize=11, fontweight='normal', pad=15)
        self.fig.canvas.draw_idle()

    def configurar_titulo_paleta(
        self, 
        cantidad_familias: int | None = None, 
        configuracion: ParametrosAnalisis | None = None, 
        estado_original: Any | None = None
    ) -> None:
        """
        Actualiza el título de la retícula lateral. Añade de forma independiente 
        el bloque de Cobertura (si se pasa estado_original) y el de Parámetros 
        (si se pasa configuracion), ubicando las Familias Encontradas al final.
        """
        if self.ax_reticula is None or self.fig is None:
            return
        
        if self._texto_titulo_paleta is not None:
            try:
                self._texto_titulo_paleta.remove()
            except Exception:
                pass

        lineas_texto = []
        
        # BLOQUE 1: COBERTURA Y MUESTREO (Solo Presentación - O(1))
        if estado_original is not None:
            lineas_texto.append("─────────────────────────────────────────")
            lineas_texto.append("COBERTURA Y MUESTREO")
            
            m = self.calculador
            pct_util = (m.muestras_utiles / m.total_absoluto) * 100 if m.total_absoluto > 0 else 0.0
            pct_enmascarado = (m.muestras_enmascaradas / m.total_absoluto) * 100 if m.total_absoluto > 0 else 0.0

            lineas_texto.append(f" - Area total: {f_num(m.total_absoluto)} px ({m.ancho_px} x {m.alto_px})")
            lineas_texto.append(f" - Area útil: {f_num(m.muestras_utiles)} px ({f_num(pct_util)}%)")
            lineas_texto.append(f" - Area enmascarada: {f_num(m.muestras_enmascaradas)} px ({f_num(pct_enmascarado)}%)")
            lineas_texto.append(f" - Cobertura de paleta s/ área total: {f_num(m.cobertura_total_pct)}%")
            lineas_texto.append(f" - Cobertura de paleta s/ área útil: {f_num(m.cobertura_util_pct)}%")
            lineas_texto.append(f" - Ruido sin clasificar: {f_num(m.ruido_pct)}%")
            lineas_texto.append("─────────────────────────────────────────")

        # BLOQUE 2: PARÁMETROS DEL ALGORITMO (Independiente - Requiere configuracion)
        if configuracion is not None:
            # Si no se renderizó el bloque anterior, agregamos la línea divisoria inicial
            if estado_original is None:
                lineas_texto.append("─────────────────────────────────────────")
                
            lineas_texto.append("PARÁMETROS DEL ALGORITMO")
            lineas_texto.append(f" - Ancho de Reducción: {f_num(configuracion.ancho_analisis)} px")
            
            tol_l = f_num(configuracion.tolerancia_L, decimales=1)
            tol_c = f_num(configuracion.tolerancia_C, decimales=1)
            tol_t = f_num(configuracion.tolerancia_T, decimales=1)

            lineas_texto.append(f" - Tolerancias CIELAB:\n    ΔL* = {tol_l}  ΔC* = {tol_c}  ΔT* = {tol_t}")
            lineas_texto.append("─────────────────────────────────────────")


        # Línea de Familias Encontradas al final
        if cantidad_familias is not None:
            lineas_texto.append(f"FAMILIAS ENCONTRADAS ({cantidad_familias})")
        else:
            lineas_texto.append("FAMILIAS ENCONTRADAS")
        lineas_texto.append("─────────────────────────────────────────")


        txt_ficha = "\n".join(lineas_texto)
        
        self._texto_titulo_paleta = self.ax_reticula.text(
            0.0, 1.05, 
            txt_ficha, 
            fontsize=9, 
            fontweight='normal',
            linespacing=1.4,
            va='bottom',
            ha='left',
            transform=self.ax_reticula.transAxes
        )
        
        self.fig.canvas.draw_idle()

    def inicializar_grafico_3d(
        self, 
        estado_analisis: EstadoImagen,
        estado_original: EstadoImagen, 
        anclas: np.ndarray | None,
        paleta_mediana: EstadoImagen,
        familias: np.ndarray,
        etiquetas_familias: np.ndarray,
        ventana_detalle: VentanaDetalleColor,
        agrupador: AgrupadorColor,
        nomenclador: NomencladorColor | None = None
    ) -> None:
        """Construye la visualización tridimensional CIELAB y mapea la interactividad."""
        # Asignaciones críticas al inicio para evitar AttributeErrors o NoneTypes internos
        self.agrupador = agrupador
        self.anclas = anclas
        self.nomenclador = nomenclador
        self.estado_analisis = estado_analisis
        self.estado_original = estado_original
        self.lab_total = estado_original.lab_plano
        
        figura = plt.figure(figsize=(10, 9))
        if not isinstance(figura, Figure):
            return
        self.fig = figura

        self.datos_centros_lab = familias
        self.datos_colores_rgb = [np.clip(color.lab2rgb(c.reshape(1, 1, 3)).flatten(), 0, 1) for c in familias]

        # --- PRECALCULO UNIFICADO DE ÍNDICES EN LA CARGA ---
        self.familias_indices: list[np.ndarray] = []
        mascara_plana = estado_original.mascara.flatten() if estado_original.mascara is not None else None

        for centro in familias:
            sistema_local = self.agrupador.obtener_sistema_local(centro)
            indices_familia = self.agrupador.filtrar_pixeles_por_elipsoide(
                centro, self.lab_total, tolerancias=self.tolerancias, sistema_local=sistema_local
            )
            if mascara_plana is not None:
                indices_familia = indices_familia[mascara_plana[indices_familia].astype(bool)]
            self.familias_indices.append(indices_familia)

        self._configurar_ejes_3d()
        self._inicializar_widgets_control()
        self._construir_reticula_lateral(familias=familias, colores=self.datos_colores_rgb)

        alto_real = estado_original.alto
        ancho_real = estado_original.ancho
        total_absoluto = alto_real * ancho_real
        muestras_utiles = int(np.sum(estado_original.mascara)) if estado_original.mascara is not None else total_absoluto

        mapa_total = np.ones((alto_real, ancho_real, 3), dtype=np.float32)
        mapa_total_plano = mapa_total.reshape(-1, 3)
        
        pixeles_cubiertos_globales = np.zeros(total_absoluto, dtype=bool)

        # --- RESPONSABILIDAD DE PRESENTACIÓN EN CONSOLA ---
        print("\n--- Desglose de Cobertura de Familias ---")
        for idx, indices in enumerate(self.familias_indices):
            cantidad_mancha = len(indices)
            porcentaje_mancha = (cantidad_mancha / muestras_utiles) * 100.0 if muestras_utiles > 0 else 0.0
            
            centro_print = familias[idx].round(1)
            print(f"Familia {idx+1} {centro_print}: Cobertura de Fachada = {porcentaje_mancha:.2f}%")
            
            if cantidad_mancha > 0:
                mapa_total_plano[indices] = estado_original.rgb_plano[indices]
                pixeles_cubiertos_globales[indices] = True
                
        pixeles_cubiertos = np.sum(pixeles_cubiertos_globales)
        cobertura_util = (pixeles_cubiertos / muestras_utiles) * 100 if muestras_utiles > 0 else 0.0

        print("-----------------------------------------")
        print(f"Área Útil Analizada (Fachada sin NaNs): {muestras_utiles} muestras.")
        print(f"Cobertura Total de las Familias sobre Fachada: {cobertura_util:.2f}%")
        print(f"Residuos / Ruido sin clasificar: {100.0 - cobertura_util:.2f}%\n")

        self._ventana_detalle = ventana_detalle
        self._ventana_detalle.mapa_total_familias = mapa_total

        self.calculador = MetricasProcesamiento(
            alto_px=alto_real,
            ancho_px=ancho_real,
            total_absoluto=total_absoluto,
            muestras_utiles=muestras_utiles,
            muestras_enmascaradas=total_absoluto - muestras_utiles,
            pixeles_cubiertos=int(pixeles_cubiertos),
            cobertura_total_pct=(pixeles_cubiertos / total_absoluto) * 100 if total_absoluto > 0 else 0.0,
            cobertura_util_pct=cobertura_util,
            ruido_pct=100.0 - cobertura_util,
            familias={
                idx: {
                    "cantidad_pixeles": len(indices),
                    "porcentaje_cobertura": (len(indices) / muestras_utiles) * 100.0 if muestras_utiles > 0 else 0.0
                }
                for idx, indices in enumerate(self.familias_indices)
            }
        )

        self._renderizar_capas_datos(
            lab_muestreo=estado_analisis.lab_plano,
            rgb_muestreo=estado_analisis.rgb_plano,
            anclas=anclas,
            paleta_mediana=paleta_mediana.lab_plano,
            familias=familias,
            etiquetas_familias=etiquetas_familias,
            colores=self.datos_colores_rgb
        )

        self._setup_callbacks()

    def _configurar_ejes_3d(self) -> None:
        """Inicializa el contenedor 3D moderno y establece las escalas fijas del espacio."""
        assert self.fig is not None
        
        # Inicialización limpia que evita el warning de Matplotlib
        eje_3d = Axes3D(self.fig, rect=[0.28, 0.05, 0.65, 0.9])
        #self.ax = self.fig.add_axes(eje_3d)
        self.ax = cast(Axes3D, self.fig.add_axes(eje_3d))

        self.ax.set_xlabel('a* (Verde - Rojo +)')
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylabel('b* (Azul - Amarillo +)')
        self.ax.set_ylim(-100, 100)
        self.ax.set_zlabel('L* (Luminosidad)')
        self.ax.set_zlim(0, 100)
        self.ax.set_title('Espacio CIELAB (Elipsoides Rígidas Orientadas)')

    def _renderizar_capas_datos(
        self, 
        lab_muestreo: np.ndarray, 
        rgb_muestreo: np.ndarray, 
        anclas: np.ndarray | None,
        paleta_mediana: np.ndarray,  # <- Volvemos al tipo ndarray original
        familias: np.ndarray, 
        etiquetas_familias: np.ndarray, 
        colores: list
    ) -> None:
        """Dibuja cada una de las colecciones de puntos y geometrías en el espacio."""
        assert self.ax is not None
        assert self.ax_reticula is not None
        
        # Capa 1: Datos Brutos de la imagen (Filtrados estrictamente por máscara)
        if self.estado_analisis.mascara is not None:
            mascara_analisis_plana = self.estado_analisis.mascara.flatten().astype(bool)
            lab_muestreo_filtrado = lab_muestreo[mascara_analisis_plana]
            rgb_muestreo_filtrado = rgb_muestreo[mascara_analisis_plana]
        else:
            lab_muestreo_filtrado = lab_muestreo
            rgb_muestreo_filtrado = rgb_muestreo

        self.sc_muestras = self.ax.scatter(
            lab_muestreo_filtrado[:, 1], 
            lab_muestreo_filtrado[:, 2], 
            zs=lab_muestreo_filtrado[:, 0].tolist(), 
            c=rgb_muestreo_filtrado, 
            s=5, 
            alpha=0.02, 
            label='Datos Brutos'
        )

        # Capa 2: Anclas de densidad
        if anclas is not None and len(anclas) > 0:
            self.sc_anclas = self.ax.scatter(anclas[:, 1], anclas[:, 2], zs=anclas[:, 0].tolist(), 
                                             c='yellow', s=50, marker='.', edgecolors='black', label='Anclas', zorder=10)

        # Capa 3: Paleta Mediana de soporte
        # Se procesa directamente el ndarray como estaba planeado originalmente
        rgb_paleta = np.clip(color.lab2rgb(paleta_mediana.reshape(-1, 1, 3)).reshape(-1, 3), 0.0, 1.0)
        self.sc_paleta_mediana = self.ax.scatter(paleta_mediana[:, 1], paleta_mediana[:, 2], zs=paleta_mediana[:, 0].tolist(), 
                                                 c=rgb_paleta, s=3, alpha=0.8, label='Paleta Mediana')

        # Capa 4: Centros de familias con tamaños dinámicos
        # --- UNIFICACIÓN DE FUENTE DE VERDAD (PIXELES REALES HD) ---
        conteos_reales = []
        # Usamos la máscara nativa de la fachada original sobre el set de datos HD completo
        mascara_plana = self.estado_original.mascara.flatten() if self.estado_original.mascara is not None else None
        
        for centro in familias:
            # Filtrado directo en alta resolución sobre la imagen real
            if self.lab_total is not None:
                indices = self.agrupador.filtrar_pixeles_por_elipsoide(
                    centro, self.lab_total, tolerancias=self.tolerancias
                )
            if mascara_plana is not None:
                indices = indices[mascara_plana[indices].astype(bool)]
            conteos_reales.append(len(indices))

        conteos_reales = np.array(conteos_reales)
        tamanos_reales: Any = np.full(len(familias), 80.0, dtype=np.float64)
        
        if len(conteos_reales) > 0 and np.max(conteos_reales) != np.min(conteos_reales):
            conteos_raiz = np.sqrt(conteos_reales)
            min_r, max_r = np.sqrt(np.min(conteos_reales)), np.sqrt(np.max(conteos_reales))
            # Mapeo lineal de diámetros basado exclusivamente en el área útil real de la fachada
            tamanos_reales = 1 + (conteos_raiz - min_r) / (max_r - min_r) * (300 - 1)

        self.sc_centros = self.ax.scatter(
            familias[:, 1], familias[:, 2], zs=familias[:, 0].tolist(),
            c=colores, s=tamanos_reales, marker='o', edgecolors='black', zorder=100, depthshade=False
        )
        # ------------------------------------------------------------

        # Capa 5: Elipsoides radiales locales
        self.lista_elipsoides = []
        for i, centro in enumerate(familias):
            self._dibujar_elipsoide_radial(centro, colores[i])

        # Visibilidad inicial
        estados = [True, False, False, False, True] 
        if self.sc_muestras: self.sc_muestras.set_visible(estados[0])
        if self.sc_anclas: self.sc_anclas.set_visible(estados[1])
        if self.sc_paleta_mediana: self.sc_paleta_mediana.set_visible(estados[2])
        for elipsoide in self.lista_elipsoides: elipsoide.set_visible(estados[3])
        if self.sc_centros: self.sc_centros.set_visible(estados[4])

        # Anotaciones
        self.anotacion = self.ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"), zorder=200)
        self.anotacion_reticula = self.ax_reticula.annotate("", xy=(0,0), xytext=(15, 15), textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"), zorder=200)
        
        self.anotacion.set_visible(False)
        self.anotacion_reticula.set_visible(False)

    def _inicializar_widgets_control(self) -> None:
        """Construye los CheckButtons nativos para alternar la visibilidad de las capas."""
        assert self.fig is not None
        from matplotlib.widgets import CheckButtons
        
        ax_check = self.fig.add_axes((0.02, 0.02, 0.22, 0.16))
        etiquetas = ["Ver Muestras", "Ver Anclas", "Paleta Mediana", "Ver Elipsoides", "Ver Familias"]
        estados_iniciales = [True, False, False, False, True]
        
        self.check_visualizacion = CheckButtons(
            ax=ax_check, labels=etiquetas, actives=estados_iniciales,
            frame_props={'s': 80}, check_props={'s': 80}
        )
        
        for label_text in self.check_visualizacion.labels:
            label_text.set_fontsize(10)

        def alternar_capas(label: str | None) -> None:
            if not self.fig: return
            if label == "Ver Muestras" and self.sc_muestras:
                self.sc_muestras.set_visible(not self.sc_muestras.get_visible())
            elif label == "Ver Anclas" and self.sc_anclas:
                self.sc_anclas.set_visible(not self.sc_anclas.get_visible())
            elif label == "Paleta Mediana" and self.sc_paleta_mediana:
                self.sc_paleta_mediana.set_visible(not self.sc_paleta_mediana.get_visible())
            elif label == "Ver Elipsoides" and len(self.lista_elipsoides) > 0:
                vis = not self.lista_elipsoides[0].get_visible()
                for e in self.lista_elipsoides: e.set_visible(vis)
            elif label == "Ver Familias" and self.sc_centros:
                self.sc_centros.set_visible(not self.sc_centros.get_visible())
            self.fig.canvas.draw_idle()

        self.check_visualizacion.on_clicked(alternar_capas)

    def _construir_reticula_lateral(
        self, 
        familias: np.ndarray, 
        colores: list,
    ) -> None:
        """Genera la grilla 2D lateral de muestras indexadas y mapea sus eventos particulares."""
        assert self.fig is not None
        
        self.ax_reticula = self.fig.add_axes((0.02, 0.40, 0.24, 0.52))
        self.ax_reticula.axis('off')

        assert self.ax_reticula is not None
        
        n_familias = len(familias)
        n_columnas = 8  
        n_filas = int(np.ceil(n_familias / n_columnas))
        
        self.ax_reticula.set_xlim(-0.5, n_columnas - 0.5)
        self.ax_reticula.set_ylim(n_filas - 0.5, -0.5)  
        self.ax_reticula.set_aspect('equal', adjustable='box')
        
        mapeo = {}
        for idx in range(n_familias):
            f, c = idx // n_columnas, idx % n_columnas
            self.ax_reticula.scatter(c, f, c=[colores[idx]], s=150, marker='o', edgecolors='black', linewidths=0.5)
            mapeo[(c, f)] = idx
        
        self._mapeo_celdas = mapeo

    def _calcular_mapa_residuos_estatico(self, estado: EstadoImagen, centros: np.ndarray) -> np.ndarray:
        """Calcula una sola vez el mapa de residuos HD nativo con fondo de contexto original."""
        alto, ancho = estado.alto, estado.ancho
        imagen_base_rgb = np.where(np.isnan(estado.rgb), 1.0, estado.rgb)
        lienzo_plano_rgb = imagen_base_rgb.reshape(-1, 3).copy()
        
        if estado.mascara is not None:
            mascara_plana = estado.mascara.flatten().astype(bool)
        else:
            mascara_plana = ~np.isnan(estado.rgb.reshape(-1, 3)).any(axis=1)
            
        datos_validos = estado.lab_plano[mascara_plana]
        
        if len(datos_validos) == 0 or len(centros) == 0:
            return (imagen_base_rgb * 255).astype(np.uint8)

        asignacion = np.full(len(datos_validos), -1, dtype=int)
        r_L, r_C, r_T = self.tolerancias["L"]/2, self.tolerancias["C"]/2, self.tolerancias["T"]/2

        for idx, centro in enumerate(centros):
            uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centro)
            da = datos_validos[:, 1] - centro[1]
            db = datos_validos[:, 2] - centro[2]
            dL = (datos_validos[:, 0] - centro[0]) / r_L
            dC = (da * uR_a + db * uR_b) / r_C
            dT = (da * uT_a + db * uT_b) / r_T
            asignacion[(dL**2 + dC**2 + dT**2) <= 1.0] = idx

        # Coloreamos por el color original de la imagen si está cubierto, y rojo si quedó huérfano
        colores_fachada = estado.rgb_plano[mascara_plana].copy()
        colores_fachada[asignacion == -1] = [1.0, 0.0, 0.0]  # Rojo Neón

        lienzo_plano_rgb[mascara_plana] = colores_fachada
        return lienzo_plano_rgb.reshape((alto, ancho, 3))

# Función de formateo unificada regional (punto para miles, coma para decimales)
def f_num(v, decimales: int = 2) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", ".")
    elif isinstance(v, (float, np.floating)):
        # Formateamos con el estándar inglés usando los decimales pedidos
        cadena = f"{v:,.{decimales}f}"
        # Intercambiamos los signos de forma segura usando un marcador temporal
        return cadena.replace(",", "X").replace(".", ",").replace("X", ".")
    return str(v)
        