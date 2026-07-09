import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation
from skimage import color
from typing import Any, cast
from modulos.modelo_color_bk import AgrupadorColor, ContenedorMuestrasColor


class VentanaDetalleColor:
    """
    Manejador persistente para mostrar el análisis detallado y la mancha 
    espacial de la familia analizada en resolución nativa.
    """
    def __init__(self) -> None:
        self.fig: Figure | None = None
        self.ax_imagen_origen: Axes | None = None
        self.ax_segmentada: Axes | None = None
        self.ax_color_centro: Axes | None = None
        self.ax_color_promedio: Axes | None = None
        self.ax_color_pico: Axes | None = None
        self.activa: bool = False

    def inicializar(self) -> None:
        """Inicializa la ventana con la grilla de visualización base."""
        figura = plt.figure(figsize=(10, 9))
        if not isinstance(figura, Figure): return
        self.fig = figura
        
        self.ax_imagen_origen = self.fig.add_subplot(2, 3, 1)
        self.ax_segmentada = self.fig.add_subplot(2, 3, 2)
        self.ax_color_centro = self.fig.add_subplot(2, 3, 4)
        self.ax_color_promedio = self.fig.add_subplot(2, 3, 5)
        self.ax_color_pico = self.fig.add_subplot(2, 3, 6)
        
        ejes: list[Axes] = [
            self.ax_imagen_origen, self.ax_segmentada, 
            self.ax_color_centro, self.ax_color_promedio, self.ax_color_pico
        ]
        
        for ax in ejes:
            ax.axis('off')
            
        self.ax_imagen_origen.text(0.5, 0.5, "Seleccione una familia\nen el gráfico 3D", 
                                   ha='center', va='center', color='gray', fontsize=12)
        
        self.fig.tight_layout()
        self.activa = True
        
        # REEMPLAZÁ LA LÍNEA DEL EVENTO DE CIERRE POR ESTA:
        def _al_cerrar_ficha(event: Any) -> None:
            self.activa = False
            # Cerramos todas las ventanas de forma segura rompiendo el bucle recursivo
            for num in plt.get_fignums():
                fig_obj = plt.figure(num)
                if fig_obj != self.fig:
                    plt.close(fig_obj)

        self.fig.canvas.mpl_connect('close_event', _al_cerrar_ficha)
  
    def _on_close(self, event: Any) -> None:
        self.activa = False

    def actualizar(
        self, 
        idx: int, 
        centro_lab: np.ndarray, 
        rgb_centro: np.ndarray, 
        indices_pixeles_familia: np.ndarray, 
        anclas: np.ndarray, 
        contexto_imagen: ContenedorMuestrasColor,
        nomenclador: Any | None = None
    ) -> None:
        """Actualiza todos los paneles con la información de la elipsoide seleccionada."""
        if not self.activa or self.fig is None:
            return
            
        assert self.ax_imagen_origen is not None
        assert self.ax_segmentada is not None
        assert self.ax_color_centro is not None
        assert self.ax_color_promedio is not None
        assert self.ax_color_pico is not None

        # Desempaquetamos los atributos de manera interna, limpia y localizada
        alto_real = contexto_imagen.alto_real
        ancho_real = contexto_imagen.ancho_real
        lab_total = contexto_imagen.lab_total
        rgb_completo = contexto_imagen.rgb_total
        imagen_original_pil = contexto_imagen.imagen_pil

        # 1. Buscar nombres en el nomenclador para cada bloque si está disponible
        nombre_centro = "Desconocido"
        nombre_promedio = "Desconocido"
        nombre_pico = "Desconocido"
        
        if nomenclador is not None:
            nombre_centro = nomenclador.obtener_nombre(centro_lab)

        pixeles_f_lab = lab_total[indices_pixeles_familia]

        # 2. Cálculo del Promedio Real y su nombre
        if len(pixeles_f_lab) > 0:
            promedio_lab = np.mean(pixeles_f_lab, axis=0)
            rgb_promedio = np.clip(color.lab2rgb(promedio_lab.reshape(1, 1, 3)).flatten(), 0, 1)
            if nomenclador is not None:
                nombre_promedio = nomenclador.obtener_nombre(promedio_lab)
        else:
            promedio_lab = centro_lab
            rgb_promedio = rgb_centro
            nombre_promedio = nombre_centro

        # 3. Cálculo del Pico de Densidad (Ancla) y su nombre
        if len(anclas) > 0 and len(pixeles_f_lab) > 0:
            distancias_anclas = np.linalg.norm(anclas - centro_lab, axis=1)
            idx_ancla_pico = np.argmin(distancias_anclas)
            lab_pico = anclas[idx_ancla_pico]
            rgb_pico = np.clip(color.lab2rgb(lab_pico.reshape(1, 1, 3)).flatten(), 0, 1)
            if nomenclador is not None:
                nombre_pico = nomenclador.obtener_nombre(lab_pico)
        else:
            lab_pico = centro_lab
            rgb_pico = rgb_centro
            nombre_pico = nombre_centro

        # Armar los textos con los nombres correspondientes arriba de los valores Lab
        lab_centro_txt = f"{nombre_centro}\n\nL*: {centro_lab[0]:.1f}\na*: {centro_lab[1]:.1f}\nb*: {centro_lab[2]:.1f}"
        lab_promedio_txt = f"{nombre_promedio}\n\nL*: {promedio_lab[0]:.1f}\na*: {promedio_lab[1]:.1f}\nb*: {promedio_lab[2]:.1f}"
        lab_pico_txt = f"{nombre_pico}\n\nL*: {lab_pico[0]:.1f}\na*: {lab_pico[1]:.1f}\nb*: {lab_pico[2]:.1f}"


        # Inicializamos el lienzo directamente en negro (0.0) como float32
        mancha_perfecta = np.ones((alto_real, ancho_real, 3), dtype=np.float32)
        
        # Generamos una vista aplanada (no consume memoria extra)
        mancha_perfecta_plana = mancha_perfecta.reshape(-1, 3)
        
        # Asignamos de forma directa usando los índices indexados
        if len(indices_pixeles_familia) > 0:
            mancha_perfecta_plana[indices_pixeles_familia] = rgb_completo[indices_pixeles_familia]
            
        # Volvemos a la estructura 2D original de manera instantánea
        mancha_perfecta = mancha_perfecta.reshape(alto_real, ancho_real, 3)


        # Renderizado de los paneles correspondientes
        self.ax_imagen_origen.clear()
        self.ax_imagen_origen.imshow(np.array(imagen_original_pil))
        self.ax_imagen_origen.set_title("Imagen Original", fontsize=11, fontweight='bold')
        self.ax_imagen_origen.axis('off')

        self.ax_segmentada.clear()
        self.ax_segmentada.imshow(mancha_perfecta)
        self.ax_segmentada.set_title(f"Mancha de la Familia #{idx}\n(Resolución Real s/ Blanco)", fontsize=11, fontweight='bold')
        self.ax_segmentada.axis('off')

        # Panel del Centroide
        self.ax_color_centro.clear()
        self.ax_color_centro.imshow([[np.clip(rgb_centro, 0, 1)]])
        self.ax_color_centro.axis('off')
        self.ax_color_centro.set_title("1. Centroide Geométrico\n(Media Teórica)", fontsize=10)
        self.ax_color_centro.text(0.5, -0.15, lab_centro_txt, transform=self.ax_color_centro.transAxes, 
                                  ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Panel del Promedio
        self.ax_color_promedio.clear()
        self.ax_color_promedio.imshow([[np.clip(rgb_promedio, 0, 1)]])
        self.ax_color_promedio.axis('off')
        self.ax_color_promedio.set_title("2. Promedio Real\nde la Familia", fontsize=10)
        self.ax_color_promedio.text(0.5, -0.15, lab_promedio_txt, transform=self.ax_color_promedio.transAxes, 
                                    ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Panel del Pico
        self.ax_color_pico.clear()
        self.ax_color_pico.imshow([[np.clip(rgb_pico, 0, 1)]])
        self.ax_color_pico.axis('off')
        self.ax_color_pico.set_title("3. Pico de Densidad\n(Ancla de la Familia)", fontsize=10)
        self.ax_color_pico.text(0.5, -0.15, lab_pico_txt, transform=self.ax_color_pico.transAxes, 
                                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Título principal de la ficha usando el nombre del centroide de la familia
        self.fig.suptitle(f"ANÁLISIS DE ATRIBUTOS: {nombre_centro.upper()} (FAMILIA #{idx})", fontsize=13, fontweight='bold')
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.show()


class VisualizadorEspacioColor:
    """
    Encapsula el entorno gráfico 3D de Matplotlib, renderiza los datos
    espaciales y gestiona las colisiones de eventos de mouse del usuario.
    """
    def __init__(self, tol_L: float, tol_C: float, tol_T: float) -> None:
        self.tol_L: float = tol_L
        self.tol_C: float = tol_C
        self.tol_T: float = tol_T
        
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
        self.nomenclador: Any | None = None
        self._texto_titulo_paleta: Any | None = None

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
        
        r_L = self.tol_L / 2
        r_C = self.tol_C / 2
        r_T = self.tol_T / 2
        
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
            "Espacio CIELAB (Elipsoides Rígidas Orientadas)\n"
            f"Archivo: {nombre_archivo}"
        )
        
        # Aplicamos el título al eje y forzamos el redibujado de la interfaz
        self.ax.set_title(titulo_formateado, fontsize=11, fontweight='normal', pad=15)
        self.fig.canvas.draw_idle()

    def configurar_titulo_paleta(self, cantidad_familias: int) -> None:
        """
        Permite actualizar de forma externa el título de la retícula lateral
        indicando el número de muestras encontradas.
        """
        if self.ax_reticula is None or self.fig is None:
            return
        
        if self._texto_titulo_paleta is not None:
            try:
                self._texto_titulo_paleta.remove()
            except Exception:
                pass
            
        titulo_reticula = f"Familias Encontradas: {cantidad_familias}"
        #self.ax_reticula.set_title(titulo_reticula, fontsize=10, pad=10, loc='left')
        self._texto_titulo_paleta = self.ax_reticula.text(
            -0.3, -0.8, 
            titulo_reticula, 
            fontsize=10, 
            fontweight='normal'
        )
        
        self.fig.canvas.draw_idle()

    def inicializar_grafico_3d(
        self, 
        lab_muestreo: np.ndarray, 
        rgb_muestreo: np.ndarray, 
        anclas: np.ndarray,
        paleta_mediana: np.ndarray,
        familias: np.ndarray,
        etiquetas_familias: np.ndarray,
        ventana_detalle: VentanaDetalleColor,
        contexto_imagen: ContenedorMuestrasColor,
        agrupador: Any,
        nomenclador: Any | None = None
    ) -> None:
        """Construye la visualización tridimensional CIELAB y mapea la interactividad."""
        figura = plt.figure(figsize=(10, 9))
        if not isinstance(figura, Figure):
            return
        self.fig = figura

        self.agrupador = agrupador
        self.nomenclador = nomenclador
        self.contexto_imagen = contexto_imagen

        self.datos_centros_lab = familias
        
        # 1. Configuración del entorno de ejes 3D y límites CIELAB
        self._configurar_ejes_3d()
        
        # 2. Renderizado de las capas de datos (Brutos, Anclas, Centros y Paleta)
        colores = [np.clip(color.lab2rgb(c.reshape(1, 1, 3)).flatten(), 0, 1) for c in familias]
        self.datos_colores_rgb = colores
        
        self._renderizar_capas_datos(lab_muestreo, rgb_muestreo, anclas, paleta_mediana, familias, etiquetas_familias, colores)
        
        # 3. Inicialización de los CheckButtons de control de capas
        self._inicializar_widgets_control(familias, colores)

        # 4. Construcción de la retícula interactiva lateral
        self._construir_reticula_lateral(familias, colores, ventana_detalle, contexto_imagen.imagen_pil, contexto_imagen.lab_total, contexto_imagen.rgb_total, anclas, contexto_imagen.alto_real, contexto_imagen.ancho_real)

        # 5. Conexión de eventos interactivos globales (Hover, Click y Cierre)
        self._conectar_eventos_globales(ventana_detalle, contexto_imagen.imagen_pil, contexto_imagen.lab_total, contexto_imagen.rgb_total, anclas, contexto_imagen.alto_real, contexto_imagen.ancho_real)

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
        anclas: np.ndarray, 
        paleta_mediana: np.ndarray, 
        familias: np.ndarray, 
        etiquetas_familias: np.ndarray, 
        colores: list
    ) -> None:
        """Dibuja cada una de las colecciones de puntos y geometrías en el espacio."""
        assert self.ax is not None
        
        # Capa 1: Datos Brutos de la imagen
        self.sc_muestras = self.ax.scatter(lab_muestreo[:, 1], lab_muestreo[:, 2], zs=lab_muestreo[:, 0].tolist(), 
                                           c=rgb_muestreo, s=5, alpha=0.02, label='Datos Brutos')

        # Capa 2: Anclas de densidad
        if len(anclas) > 0:
            self.sc_anclas = self.ax.scatter(anclas[:, 1], anclas[:, 2], zs=anclas[:, 0].tolist(), 
                                             c='yellow', s=50, marker='.', edgecolors='black', label='Anclas', zorder=10)

        # Capa 3: Paleta Mediana de soporte
        rgb_paleta = np.clip(color.lab2rgb(paleta_mediana.reshape(-1, 1, 3)).reshape(-1, 3), 0.0, 1.0)
        self.sc_paleta_mediana = self.ax.scatter(paleta_mediana[:, 1], paleta_mediana[:, 2], zs=paleta_mediana[:, 0].tolist(), 
                                                 c=rgb_paleta, s=3, alpha=0.8, label='Paleta Mediana')

        # Capa 4: Centros de familias con tamaños dinámicos por conteo
        ids_clusters, conteos = np.unique(etiquetas_familias, return_counts=True)
        conteos_validos = conteos[ids_clusters >= 0]
        
        tamanos_familias: Any = np.full(len(familias), 80.0, dtype=np.float64)
        if len(conteos_validos) > 0 and np.max(conteos_validos) != np.min(conteos_validos):
            conteos_raiz = np.sqrt(conteos_validos)
            min_r, max_r = np.sqrt(np.min(conteos_validos)), np.sqrt(np.max(conteos_validos))
            tamanos_familias = 1 + (conteos_raiz - min_r) / (max_r - min_r) * (300 - 1)

        self.sc_centros = self.ax.scatter(familias[:, 1], familias[:, 2], zs=familias[:, 0].tolist(),
                                          c=colores, s=tamanos_familias, marker='o', edgecolors='black', zorder=100, depthshade=False)

        # Capa 5: Elipsoides radiales locales
        self.lista_elipsoides = []
        for i, centro in enumerate(familias):
            self._dibujar_elipsoide_radial(centro, colores[i])

        # Seteo de visibilidad inicial por defecto
        estados = [True, False, True, False, False] # Muestras, Anclas, Paleta, Elipsoides, Familias
        if self.sc_muestras: self.sc_muestras.set_visible(estados[0])
        if self.sc_anclas: self.sc_anclas.set_visible(estados[1])
        if self.sc_paleta_mediana: self.sc_paleta_mediana.set_visible(estados[2])
        for elipsoide in self.lista_elipsoides: elipsoide.set_visible(estados[3])
        if self.sc_centros: self.sc_centros.set_visible(estados[4])

        # Inicialización de la anotación flotante para el Hover 3D
        self.anotacion = self.ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"), zorder=200)
        self.anotacion.set_visible(False)

    def _inicializar_widgets_control(self, familias: np.ndarray, colores: list) -> None:
        """Construye los CheckButtons nativos para alternar la visibilidad de las capas."""
        assert self.fig is not None
        from matplotlib.widgets import CheckButtons
        
        ax_check = self.fig.add_axes((0.02, 0.02, 0.22, 0.16))
        etiquetas = ["Ver Muestras", "Ver Anclas", "Paleta Mediana", "Ver Elipsoides", "Ver Familias"]
        estados_iniciales = [True, False, True, False, False]
        
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
        ventana_detalle: VentanaDetalleColor, 
        imagen_pil: Any, 
        lab_total: np.ndarray, 
        rgb_total: np.ndarray, 
        anclas: np.ndarray, 
        alto_real: int, 
        ancho_real: int
    ) -> None:
        """Genera la grilla 2D lateral de muestras indexadas y mapea sus eventos particulares."""
        assert self.fig is not None
        self.ax_reticula = self.fig.add_axes((0.02, 0.40, 0.24, 0.52))
        self.ax_reticula.axis('off')
        
        n_familias = len(familias)
        n_columnas = 8  
        n_filas = int(np.ceil(n_familias / n_columnas))
        
        self.ax_reticula.set_xlim(-0.5, n_columnas - 0.5)
        self.ax_reticula.set_ylim(n_filas - 0.5, -0.5)  
        self.ax_reticula.set_aspect('equal', adjustable='box')
        
        mapeo_celdas = {}
        scatters_reticula = []  
        
        for idx in range(n_familias):
            f, c = idx // n_columnas, idx % n_columnas
            sc = self.ax_reticula.scatter(c, f, c=[colores[idx]], s=150, marker='o', edgecolors='black', linewidths=0.5)
            scatters_reticula.append(sc)
            mapeo_celdas[idx] = (c, f)

        anotacion_reticula = self.ax_reticula.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", alpha=0.95, ec="gray", lw=0.5), zorder=200
        )
        anotacion_reticula.set_visible(False)

        def on_hover_reticula(event: Any) -> None:
            if not self.fig: return
            if event.inaxes == self.ax_reticula:
                encontrado = False
                for idx, sc in enumerate(scatters_reticula):
                    contiene, _ = sc.contains(event)
                    if contiene:
                        centro_lab = self.datos_centros_lab[idx]
                        color_rgb = self.datos_colores_rgb[idx]
                        nombre = self.nomenclador.obtener_nombre(centro_lab) if self.nomenclador else f"Familia #{idx}"
                        
                        luminancia = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
                        color_texto = "black" if luminancia > 0.5 else "white"
                        
                        anotacion_reticula.xy = (mapeo_celdas[idx][0], mapeo_celdas[idx][1])
                        anotacion_reticula.set_text(f"Familia #{idx}\n{nombre}\nL*={centro_lab[0]:.1f} a*={centro_lab[1]:.1f} b*={centro_lab[2]:.1f}")
                        anotacion_reticula.set_color(color_texto)
                        
                        bbox_patch = anotacion_reticula.get_bbox_patch()
                        if bbox_patch is not None: bbox_patch.set_facecolor(color_rgb.tolist())
                            
                        anotacion_reticula.set_visible(True)
                        encontrado = True
                        break
                if not encontrado and anotacion_reticula.get_visible(): anotacion_reticula.set_visible(False)
            elif anotacion_reticula.get_visible(): anotacion_reticula.set_visible(False)
            self.fig.canvas.draw_idle()

        def onclick_reticula(event: Any) -> None:
            if event.inaxes == self.ax_reticula and event.xdata is not None and event.ydata is not None:
                c_click, f_click = int(np.round(event.xdata)), int(np.round(event.ydata))
                idx_sel = next((idx for idx, (c, f) in mapeo_celdas.items() if c == c_click and f == f_click), -1)
                        
                if idx_sel != -1:
                    centro_lab = self.datos_centros_lab[idx_sel]
                    indices_f = self.agrupador.filtrar_pixeles_por_elipsoide(centro_lab, lab_total)

                    ventana_detalle.actualizar(
                        idx=idx, 
                        centro_lab=centro_lab, 
                        rgb_centro=self.datos_colores_rgb[idx_sel], 
                        anclas=anclas, 
                        indices_pixeles_familia=indices_f, 
                        contexto_imagen=self.contexto_imagen, 
                        nomenclador=self.nomenclador
                    )

        self.fig.canvas.mpl_connect('motion_notify_event', on_hover_reticula)
        self.fig.canvas.mpl_connect('button_press_event', onclick_reticula)

    def _conectar_eventos_globales(
        self, 
        ventana_detalle: VentanaDetalleColor, 
        imagen_pil: Any, 
        lab_total: np.ndarray, 
        rgb_total: np.ndarray, 
        anclas: np.ndarray, 
        alto_real: int, 
        ancho_real: int
    ) -> None:
        """Enlaza los callbacks interactivos principales del lienzo y de la ventana de la aplicación."""
        assert self.fig is not None
        self._conectar_eventos(ventana_detalle, imagen_pil, lab_total, rgb_total, anclas, alto_real, ancho_real)
        
        def cerrar_aplicacion_limpia(event: Any) -> None:
            for num in plt.get_fignums():
                fig_obj = plt.figure(num)
                if fig_obj != self.fig: plt.close(fig_obj)

        self.fig.canvas.mpl_connect('close_event', cerrar_aplicacion_limpia)
        
        def on_leave_figure(event: Any) -> None:
            if not self.fig: return
            # Forzar la ocultación de la anotación si el puntero abandona el canvas
            if self.anotacion and self.anotacion.get_visible():
                self.anotacion.set_visible(False)
                self.fig.canvas.draw_idle()
                
        self.fig.canvas.mpl_connect('figure_leave_event', on_leave_figure)
    
    def _conectar_eventos(self, ventana_detalle: VentanaDetalleColor, imagen_pil: Any, lab_total: np.ndarray, rgb_total: np.ndarray, anclas: np.ndarray, alto_real: int, ancho_real: int) -> None:
        
        def on_hover(event: Any) -> None:
            if self.fig is None or self.sc_centros is None or self.anotacion is None:
                return
            if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                # Usar el método nativo de Matplotlib para detectar colisión
                cont, ind = self.sc_centros.contains(event)
                if cont:
                    indices_candidatos = ind.get("ind", [])
                    if len(indices_candidatos) > 0:
                        idx = int(indices_candidatos[0])
                        centro_lab = self.datos_centros_lab[idx]
                        rgb_actual = self.datos_colores_rgb[idx]
                        
                        # Buscar el nombre usando self de la instancia de forma estricta
                        nombre_color = "Desconocido"
                        if self.nomenclador is not None:
                            nombre_color = self.nomenclador.obtener_nombre(centro_lab)
                        
                        self.anotacion.set_text(
                            f"Familia #{idx}\n"
                            f"{nombre_color}\n"
                            f"L*: {centro_lab[0]:.1f} | a*: {centro_lab[1]:.1f} | b*: {centro_lab[2]:.1f}"
                        )
                        
                        # Posicionar el cartel usando las coordenadas del evento 2D en los ejes
                        self.anotacion.xy = (event.xdata, event.ydata)
                        
                        # Cambiar el color de fondo de manera directa y segura sin .tolist()
                        bbox = self.anotacion.get_bbox_patch()
                        if bbox is not None:
                            bbox.set_facecolor(rgb_actual.tolist())
                            
                        texto_color = "black" if centro_lab[0] > 40 else "white"
                        self.anotacion.set_color(texto_color)
                        
                        self.anotacion.set_visible(True)
                        self.fig.canvas.draw_idle()
                else:
                    if self.anotacion.get_visible():
                        self.anotacion.set_visible(False)
                        self.fig.canvas.draw_idle()

        def on_click(event: Any) -> None:
            if self.sc_centros is None or self.fig is None:
                return
            if event.inaxes == self.ax and event.button == 1 and ventana_detalle.activa:
                # El clic nativo usa el mismo contendor de eventos
                cont, ind = self.sc_centros.contains(event)
                if cont:
                    indices_candidatos = ind.get("ind", [])
                    if len(indices_candidatos) > 0:
                        idx = int(indices_candidatos[0])
                        centro_lab = self.datos_centros_lab[idx]
                        rgb_color = self.datos_colores_rgb[idx]
                        
                        indices_pixeles_familia = self.agrupador.filtrar_pixeles_por_elipsoide(centro_lab, lab_total)
                        
                        cantidad = len(indices_pixeles_familia)
                        porcentaje = (cantidad / len(lab_total)) * 100
                        print(f"\n[ANÁLISIS TOTAL] Familia #{idx}:")
                        print(f" -> Cantidad de píxeles reales: {cantidad}")
                        print(f" -> Cobertura de la imagen: {porcentaje:.2f}%")
                        
                        # Pasar self.nomenclador garantizando que la ventana hija lo reciba
                        ventana_detalle.actualizar(
                                idx=idx, 
                                centro_lab=centro_lab, 
                                rgb_centro=rgb_color, 
                                anclas=anclas, 
                                indices_pixeles_familia=indices_pixeles_familia, 
                                contexto_imagen=self.contexto_imagen, 
                                nomenclador=self.nomenclador
                            )
                        
        if self.fig is not None:
            self.fig.canvas.mpl_connect('motion_notify_event', on_hover)
            self.fig.canvas.mpl_connect('button_press_event', on_click)
    