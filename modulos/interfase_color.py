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
from modulos.modelo_color import AgrupadorColor, EstadoImagen


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
        self.activa: bool = False

    def inicializar(self) -> None:
        """Inicializa la ventana con una grilla geométrica fija y de tipo estricto."""
        figura = plt.figure(figsize=(12, 9))
        if not isinstance(figura, Figure): return
        self.fig = figura
        
        # Usamos tuplas explícitas y cast a Axes para satisfacer a Pylance
        self.ax_imagen_origen = cast(Axes, self.fig.add_axes((0.05, 0.52, 0.28, 0.40)))
        self.ax_todas_las_familias = cast(Axes, self.fig.add_axes((0.36, 0.52, 0.28, 0.40)))
        self.ax_segmentada = cast(Axes, self.fig.add_axes((0.67, 0.52, 0.28, 0.40)))
        
        self.ax_color_centro = cast(Axes, self.fig.add_axes((0.08, 0.18, 0.22, 0.22)))
        self.ax_color_promedio = cast(Axes, self.fig.add_axes((0.39, 0.18, 0.22, 0.22)))
        self.ax_color_pico = cast(Axes, self.fig.add_axes((0.70, 0.18, 0.22, 0.22)))
        
        ejes: list[Axes] = [
            self.ax_imagen_origen, self.ax_segmentada, self.ax_todas_las_familias,
            self.ax_color_centro, self.ax_color_promedio, self.ax_color_pico
        ]
        
        for ax in ejes:
            ax.axis('off')

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
        assert self.ax_todas_las_familias is not None # <--- AGREGAR ESTO
        assert self.ax_color_centro is not None
        assert self.ax_color_promedio is not None
        assert self.ax_color_pico is not None

        alto_real = estado_contexto.alto
        ancho_real = estado_contexto.ancho
        lab_total = estado_contexto.lab_plano
        rgb_completo = estado_contexto.rgb_plano

        nombre_centro = "Desconocido"
        nombre_promedio = "Desconocido"
        nombre_pico = "Desconocido"
        
        if nomenclador is not None:
            nombre_centro = nomenclador.obtener_nombre(centro_lab)

        pixeles_f_lab = lab_total[indices_pixeles_familia]

        if len(pixeles_f_lab) > 0:
            promedio_lab = np.mean(pixeles_f_lab, axis=0)
            rgb_promedio = np.clip(color.lab2rgb(promedio_lab.reshape(1, 1, 3)).flatten(), 0, 1)
            if nomenclador is not None:
                nombre_promedio = nomenclador.obtener_nombre(promedio_lab)
        else:
            promedio_lab = centro_lab
            rgb_promedio = rgb_centro
            nombre_promedio = nombre_centro

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

        lab_centro_txt = f"{nombre_centro}\n\nL*: {centro_lab[0]:.1f}\na*: {centro_lab[1]:.1f}\nb*: {centro_lab[2]:.1f}"
        lab_promedio_txt = f"{nombre_promedio}\n\nL*: {promedio_lab[0]:.1f}\na*: {promedio_lab[1]:.1f}\nb*: {promedio_lab[2]:.1f}"
        lab_pico_txt = f"{nombre_pico}\n\nL*: {lab_pico[0]:.1f}\na*: {lab_pico[1]:.1f}\nb*: {lab_pico[2]:.1f}"

        # 1. GENERACIÓN DE MANCHA PURA (Fondo Blanco)
        mancha_perfecta = np.ones((alto_real, ancho_real, 3), dtype=np.float32)
        mancha_perfecta_plana = mancha_perfecta.reshape(-1, 3)
        
        if len(indices_pixeles_familia) > 0:
            mancha_perfecta_plana[indices_pixeles_familia] = rgb_completo[indices_pixeles_familia]
            
        mancha_perfecta = mancha_perfecta.reshape(alto_real, ancho_real, 3)

        # 2. GENERACIÓN DE MANCHA SUPERPUESTA (Fachada original atenuada de fondo)
        imagen_origen_limpia = np.nan_to_num(estado_contexto.rgb, nan=1.0)
        imagen_combinada = imagen_origen_limpia.copy()
        
        # Atenuamos toda la imagen original al 30% para que resalte la ubicación real de la mancha
        mascara_resto = np.ones(alto_real * ancho_real, dtype=bool)
        if len(indices_pixeles_familia) > 0:
            mascara_resto[indices_pixeles_familia] = False
        mascara_resto = mascara_resto.reshape(alto_real, ancho_real)
        imagen_combinada[mascara_resto] *= 0.3

        # Renderizado de ejes
        self.ax_imagen_origen.clear()
        self.ax_imagen_origen.imshow(np.nan_to_num(estado_contexto.rgb, nan=1.0))
        self.ax_imagen_origen.set_title("Imagen Original", fontsize=11, fontweight='normal')
        self.ax_imagen_origen.axis('off')

        self.ax_segmentada.clear()
        self.ax_segmentada.imshow(mancha_perfecta)
        self.ax_segmentada.set_title(f"Mancha de la Familia #{idx + 1}", fontsize=11, fontweight='normal')
        self.ax_segmentada.axis('off')

        # NUEVO: Renderizado del mapa total unificado
        self.ax_todas_las_familias.clear()
        if self.mapa_total_familias is not None:
            self.ax_todas_las_familias.imshow(self.mapa_total_familias)
        self.ax_todas_las_familias.set_title("Reconstrucción Total", fontsize=11, fontweight='normal')
        self.ax_todas_las_familias.axis('off')

        self.ax_color_centro.clear()
        self.ax_color_centro.imshow([[np.clip(rgb_centro, 0, 1)]])
        self.ax_color_centro.axis('off')
        self.ax_color_centro.set_title("1. Centroide Geométrico\n(Media Teórica)", fontsize=10)
        self.ax_color_centro.text(0.5, -0.15, lab_centro_txt, transform=self.ax_color_centro.transAxes, 
                                  ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        self.ax_color_promedio.clear()
        self.ax_color_promedio.imshow([[np.clip(rgb_promedio, 0, 1)]])
        self.ax_color_promedio.axis('off')
        self.ax_color_promedio.set_title("2. Promedio Real\nde la Familia", fontsize=10)
        self.ax_color_promedio.text(0.5, -0.15, lab_promedio_txt, transform=self.ax_color_promedio.transAxes, 
                                    ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        self.ax_color_pico.clear()
        self.ax_color_pico.imshow([[np.clip(rgb_pico, 0, 1)]])
        self.ax_color_pico.axis('off')
        self.ax_color_pico.set_title("3. Pico de Densidad\n(Ancla de la Familia)", fontsize=10)
        self.ax_color_pico.text(0.5, -0.15, lab_pico_txt, transform=self.ax_color_pico.transAxes, 
                                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        self.fig.suptitle(f"FAMILIA #{idx + 1}: {nombre_centro}", fontsize=11, fontweight='normal')
        self.fig.canvas.draw_idle()
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.show()


class VisualizadorEspacioColor:
    """
    Encapsula el entorno gráfico 3D de Matplotlib, renderiza los datos
    espaciales y gestiona las colisiones de eventos de mouse del usuario.
    """
    def __init__(self, tolerancias: dict[str, float] | None = None) -> None:
        self.tolerancias = tolerancias or {"L": 20.0, "C": 20.0, "T": 10.0}
        
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
        if (self._agrupador is None or 
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
            
            indices_pixeles_familia = self._agrupador.filtrar_pixeles_por_elipsoide(
                centro_lab, self.lab_total, tolerancias=self.tolerancias
            )

            if self.estado_original.mascara is not None:
                mascara_plana = self.estado_original.mascara.flatten()
                indices_pixeles_familia = indices_pixeles_familia[mascara_plana[indices_pixeles_familia].astype(bool)]

            cantidad = len(indices_pixeles_familia)
            
            # CORRECCIÓN DE ÁREA ÚTIL: Eliminamos el sesgo del lienzo de fondo negro
            if self.estado_original.mascara is not None:
                total_util_hd = np.sum(self.estado_original.mascara)
            else:
                total_util_hd = np.sum(~np.isnan(self.lab_total).any(axis=1))
                
            porcentaje = (cantidad / total_util_hd) * 100
            
            # LOG EN CONSOLA EN BASE 1
            print(f"\n[ANÁLISIS TOTAL HD] Familia #{idx + 1}:")
            print(f" -> Cantidad de píxeles reales: {cantidad}")
            print(f" -> Cobertura real sobre fachada nativa: {porcentaje:.2f}%")

            anclas_envio = self.anclas if self.anclas is not None else np.empty((0, 3))

            # ENVIAMOS LOS DATOS: La ventana se encarga de sus propios paneles de manera segura
            self._ventana_detalle.actualizar(
                idx=idx,
                centro_lab=centro_lab,
                rgb_centro=rgb_color,
                anclas=anclas_envio,
                indices_pixeles_familia=indices_pixeles_familia,
                estado_contexto=self._estado_contexto,
                nomenclador=self._nomenclador
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
        configuracion: dict | None = None, 
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

        lineas_texto = []
        
        # BLOQUE 1: COBERTURA Y MUESTREO (Independiente - Requiere estado_original)
        if estado_original is not None:
            lineas_texto.append("─────────────────────────────────────────")
            lineas_texto.append("COBERTURA Y MUESTREO")
            
            alto_real, ancho_real = estado_original.alto, estado_original.ancho
            total_muestras_absoluto = alto_real * ancho_real
            
            if estado_original.mascara is not None:
                total_muestras_utiles = int(np.sum(estado_original.mascara))
                total_muestras_enmascaradas = total_muestras_absoluto - total_muestras_utiles
            else:
                total_muestras_utiles = total_muestras_absoluto
                total_muestras_enmascaradas = 0
                
            pixeles_cubiertos = 0
            mascara_plana = estado_original.mascara.flatten() if estado_original.mascara is not None else None
            
            for centro in self.datos_centros_lab:
                indices = self.agrupador.filtrar_pixeles_por_elipsoide(centro, self.lab_total, tolerancias=self.tolerancias)
                if mascara_plana is not None:
                    indices = indices[mascara_plana[indices].astype(bool)]
                pixeles_cubiertos += len(indices)
                
            # Cálculos de proporciones del muestreo respecto al total absoluto
            pct_util_del_total = (total_muestras_utiles / total_muestras_absoluto) * 100 if total_muestras_absoluto > 0 else 0.0
            pct_enmascarado_del_total = (total_muestras_enmascaradas / total_muestras_absoluto) * 100 if total_muestras_absoluto > 0 else 0.0

            # Cálculos de coberturas (Útil vs Absoluta)
            porcentaje_cobertura_util = (pixeles_cubiertos / total_muestras_utiles) * 100 if total_muestras_utiles > 0 else 0.0
            porcentaje_cobertura_total = (pixeles_cubiertos / total_muestras_absoluto) * 100 if total_muestras_absoluto > 0 else 0.0
            porcentaje_ruido = 100.0 - porcentaje_cobertura_util

            lineas_texto.append(f" - Muestras Totales: {f_num(total_muestras_absoluto)} px ({ancho_real} x {alto_real})")
            lineas_texto.append(f" - Muestras Utiles: {f_num(total_muestras_utiles)} px ({f_num(pct_util_del_total)}%)")
            lineas_texto.append(f" - Muestras Enmascaradas: {f_num(total_muestras_enmascaradas)} px ({f_num(pct_enmascarado_del_total)}%)")
            lineas_texto.append(f" - Cobertura de Paleta s/ Area Total: {f_num(porcentaje_cobertura_total)}%")
            lineas_texto.append(f" - Cobertura de Paleta s/ Area Util: {f_num(porcentaje_cobertura_util)}%")
            lineas_texto.append(f" - Ruido sin clasificar: {f_num(porcentaje_ruido)}%")
            lineas_texto.append("─────────────────────────────────────────")

        # BLOQUE 2: PARÁMETROS DEL ALGORITMO (Independiente - Requiere configuracion)
        if configuracion is not None:
            # Si no se renderizó el bloque anterior, agregamos la línea divisoria inicial
            if estado_original is None:
                lineas_texto.append("─────────────────────────────────────────")
                
            lineas_texto.append("PARÁMETROS DEL ALGORITMO")
            lineas_texto.append(f" - Ancho de Reducción: {f_num(configuracion['ancho_analisis'])} px")
            
            tol = configuracion["tolerancias"]
            tol_l = f_num(tol['L'], decimales=1)
            tol_c = f_num(tol['C'], decimales=1)
            tol_t = f_num(tol['T'], decimales=1)

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
        self.estado_analisis = estado_analisis
        self.estado_original = estado_original
        
        self.datos_centros_lab = familias
        
        # 1. Configuración del entorno de ejes 3D y límites CIELAB
        self._configurar_ejes_3d()
        
        # 2. Renderizado de las capas de datos (Brutos, Anclas, Centros y Paleta)
        colores = [np.clip(color.lab2rgb(c.reshape(1, 1, 3)).flatten(), 0, 1) for c in familias]
        self.datos_colores_rgb = colores

        # 3. Inicialización de los CheckButtons de control de capas
        self._inicializar_widgets_control(familias, colores)

        # 4. Construcción de la retícula interactiva lateral
        # --- SOLUCIÓN PYLANCE & MEMORIA: Extraemos las propiedades de estado_original directamente ---
        self._construir_reticula_lateral(
            familias=familias, 
            colores=colores, 
            ventana_detalle=ventana_detalle, 
            imagen_pil=None, # Ya no hace falta pasar PIL acá
            lab_total=estado_original.lab_plano, 
            rgb_total=estado_original.rgb_plano, 
            anclas=anclas, 
            alto_real=estado_original.alto, 
            ancho_real=estado_original.ancho
        )

        # --- SOLUCIÓN PYLANCE: Definimos las variables locales antes de usarlas en las capas ---
        lab_muestreo = estado_analisis.lab_plano
        rgb_muestreo = estado_analisis.rgb_plano
        matriz_paleta_lab = paleta_mediana.lab_plano
        self.lab_total = estado_original.lab_plano

        self._renderizar_capas_datos(
            lab_muestreo=lab_muestreo,
            rgb_muestreo=rgb_muestreo,
            anclas=anclas,
            paleta_mediana=matriz_paleta_lab,
            familias=familias,
            etiquetas_familias=etiquetas_familias,
            colores=colores
        )

        # 5. Conexión de eventos interactivos globales (Hover, Click y Cierre)
        self._ventana_detalle = ventana_detalle
        self._agrupador = agrupador
        self.anclas = anclas
        self._estado_contexto = estado_original     # Guardamos el estado para evitar duplicar referencias
        self._nomenclador = nomenclador

        # --- PRECÁLCULO DEL MAPA GENERAL DE TODAS LAS FAMILIAS SOBRE BLANCO ---
        alto_real = estado_original.alto
        ancho_real = estado_original.ancho
        mapa_total = np.ones((alto_real, ancho_real, 3), dtype=np.float32)
        mapa_total_plano = mapa_total.reshape(-1, 3)
        mascara_plana = estado_original.mascara.flatten() if estado_original.mascara is not None else None
        
        for centro in familias:
            indices = agrupador.filtrar_pixeles_por_elipsoide(centro, self.lab_total, tolerancias=self.tolerancias)
            if mascara_plana is not None:
                indices = indices[mascara_plana[indices].astype(bool)]
            if len(indices) > 0:
                mapa_total_plano[indices] = estado_original.rgb_plano[indices]
                
        self._ventana_detalle.mapa_total_familias = mapa_total.reshape(alto_real, ancho_real, 3)
        # ----------------------------------------------------------------------

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
            indices = self.agrupador.filtrar_pixeles_por_elipsoide(centro, self.lab_total, tolerancias=self.tolerancias)
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

    def _inicializar_widgets_control(self, familias: np.ndarray, colores: list) -> None:
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
        ventana_detalle: VentanaDetalleColor, 
        imagen_pil: Any, 
        lab_total: np.ndarray, 
        rgb_total: np.ndarray, 
        anclas: np.ndarray | None, 
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

