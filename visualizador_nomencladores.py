import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color
from typing import Dict, Any, cast

def visualizar_nomenclador_avanzado() -> None:
    ruta_json: str = os.path.join("modulos", "tabla_colores.json")
    if not os.path.exists(ruta_json):
        raise FileNotFoundError(f"No se encontró '{ruta_json}'. Ejecutá el compilador primero.")

    with open(ruta_json, "r", encoding="utf-8") as f:
        tabla_colores: Dict[str, Any] = json.load(f)

    # 1. Extraer datos del JSON
    nombres: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    colores_rgb: list[tuple[float, float, float]] = []

    for nombre, datos in tabla_colores.items():
        nombres.append(nombre)
        l_val, a_val, b_val = datos["lab"]
        r, g, b = datos["rgb_original"]
        xs.append(a_val)
        ys.append(b_val)
        zs.append(l_val)
        colores_rgb.append((r / 255.0, g / 255.0, b / 255.0))

    num_muestras = len(xs)
    TAMANIO_BASE = 50
    TAMANIO_EXTREMO = 150  # Tamaño grande destacado si coincide con un extremo RGB
    
    # Inicializamos el array de tamaños individuales
    tamaños = np.full(num_muestras, TAMANIO_BASE)

    # Definimos el diccionario de extremos puros (0-255) para comparar
    vertices_rgb = {
        (0, 0, 0): "Negro",
        (255, 255, 255): "Blanco",
        (255, 0, 0): "Rojo",
        (0, 255, 0): "Verde",
        (0, 0, 255): "Azul",
        (255, 255, 0): "Amarillo",
        (0, 255, 255): "Cian",
        (255, 0, 255): "Magenta"
    }

    # Guardamos los textos que vamos a imprimir fijas en la escena después de crear el scatter
    etiquetas_a_imprimir = []

    # Recorremos tus muestras cargadas para verificar coincidencias de extremos
    for idx, (nombre, datos) in enumerate(tabla_colores.items()):
        rgb_tupla = tuple(datos["rgb_original"])
        
        if rgb_tupla in vertices_rgb:
            # Si coincide, la muestra física del catálogo se agranda
            tamaños[idx] = TAMANIO_EXTREMO
            # Guardamos la info para meter el label fijo indicando su coordenada espacial
            etiquetas_a_imprimir.append((xs[idx], ys[idx], zs[idx], vertices_rgb[rgb_tupla]))

    # 2. Configurar la escena 3D de Matplotlib (Única inicialización de figura)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar el scatter único
    sc = ax.scatter(
        xs, ys, zs=np.array(zs).tolist(),
        c=colores_rgb, s=cast(Any, tamaños.tolist()), marker='o', 
        edgecolors='black', linewidths=0.8, alpha=0.9, picker=True
    )

    # Imprimir los labels fijos de texto sobre los mismos puntos agrandados correspondientes
    for x_l, y_l, z_l, texto in etiquetas_a_imprimir:
        ax.text(x_l, y_l, z_l + 3, texto, color='black', fontsize=8, weight='bold', ha='center', zorder=15)

    # 3. Configuración de límites de la interfaz
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_zlim(0, 100)
    ax.set_xlabel('Eje a* (Verde -> Rojo)')
    ax.set_ylabel('Eje b* (Azul -> Amarillo)')
    ax.set_zlabel('Luminosidad L*')
    ax.set_title(f'Espacio CIELAB - Catálogo Morfológico ({len(tabla_colores)} Colores)\nFUENTE: Anexo Colores / https://es.wikipedia.org/wiki/Anexo:Colores')

    # 4. Crear el cartel flotante (Hover) para el cursor
    hover_box = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.95, lw=1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
    )
    hover_box.set_visible(False)

    # Variables de tracking interactivo para los eventos de la UI
    color_actual_idx = None
    mostrando_hover = False

    def actualizar_hover(idx: int, x_cursor: float, y_cursor: float) -> None:
        """Modifica la posición y el texto del cartel flotante."""
        hover_box.xy = (x_cursor, y_cursor)
        text = f"Color: {nombres[idx]}\nL*: {zs[idx]:.1f} | a*: {xs[idx]:.1f} | b*: {ys[idx]:.1f}"
        hover_box.set_text(text)
        bbox = hover_box.get_bbox_patch()
        if bbox is not None:
            bbox.set_facecolor(colores_rgb[idx])

        texto_color = "black" if zs[idx] > 40 else "white"
        hover_box.set_color(texto_color)

    def on_hover(event) -> None:
        """Manejador de movimiento de cursor de Matplotlib."""
        nonlocal color_actual_idx, mostrando_hover
        
        if event.inaxes == ax:
            contiene, detalles = sc.contains(event)
            if contiene:
                idx = detalles["ind"][0]
                color_actual_idx = idx
                mostrando_hover = True
                
                x_p, y_p = event.x, event.y
                coordenadas_pantalla = ax.figure.transSubfigure.inverted().transform((x_p, y_p))
                hover_box.xy = (float(coordenadas_pantalla[0]), float(coordenadas_pantalla[1]))                
                actualizar_hover(idx, event.xdata, event.ydata)
                hover_box.set_visible(True)
                fig.canvas.draw_idle()
                return

        if mostrando_hover:
            hover_box.set_visible(False)
            mostrando_hover = False
            fig.canvas.draw_idle()

    # Vincular evento de movimiento del mouse
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    
    ax.view_init(elev=25, azim=-55)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualizar_nomenclador_avanzado()