import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from collections import Counter
from matplotlib.collections import LineCollection


class AnalizadorAdyacenciaUnicaImagen:
    def __init__(self) -> None:
        pass

    def procesar_imagen_rgb(self, imagen_rgba: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
        """
        Identifica colores únicos ignorando por completo los píxeles transparentes.
        Retorna la matriz indexada, la paleta y una máscara booleana de píxeles válidos.
        """
        alto, ancho, canales = imagen_rgba.shape
        
        # Si tiene canal alfa, extraemos los píxeles que no son transparentes (alfa > 0)
        if canales == 4:
            mascara_valida = imagen_rgba[:, :, 3] > 0
            rgb_puro = imagen_rgba[:, :, :3]
        else:
            mascara_valida = np.ones((alto, ancho), dtype=bool)
            rgb_puro = imagen_rgba

        # Inicializamos el mapa indexado con un ID de fondo (-1) para el área transparente
        mapa_indexado_2d = np.full((alto, ancho), -1, dtype=int)
        
        # Filtramos solo los píxeles visibles
        pixeles_validos = rgb_puro[mascara_valida]
        
        # Encontramos los colores únicos reales
        colores_unicos, inversa = np.unique(pixeles_validos, axis=0, return_inverse=True)
        
        # Asignamos los IDs correspondientes solo a la zona visible
        mapa_indexado_2d[mascara_valida] = inversa
        
        # Construimos la paleta normalizada sin meter el fondo transparente
        paleta_original = {}
        for idx, color in enumerate(colores_unicos):
            paleta_original[idx] = list(color / 255.0)
            
        return mapa_indexado_2d, paleta_original, mascara_valida

    def construir_grafo_adyacencia(self, mapa_indexado_2d: np.ndarray) -> nx.Graph:
        grafo = nx.Graph()
        
        # Registrar nodos válidos (ignorando el fondo transparente -1)
        ids_unicos = np.unique(mapa_indexado_2d)
        for idx in ids_unicos:
            if idx != -1:
                grafo.add_node(idx)

        alto, ancho = mapa_indexado_2d.shape
        contador_fronteras = Counter()

        for f in range(alto - 1):
            for c in range(ancho - 1):
                p_actual = mapa_indexado_2d[f, c]
                p_derecha = mapa_indexado_2d[f, c + 1]
                p_abajo = mapa_indexado_2d[f + 1, c]
                
                # Solo registramos adyacencias si ambos píxeles pertenecen a colores de la fachada
                if p_actual != -1 and p_derecha != -1 and p_actual != p_derecha:
                    par = tuple(sorted((p_actual, p_derecha)))
                    contador_fronteras[par] += 1
                if p_actual != -1 and p_abajo != -1 and p_actual != p_abajo:
                    par = tuple(sorted((p_actual, p_abajo)))
                    contador_fronteras[par] += 1

        for (c1, c2), peso in contador_fronteras.items():
            grafo.add_edge(c1, c2, weight=peso)
            
        return grafo

    def visualizar_poc(self, ruta_imagen: str) -> None:
        if not os.path.exists(ruta_imagen):
            print(f"[Error] No se encontró el archivo en: {ruta_imagen}")
            return

        # 1. Cargar la imagen en RGBA manteniendo el canal de transparencia intacto
        img_pil = Image.open(ruta_imagen).convert("RGBA")
        imagen_rgba = np.array(img_pil)
        
        # 2. Indexación automática con máscara de transparencia
        mapa_indexado, paleta_colores, mascara_valida = self.procesar_imagen_rgb(imagen_rgba)

        # 3. Construcción del grafo
        grafo = self.construir_grafo_adyacencia(mapa_indexado)
        nodos_ordenados = list(grafo.nodes())
        
        print(f"-> Archivo analizado correctamente: '{ruta_imagen}'")
        print(f"-> Se detectaron {len(nodos_ordenados)} colores puros e independientes (excluyendo fondo transparente).")

        # 4. Renderizado
        fig, (ax_img, ax_grafo) = plt.subplots(1, 2, figsize=(14, 6), num="PoC: Grafo desde Imagen Única")
        
        # Panel Izquierdo: Tu imagen original tal cual es
        ax_img.imshow(imagen_rgba)
        ax_img.set_title("Imagen Original", fontsize=11)
        ax_img.axis('off')

        # Panel Derecho: Grafo con los colores extraídos de los píxeles
        # --- RENDERIZADO DEL GRAFO CON DETALLES ADAPTATIVOS ---
        ax_grafo.set_title("Grafo de Adyacencia Cromática", fontsize=11)
        
        # CAMBIO 1: Ordenar nodos por afinidad (para que los que se tocan queden juntos en el círculo)
        nodos_ordenados_afinidad = list(grafo.nodes())
        if len(nodos_ordenados_afinidad) > 2:
            # Ordenamiento simple por grado (conectividad) para agrupar nodos dominantes
            nodos_ordenados_afinidad = sorted(grafo.nodes(), key=lambda n: grafo.degree(n), reverse=True)
            
        # Forzamos el layout circular respetando estrictamente nuestro orden de nodos
        pos = nx.circular_layout(grafo, scale=1.0, center=None)
        ax_grafo.set_aspect('equal')
        # Re-mapeamos las posiciones para asegurar el ordenamiento perimetral limpio
        pos = {nodo: pos[nodo] for nodo in nodos_ordenados_afinidad}

        # 1. CÁLCULO DE ÁREA PROPORCIONAL EXACTA
        # Total de píxeles reales de la imagen analizada
        # Cambiamos el total de píxeles para que sea solo la sumatoria del área visible de la fachada
        total_pixeles_imagen = np.sum(mascara_valida)
        
        # Forzamos que bincount ignore el índice -1 usando un filtrado simple
        pixeles_validos_flat = mapa_indexado[mapa_indexado >= 0]
        conteo_pixeles = np.bincount(pixeles_validos_flat)
        
        # Límites de tamaño (área del marcador en puntos cuadrados de Matplotlib)
        tamano_min_base = 150
        tamano_max_disponible = 5000

        # 2. DIBUJO DE NODOS CON CONTORNO CONDICIONAL Y PROPORCIÓN DE ÁREA REAL
        for nodo in grafo.nodes():
            rgb = paleta_colores[nodo]
            luminancia = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            #color_borde = 'black' if luminancia > 0.88 else 'none'
            #ancho_borde = 0.5 if luminancia > 0.88 else 0.0
            
            color_borde = 'white'
            ancho_borde = 5.0

            # Cantidad de píxeles de este componente de color
            pixeles_nodo = conteo_pixeles[nodo] if nodo < len(conteo_pixeles) else 0
            
            # Porcentaje exacto de ocupación sobre la fachada (valor entre 0.0 y 1.0)
            porcentaje_area = pixeles_nodo / total_pixeles_imagen
            
            # El parámetro node_size de Matplotlib define el ÁREA del círculo.
            # Al escalarlo linealmente con el porcentaje, la masa visual del nodo 
            # se corresponde de forma directa con su dominancia cromática.
            tamano_proporcional = tamano_min_base + (porcentaje_area * tamano_max_disponible)
            
            # Dibujamos el nodo con su peso superficial real
            nodos_coleccion = nx.draw_networkx_nodes(
                grafo, pos, nodelist=[nodo], ax=ax_grafo,
                node_color=[rgb], node_size=int(tamano_proporcional), 
                edgecolors=color_borde, linewidths=ancho_borde
            )
            if nodos_coleccion is not None:
                nodos_coleccion.set_zorder(3)

                
        # 2. DIBUJO DE EDGES EN DEGRADÉ CROMÁTICO CON AJUSTE DE EXTREMOS
        pesos_todos = [grafo[a][b]['weight'] for a, b in grafo.edges()]
        max_peso = max(pesos_todos) if pesos_todos else 1

        for u, v in grafo.edges():
            pos_u = np.array(pos[u])
            pos_v = np.array(pos[v])
            
            rgb_u = np.array(paleta_colores[u][:3])
            rgb_v = np.array(paleta_colores[v][:3])
            
            # Calculamos el color intermedio exacto entre ambos nodos
            rgb_intermedio = (rgb_u + rgb_v) / 2.0
            
            peso = grafo[u][v]['weight']
            grosor = 0.2 + (peso / max_peso) * 2
            
            

            if True:
                # Dibujamos la arista individual con su color interpolado y transparencia (alpha)
                nx.draw_networkx_edges(
                    grafo, pos, edgelist=[(u, v)], ax=ax_grafo,
                    width=grosor, edge_color=rgb_intermedio, alpha=.8
                )
            
            if False:
                num_segmentos = 3
                # Esto evita que las líneas se crucen justo en el centro del nodo y saturen la visualización
                t = np.linspace(0.0, 1.0, num_segmentos + 1)
                        
                puntos = np.array([pos_u + ti * (pos_v - pos_u) for ti in t])
                segmentos = np.array([puntos[i:i+2] for i in range(num_segmentos)])
                
                colores_segmentos = []
                for i in range(num_segmentos):
                    t_medio = (t[i] + t[i+1]) / 2.0
                    color_rgb = rgb_u + t_medio * (rgb_v - rgb_u)
                    colores_segmentos.append([color_rgb[0], color_rgb[1], color_rgb[2], 1.0])
                    
                # Creamos y dibujamos la colección de líneas en degradé para esta arista
                lc = LineCollection(segmentos.tolist(), colors=colores_segmentos, linewidths=grosor)
                lc.set_zorder(1)
                ax_grafo.add_collection(lc)

        ax_grafo.axis('off')
        plt.tight_layout()
        plt.show()

# =========================================================================
# EJECUCIÓN DIRECTA
# =========================================================================
if __name__ == "__main__":
    # Cambiá esto por el nombre de tu archivo PNG con los 9 o 10 colores directos
    dir_entrada = "./../entrada/isleta/pruebas"
    filename = "perro.png"
    ruta_final = os.path.join(dir_entrada, filename)
    
    # Creación automática de un archivo de prueba si no existe
    if not os.path.exists(ruta_final):
        print(f" Generando archivo de prueba '{ruta_final}' con 9 colores netos...")
        test_img = np.zeros((200, 450, 3), dtype=np.uint8)
        
        # Paleta de simulación de 9 colores planos en contacto
        colores_9 = [
            [240, 240, 240], [44, 62, 80], [231, 76, 60],
            [46, 204, 113], [52, 152, 219], [241, 196, 15],
            [155, 89, 182], [149, 165, 166], [211, 84, 0]
        ]
        for idx, rgb in enumerate(colores_9):
            x_ini = idx * 50
            test_img[:, x_ini:x_ini+50] = rgb
            
        Image.fromarray(test_img).save(ruta_final)

    analizador = AnalizadorAdyacenciaUnicaImagen()
    analizador.visualizar_poc(ruta_final)