import numpy as np
import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from PIL import Image
from skimage import color
from scipy.ndimage import gaussian_filter
from typing import Any


@dataclass(frozen=True)
class EstadoImagen:
    rgb: np.ndarray  # Ahora es una matriz 3D real: (Alto, Ancho, 3) tipo float32
    lab: np.ndarray  # Ahora es una matriz 3D real: (Alto, Ancho, 3) tipo float32
    ancho: int
    alto: int
    mascara: np.ndarray | None = None  # Ahora es una matriz 2D: (Alto, Ancho) de booleanos o uint8

    @property
    def rgb_plano(self) -> np.ndarray:
        """Devuelve una vista plana (N, 3) sin duplicar datos en memoria."""
        return self.rgb.reshape(-1, 3)

    @property
    def lab_plano(self) -> np.ndarray:
        """Devuelve una vista plana (N, 3) sin duplicar datos en memoria."""
        return self.lab.reshape(-1, 3)

    @property
    def datos(self) -> np.ndarray:
        """Solo píxeles activos para tus algoritmos de agrupamiento y estadísticas."""
        if self.mascara is None:
            return self.lab_plano
        # Filtramos usando la máscara aplanada
        return self.lab_plano[self.mascara.flatten().astype(bool)]

class NomencladorColor:
    """
    Asigna nombres descriptivos estandarizados a coordenadas CIELAB 
    cargando una base de datos compilada desde un archivo JSON local.
    """
    def __init__(self) -> None:
        # Determinamos la ruta del JSON de forma relativa al directorio del módulo
        self.ruta_json: str = os.path.join(os.path.dirname(__file__), "tabla_colores.json")
        self.tabla_colores: dict[str, Any] = self._cargar_tabla()

    def _cargar_tabla(self) -> dict[str, Any]:
        """Carga el diccionario de colores con sus coordenadas LAB precargadas."""
        if not os.path.exists(self.ruta_json):
            # Si el JSON no existe, devolvemos un diccionario vacío para evitar crashes
            return {}
        
        with open(self.ruta_json, "r", encoding="utf-8") as f:
            return json.load(f)

    def obtener_nombre(self, lab: np.ndarray) -> str:
        """
        Calcula el vecino más cercano mediante la métrica ΔE (distancia Euclídea en LAB).
        
        Parámetros:
            lab: np.ndarray con formato [L*, a*, b*]
        Devuelve:
            str con el nombre descriptivo más cercano del JSON.
        """
        if not self.tabla_colores:
            return "Nomenclador no inicializado o JSON vacío"

        nombre_mas_cercano: str = "Desconocido"
        distancia_minima: float = float('inf')

        for nombre, datos in self.tabla_colores.items():
            # Convertimos la lista del JSON a un vector de numpy para el cálculo geométrico
            ref_array = np.array(datos["lab"], dtype=np.float64)
            
            # Distancia Euclídea en el espacio perceptual CIELAB (ΔE)
            distancia = float(np.linalg.norm(lab - ref_array))
            
            if distancia < distancia_minima:
                distancia_minima = distancia
                nombre_mas_cercano = nombre

        return nombre_mas_cercano

class ProcesadorEspacioColor:
    """
    Clase responsable del ciclo de vida de los datos de imagen y sus
    transformaciones estrictas entre los espacios sRGB, CIELAB y CIELCh.
    """
    def __init__(self) -> None:
        pass

    def obtener_ruta_mascara(self, ruta_imagen: str, dir_mascara: str) -> str:
        """
        Construye la ruta de la máscara a partir de la ruta original.
        Regla: 'MASC_' + nombre_sin_extension + '.png' en el dir_mascara.
        """
        nombre_base = os.path.basename(ruta_imagen)
        nombre_sin_ext = os.path.splitext(nombre_base)[0]
        nombre_mascara = f"MASC_{nombre_sin_ext}.png"
    
        return os.path.join(dir_mascara, nombre_mascara)

    def cargar_desde_archivo(self, ruta_imagen: str, ruta_mascara: str | None = None) -> EstadoImagen:
        """
        Carga una imagen y su máscara opcional desde el disco duro, 
        manteniendo la estructura tridimensional y geométrica real de los datos.
        """
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"No se encontró la imagen en la ruta: {ruta_imagen}")
            
        # 1. Cargar imagen principal con PIL y normalizar a float32 [0.0, 1.0]
        with Image.open(ruta_imagen) as imagen_abierta:
            imagen_pil = imagen_abierta.convert("RGB")
            img_rgb_2d = np.array(imagen_pil, dtype=np.float32) / 255.0
            
        alto, ancho, _ = img_rgb_2d.shape
        
        # 2. Convertir la matriz 3D completa al espacio CIELAB nativo
        img_lab_2d = color.rgb2lab(img_rgb_2d)
        
        # 3. Cargar y alinear la máscara geométrica si existe
        mascara_2d = None
        if ruta_mascara and os.path.exists(ruta_mascara):
            mascara_pil = Image.open(ruta_mascara).convert("L")
            # Forzamos consistencia dimensional exacta con la imagen original
            if mascara_pil.size != (ancho, alto):
                mascara_pil = mascara_pil.resize((ancho, alto), Image.Resampling.NEAREST)
            # Guardamos como matriz binaria 2D (True = fachada, False = entorno)
            mascara_2d = np.array(mascara_pil) > 127
            
            # Opcional: Podés aplicar NaN o blanco al entorno de entrada aquí si lo preferís,
            # pero por ahora mantenemos la consistencia pura del array limpio
            
        # 4. Retornar el nuevo EstadoImagen estructurado en 3D real
        return EstadoImagen(
            rgb=img_rgb_2d,
            lab=img_lab_2d,
            ancho=ancho,
            alto=alto,
            mascara=mascara_2d
        )

    def submuestrear_por_pasos(self, estado_origen: EstadoImagen, paso: int) -> EstadoImagen:
        # Submuestreo de datos sobre los ejes espaciales reales (Alto, Ancho)
        rgb_3d = estado_origen.rgb[::paso, ::paso, :]
        lab_3d = estado_origen.lab[::paso, ::paso, :]
        
        # Propagación de máscara manteniendo el mismo paso bidimensional
        nueva_mascara = None
        if estado_origen.mascara is not None:
            nueva_mascara = estado_origen.mascara[::paso, ::paso]
            
        # Las dimensiones reales se obtienen directo de la nueva matriz resultante
        nuevo_alto, nuevo_ancho, _ = rgb_3d.shape
        
        return EstadoImagen(
            rgb=rgb_3d, 
            lab=lab_3d, 
            ancho=nuevo_ancho, 
            alto=nuevo_alto, 
            mascara=nueva_mascara
        )

    def reducir_resolucion(self, estado: EstadoImagen, ancho_objetivo: int) -> EstadoImagen:
        # 1. Calculamos las nuevas dimensiones proporcionales
        nuevo_alto = int(ancho_objetivo * (estado.alto / estado.ancho))
        
        # 2. Convertimos el array RGB 3D nativo a imagen PIL para redimensionar
        pixels_2d = (estado.rgb * 255).astype(np.uint8)
        img_pil = Image.fromarray(pixels_2d).resize((ancho_objetivo, nuevo_alto), Image.Resampling.BOX)
        
        # 3. Propagar máscara (binarizada) si existe operando en 2D nativo
        nueva_mascara = None
        if estado.mascara is not None:
            masc_pil = Image.fromarray(estado.mascara.astype(np.uint8) * 255)
            nueva_mascara = np.array(masc_pil.resize((ancho_objetivo, nuevo_alto), Image.Resampling.NEAREST)) > 128
            
        # 4. Reconvertir la imagen PIL achicada a nuestra estructura 3D
        rgb_3d = np.array(img_pil, dtype=np.float32) / 255.0
        lab_3d = color.rgb2lab(rgb_3d)
        
        return EstadoImagen(
            rgb=rgb_3d, 
            lab=lab_3d, 
            ancho=ancho_objetivo, 
            alto=nuevo_alto, 
            mascara=nueva_mascara
        )

    def obtener_coordenadas_cielch(self, estado: EstadoImagen) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula y retorna los componentes L (Luminosidad), C (Croma) y h (Tono en radianes)
        corrigiendo la discontinuidad del espacio cíclico.
        """
        L = estado.lab[:, 0]
        C = np.sqrt(estado.lab[:, 1]**2 + estado.lab[:, 2]**2)
        
        # Tono en radianes mapeado limpiamente de 0 a 2pi
        tonos_grados = np.degrees(np.arctan2(estado.lab[:, 2], estado.lab[:, 1])) % 360
        h_rad = np.radians(tonos_grados)
        
        return L, C, h_rad
    
    def visualizar_estados_dinamico(self, estados: list[EstadoImagen], titulos: list[str]):
        """
        Dibuja en una sola ventana de Matplotlib cualquier cantidad de estados 
        recibidos, adaptando la cantidad de columnas dinámicamente.
        """
        if len(estados) != len(titulos):
            raise ValueError("La cantidad de estados debe coincidir con la cantidad de títulos.")

        cantidad = len(estados)

        fig, axes = plt.subplots(1, cantidad, figsize=(6 * cantidad, 6), sharex=False, sharey=False)

        if cantidad == 1:
            axes = [axes]

        def preparar_rgba(estado: EstadoImagen):
            rgb_limpio = np.where(np.isnan(estado.rgb), 1.0, estado.rgb)
            if estado.mascara is not None:
                alfa = estado.mascara.astype(float)
                return np.dstack((rgb_limpio, alfa))
            return rgb_limpio

        for i, (estado, titulo) in enumerate(zip(estados, titulos)):
            ax = axes[i]
            ax.set_facecolor('white')
            ax.imshow(preparar_rgba(estado))
            ax.set_title(f"{titulo}\n({estado.ancho}x{estado.alto})", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        plt.show(block=False)
    
class AgrupadorColor:
    """
    Clase que encapsula los algoritmos de análisis de densidad, reducción 
    por Median Cut y agrupamiento geométrico en el espacio elipsoidal local.
    """
    def __init__(self) -> None:
        pass

    def detectar_anclas(
        self, 
        estado: EstadoImagen, 
        divisiones: int = 30, 
        umbral_porcentaje: float = 0.001, 
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Detecta colores ancla basados en la densidad del espacio CIELAB.
        Extrae automáticamente las muestras útiles aplicando la máscara si existe.
        """
        # Extraemos los datos utilizando la propiedad del EstadoImagen (aplica máscara si existe)[cite: 2]
        lab_pixeles = estado.datos
        
        # Limpieza estricta de NaNs remanentes (por ejemplo, del entorno de Mean-Shift)
        lab_pixeles = lab_pixeles[~np.isnan(lab_pixeles).any(axis=1)]
        
        if len(lab_pixeles) == 0:
            return np.empty((0, 3))

        rango = ((0, 100), (-100, 100), (-100, 100))
        hist, bordes = np.histogramdd(lab_pixeles, bins=divisiones, range=rango)
        
        if sigma > 0:
            hist = gaussian_filter(hist, sigma=sigma)
        
        total_pixeles = len(lab_pixeles)
        umbral_absoluto = total_pixeles * umbral_porcentaje
        
        anclas = []
        indices_picos = np.argwhere(hist > umbral_absoluto)
        
        for idx in indices_picos:
            l_idx, a_idx, b_idx = idx
            L = (bordes[0][l_idx] + bordes[0][l_idx+1]) / 2
            a = (bordes[1][a_idx] + bordes[1][a_idx+1]) / 2
            b = (bordes[2][b_idx] + bordes[2][b_idx+1]) / 2
            anclas.append([L, a, b])
            
        print(f"Anclas detectadas en el módulo: {len(anclas)}")
        return np.array(anclas)

    def filtro_paleta_mediana(self, estado: EstadoImagen, n_colores: int = 12) -> EstadoImagen:
        """
        Reduce los colores de la imagen mediante el algoritmo Median Cut,
        devolviendo un nuevo EstadoImagen con la estructura 3D nativa mapeada a la paleta.
        """
        from scipy.spatial import KDTree
        
        # 1. Extraemos solo los píxeles válidos bajo la máscara en Lab para calcular la paleta
        lab_activos = estado.datos
        
        if len(lab_activos) == 0:
            return estado
            
        # 2. Tu algoritmo exacto de Median Cut para encontrar los centros
        cajas = [lab_activos]
        
        while len(cajas) < n_colores:
            rangos = [np.ptp(caja, axis=0) for caja in cajas]
            max_rangos = [np.max(r) for r in rangos]
            idx_caja_a_dividir = np.argmax(max_rangos)
            
            caja_a_dividir = cajas.pop(idx_caja_a_dividir)
            
            if len(caja_a_dividir) <= 1:
                cajas.append(caja_a_dividir)
                break
                
            eje = np.argmax(rangos[idx_caja_a_dividir])
            caja_ordenada = caja_a_dividir[caja_a_dividir[:, eje].argsort()]
            medio = len(caja_ordenada) // 2
            
            cajas.append(caja_ordenada[:medio])
            cajas.append(caja_ordenada[medio:])
            
        centros_lab = np.array([np.mean(caja, axis=0) for caja in cajas])
        
        # 3. MAPEO GEOMÉTRICO: Reemplazar cada píxel original por el color más cercano de la paleta
        lab_original_plano = estado.lab_plano
        arbol = KDTree(centros_lab)
        _, indices = arbol.query(np.nan_to_num(lab_original_plano, nan=0.0))
        
        lab_mapeado_3d = centros_lab[indices].reshape((estado.alto, estado.ancho, 3))
        rgb_mapeado_3d = color.lab2rgb(lab_mapeado_3d).astype(np.float32)
        
        # 4. RESTAURAR MÁSCARA EXPLICITA: Forzar limpieza absoluta en áreas descartadas
        if estado.mascara is not None:
            mascara_inversa = ~estado.mascara.astype(bool)
            rgb_mapeado_3d[mascara_inversa] = np.nan
            lab_mapeado_3d[mascara_inversa] = np.nan
            
        return EstadoImagen(
            rgb=rgb_mapeado_3d,
            lab=lab_mapeado_3d,
            ancho=estado.ancho,
            alto=estado.alto,
            mascara=estado.mascara
        )

    def agrupar_por_tolerancia_fija(
        self, 
        estado: EstadoImagen, 
        tolerancias: dict[str, float],
        area_minima_porcentaje: float = 0.1,
        estado_original: EstadoImagen | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encuentra picos de densidad usando un histograma 3D, consolida las familias
        y calcula el porcentaje exacto de cobertura sobre el área útil de la fachada.
        """
        # 1. Filtrado inicial de NaNs (Área útil enmascarada)
        datos_validos = estado.datos
        datos_validos = datos_validos[~np.isnan(datos_validos).any(axis=1)]
        total_pixeles_utiles = len(datos_validos)
        
        if total_pixeles_utiles == 0:
            return np.full(0, -1), np.empty((0, 3))

        # 2. DETECCIÓN DE ANCLAS POR HISTOGRAMA 3D
        bin_L = max(1.0, tolerancias["L"] / 3.0)
        bin_C = max(1.0, tolerancias["C"] / 3.0)
        bin_T = max(1.0, tolerancias["T"] / 3.0)
        
        min_bounds = np.array([0.0, -128.0, -128.0])
        max_bounds = np.array([100.0, 127.0, 127.0])
        
        bins = [
            int((max_bounds[0] - min_bounds[0]) / bin_L),
            int((max_bounds[1] - min_bounds[1]) / bin_C),
            int((max_bounds[2] - min_bounds[2]) / bin_T)
        ]
        
        hist, edges = np.histogramdd(datos_validos, bins=bins, range=[(min_bounds[i], max_bounds[i]) for i in range(3)])
        
        umbral_pixeles_minimo = total_pixeles_utiles * (area_minima_porcentaje / 100.0)
        indices_picos = np.argwhere(hist >= umbral_pixeles_minimo)
        
        if len(indices_picos) == 0:
            return np.full(total_pixeles_utiles, -1), np.empty((0, 3))
        
        densidades = hist[tuple(indices_picos.T)]
        orden_densidad = np.argsort(densidades)[::-1]
        indices_picos = indices_picos[orden_densidad]
        
        anclas_candidatas = np.zeros((len(indices_picos), 3))
        for i in range(3):
            anclas_candidatas[:, i] = edges[i][indices_picos[:, i]] + (edges[i][1] - edges[i][0]) / 2.0

        # 3. CONSOLIDACIÓN DE FAMILIAS
        centros_finales: list[np.ndarray] = []
        r_L = tolerancias["L"] / 2.0
        r_C = tolerancias["C"] / 2.0
        r_T = tolerancias["T"] / 2.0
        
        for centro_propuesto in anclas_candidatas:
            if len(centros_finales) == 0:
                centros_finales.append(centro_propuesto)
                continue
                
            matriz_bloqueo = np.array(centros_finales)
            L_g, a_g, b_g = matriz_bloqueo[:, 0], matriz_bloqueo[:, 1], matriz_bloqueo[:, 2]
            C_g = np.sqrt(a_g**2 + b_g**2)
            
            valido = C_g > 1e-5
            uR_ag = np.where(valido, a_g / np.where(valido, C_g, 1.0), 1.0)
            uR_bg = np.where(valido, b_g / np.where(valido, C_g, 1.0), 0.0)
            uT_ag = np.where(valido, -b_g / np.where(valido, C_g, 1.0), 0.0)
            uT_bg = np.where(valido, a_g / np.where(valido, C_g, 1.0), 1.0)
            
            dL_g = (centro_propuesto[0] - L_g) / r_L
            da_g = centro_propuesto[1] - a_g
            db_g = centro_propuesto[2] - b_g
            dC_g = (da_g * uR_ag + db_g * uR_bg) / r_C
            dT_g = (da_g * uT_ag + db_g * uT_bg) / r_T
            
            if np.any((dL_g**2 + dC_g**2 + dT_g**2) < 4.0):
                continue
                
            centros_finales.append(centro_propuesto)
            
        centros_finales_np = np.array(centros_finales)
        
        # 4. CÁLCULO VECTORIAL DE COBERTURA REAL DE LAS FAMILIAS (UNIFICADO HD)
        if estado_original is not None:
            lab_calculo = estado_original.lab_plano
            mascara_hd = estado_original.mascara
            mascara_plana = mascara_hd.flatten() if mascara_hd is not None else None
            
            if mascara_hd is not None:
                total_util_impresion = int(np.sum(mascara_hd))
            else:
                total_util_impresion = int(np.sum(~np.isnan(lab_calculo).any(axis=1)))
        else:
            lab_calculo = datos_validos
            mascara_plana = None
            total_util_impresion = int(total_pixeles_utiles)  # Garantizado para el flujo estándar

        # Ahora sí, la variable siempre tiene un valor asignado
        pixeles_cubiertos = np.zeros(total_util_impresion, dtype=bool)
        
        print("\n--- Desglose de Cobertura de Familias ---")
        for idx, centro in enumerate(centros_finales_np):
            uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centro)
            
            if estado_original is not None:
                indices_familia = self.filtrar_pixeles_por_elipsoide(centro, lab_calculo, tolerancias=tolerancias)
                
                # Validación estricta para Pylance: aseguramos que mascara_plana no es None ni ambiguo
                if mascara_plana is not None:
                    condicion_mascara = mascara_plana[indices_familia].astype(bool)
                    indices_familia = indices_familia[condicion_mascara]
                
                cantidad_familia = len(indices_familia)
                porcentaje_individual = (cantidad_familia / total_util_impresion) * 100.0
                
                if mascara_plana is not None:
                    indices_activos_globales = np.where(mascara_plana)[0]
                    _, _, indices_locales = np.intersect1d(indices_familia, indices_activos_globales, return_indices=True)
                    pixeles_cubiertos[indices_locales] = True
                else:
                    pixeles_cubiertos[indices_familia] = True
            else:
                da_c = lab_calculo[:, 1] - centro[1]
                db_c = lab_calculo[:, 2] - centro[2]
                dL = (lab_calculo[:, 0] - centro[0]) / r_L
                dC = (da_c * uR_a + db_c * uR_b) / r_C
                dT = (da_c * uT_a + db_c * uT_b) / r_T
                
                pertenece_a_familia = (dL**2 + dC**2 + dT**2) <= 1.0
                pixeles_cubiertos |= pertenece_a_familia
                porcentaje_individual = (np.sum(pertenece_a_familia) / total_util_impresion) * 100.0
            
            print(f"Familia {idx+1} {centro.round(1)}: Cobertura de Fachada = {porcentaje_individual:.2f}%")
            
        # Métricas Globales
        total_cobertura_porcentaje = (np.sum(pixeles_cubiertos) / total_util_impresion) * 100.0
        print("-----------------------------------------")
        print(f"Área Útil Analizada (Fachada sin NaNs): {total_util_impresion} muestras.")
        print(f"Cobertura Total de las Familias sobre Fachada: {total_cobertura_porcentaje:.2f}%")
        print(f"Residuos / Ruido sin clasificar: {100.0 - total_cobertura_porcentaje:.2f}%\n")
        
        etiquetas_compatibilidad = np.full(total_pixeles_utiles, -1)
        return etiquetas_compatibilidad, centros_finales_np

    def agrupar_por_tolerancia_fija_original(
        self, 
        estado: EstadoImagen, 
        tolerancias: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Agrupa los píxeles activos del EstadoImagen usando un sistema cartesiano local 
        (Radial/Tangencial). Garantiza que ningún volumen de elipsoide se solape.
        """
        # --- CAMBIO CLAVE ---
        # Extraemos los datos Lab y nos quedamos ÚNICAMENTE con los colores únicos.
        # Si el estado viene del Median Cut, esto reduce 60.000 píxeles a solo 15 centros.
        # Filtramos vectores que contengan NaN antes de buscar valores únicos
        datos_validos = estado.datos
        datos_validos = datos_validos[~np.isnan(datos_validos).any(axis=1)]
        
        centroides = np.unique(datos_validos, axis=0) 
        
        indices_restantes = list(range(len(centroides)))
        centros_finales: list[np.ndarray] = []
        etiquetas = np.full(len(centroides), -1)
        id_cluster = 0
        
        # Precalculamos los radios para evitar divisiones redundantes en el bucle
        r_L = tolerancias["L"] / 2
        r_C = tolerancias["C"] / 2
        r_T = tolerancias["T"] / 2
        
        # --- EL RESTO DE TU CÓDIGO QUEDA EXACTAMENTE IGUAL ---
        while len(indices_restantes) > 0:
            mejores_vecinos = []
            mejor_centro_idx = -1
            
            matriz_centros = np.array(centros_finales) if centros_finales else None
            
            if matriz_centros is not None:
                L_g = matriz_centros[:, 0]
                a_g = matriz_centros[:, 1]
                b_g = matriz_centros[:, 2]
                C_g = np.sqrt(a_g**2 + b_g**2)
                
                valido = C_g > 1e-5
                uR_ag = np.where(valido, a_g / np.where(valido, C_g, 1.0), 1.0)
                uR_bg = np.where(valido, b_g / np.where(valido, C_g, 1.0), 0.0)
                uT_ag = np.where(valido, -b_g / np.where(valido, C_g, 1.0), 0.0)
                uT_bg = np.where(valido, a_g / np.where(valido, C_g, 1.0), 1.0)
            
            for idx in indices_restantes:
                L0, a0, b0 = centroides[idx]
                
                uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centroides[idx])
                
                if matriz_centros is not None:
                    dL_g = (L0 - L_g) / r_L
                    da_g = a0 - a_g
                    db_g = b0 - b_g
                    dC_g = (da_g * uR_ag + db_g * uR_bg) / r_C
                    dT_g = (da_g * uT_ag + db_g * uT_bg) / r_T
                    
                    if np.any((dL_g**2 + dC_g**2 + dT_g**2) < 4.0):
                        continue
                
                vecinos_candidatos = []
                for cand_idx in indices_restantes:
                    L1, a1, b1 = centroides[cand_idx]
                    
                    dL = (L1 - L0) / r_L
                    da, db = a1 - a0, b1 - b0
                    dC = (da * uR_a + db * uR_b) / r_C
                    dT = (da * uT_a + db * uT_b) / r_T
                    
                    if (dL**2 + dC**2 + dT**2) <= 1.0:
                        vecinos_candidatos.append(cand_idx)
                        
                if len(vecinos_candidatos) > len(mejores_vecinos):
                    mejores_vecinos = vecinos_candidatos
                    mejor_centro_idx = idx
                    
            if mejor_centro_idx == -1 or len(mejores_vecinos) == 0:
                break
                
            etiquetas[mejores_vecinos] = id_cluster
            centros_finales.append(centroides[mejor_centro_idx])
            id_cluster += 1
            
            for v in mejores_vecinos:
                indices_restantes.remove(v)
                
        print(f"Familias rígidas calculadas en el módulo: {id_cluster}")
        return etiquetas, np.array(centros_finales)
        
    def filtrar_pixeles_por_elipsoide(
        self, 
        centro_lab: np.ndarray, 
        lab_total: np.ndarray, 
        tolerancias: dict[str, float]
    ) -> np.ndarray:
        """
        Aplica la proyección ortogonal y geométrica del elipsoide rígido orientado.
        Devuelve los índices de los píxeles que pertenecen a la familia.
        """
        L0, a0, b0 = centro_lab
        
        # Uso de la función unificada
        uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centro_lab)
        
        # Distancias normalizadas
        dL = (lab_total[:, 0] - L0) / (tolerancias["L"] / 2)
        da = lab_total[:, 1] - a0
        db = lab_total[:, 2] - b0
        
        dC = (da * uR_a + db * uR_b) / (tolerancias["C"] / 2)
        dT = (da * uT_a + db * uT_b) / (tolerancias["T"] / 2)
        
        return np.where((dL**2 + dC**2 + dT**2) <= 1.0)[0]
    
    def filtro_desplazamiento_media(self, estado: EstadoImagen, radio_color: int = 20) -> EstadoImagen:
        import cv2
        
        # 1. Recuperamos el lienzo nativo 3D.
        canvas = estado.rgb.copy()
        
        # Rellenamos el entorno con el color promedio real de la fachada para neutralizar bordes
        if estado.mascara is not None:
            mascara_bool = estado.mascara.astype(bool)
            pixeles_fachada = canvas[mascara_bool]
            if len(pixeles_fachada) > 0:
                color_promedio_neutral = np.mean(pixeles_fachada, axis=0)
                canvas[~mascara_bool] = color_promedio_neutral
            else:
                canvas[~mascara_bool] = 1.0
        else:
            canvas = np.where(np.isnan(canvas), 1.0, canvas)
            
        canvas_uint8 = (canvas * 255).astype(np.uint8)
        
        # 2. Cálculo dinámico del radio espacial en base a las dimensiones reales
        radio_espacial = max(5, int(min(estado.alto, estado.ancho) * 0.015))
        
        # 3. Procesamiento BGR nativo con OpenCV
        imagen_bgr = cv2.cvtColor(canvas_uint8, cv2.COLOR_RGB2BGR)
        imagen_filtrada_bgr = cv2.pyrMeanShiftFiltering(imagen_bgr, sp=radio_espacial, sr=radio_color, maxLevel=1)
        imagen_filtrada_rgb = cv2.cvtColor(imagen_filtrada_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        
        # 4. RESTAURAR MÁSCARA ESTRICTA: Forzamos NaN en el entorno de inmediato
        if estado.mascara is not None:
            imagen_filtrada_rgb[~estado.mascara.astype(bool)] = np.nan
        
        # 5. El espacio Lab nace completamente libre de contaminación
        lab_filtrado = color.rgb2lab(np.nan_to_num(imagen_filtrada_rgb, nan=0.0))
        if estado.mascara is not None:
            lab_filtrado[~estado.mascara.astype(bool)] = np.nan

        return EstadoImagen(
            rgb=imagen_filtrada_rgb,
            lab=lab_filtrado,
            ancho=estado.ancho,
            alto=estado.alto,
            mascara=estado.mascara
        )
        

    @staticmethod
    def obtener_sistema_local(centro_lab: np.ndarray) -> tuple[float, float, float, float]:
        """
        Calcula los vectores directores unitarios (Radial y Tangencial) 
        para un centro dado en el espacio CIELAB.
        Devuelve: (uR_a, uR_b, uT_a, uT_b)
        """
        _, a0, b0 = centro_lab
        C0 = np.sqrt(a0**2 + b0**2)
        
        if C0 > 1e-5:
            uR_a, uR_b = a0 / C0, b0 / C0
            uT_a, uT_b = -b0 / C0, a0 / C0
        else:
            uR_a, uR_b = 1.0, 0.0
            uT_a, uT_b = 0.0, 1.0
            
        return uR_a, uR_b, uT_a, uT_b
    