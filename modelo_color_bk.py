import numpy as np
import os
import json
from dataclasses import dataclass
from PIL import Image
from skimage import color
from scipy.ndimage import gaussian_filter
from typing import Any


@dataclass(frozen=True)
class ContenedorMuestrasColor:
    """Encapsula el universo completo de datos de la imagen analizada."""
    imagen_pil: Image.Image
    alto_real: int
    ancho_real: int
    lab_total: np.ndarray
    rgb_total: np.ndarray


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
    def __init__(self, ruta_imagen: str, paso_submuestreo: int = 50):
        self.ruta_imagen: str = ruta_imagen
        self.paso_submuestreo: int = paso_submuestreo
        
        # Atributos que encapsulan el estado de los datos reales
        self.alto_original: int = 0
        self.ancho_original: int = 0
        self.rgb_total: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.lab_total: np.ndarray = np.empty((0, 3), dtype=np.float64)
        
        # Atributos optimizados para visualización 3D y análisis rápido
        self.lab_pixeles_muestreo: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.rgb_pixeles_muestreo: np.ndarray = np.empty((0, 3), dtype=np.float64)
        
        # Ejecutar la carga y conversión única del universo de datos
        self._cargar_y_procesar_imagen()

    def _cargar_y_procesar_imagen(self) -> None:
        """
        Carga la imagen desde disco una única vez y calcula las matrices
        de color globales y submuestreadas.
        """
        # Carga única optimizada para evitar redundancia en memoria
        with Image.open(self.ruta_imagen).convert('RGB') as img:
            imagen_np = np.array(img)
            
        self.alto_original, self.ancho_original = imagen_np.shape[:2]
        
        # Conversión del universo completo a CIELAB con máxima precisión
        self.rgb_total = imagen_np.reshape(-1, 3) / 255.0
        self.lab_total = color.rgb2lab(self.rgb_total)
        
        # Submuestreo controlado para el motor de renderizado 3D y histogramas
        self.lab_pixeles_muestreo = self.lab_total[::self.paso_submuestreo]
        self.rgb_pixeles_muestreo = self.rgb_total[::self.paso_submuestreo]

    def obtener_dimensiones_reales(self) -> tuple[int, int]:
        """Retorna el alto y ancho original de la imagen procesada."""
        return self.alto_original, self.ancho_original

    def obtener_coordenadas_cielch(self, usar_muestreo: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula y retorna los componentes L (Luminosidad), C (Croma) y h (Tono en radianes)
        corrigiendo la discontinuidad del espacio cíclico.
        """
        datos_lab = self.lab_pixeles_muestreo if usar_muestreo else self.lab_total
        
        L = datos_lab[:, 0]
        C = np.sqrt(datos_lab[:, 1]**2 + datos_lab[:, 2]**2)
        
        # Tono en radianes mapeado limpiamente de 0 a 2pi
        tonos_grados = np.degrees(np.arctan2(datos_lab[:, 2], datos_lab[:, 1])) % 360
        h_rad = np.radians(tonos_grados)
        
        return L, C, h_rad
    
class AgrupadorColor:
    """
    Clase que encapsula los algoritmos de análisis de densidad, reducción 
    por Median Cut y agrupamiento geométrico en el espacio elipsoidal local.
    """
    def __init__(self, tol_L: float = 20.0, tol_C: float = 20.0, tol_T: float = 10.0):
        # Parámetros rígidos de la elipsoide de tolerancia
        self.tol_L: float = tol_L
        self.tol_C: float = tol_C
        self.tol_T: float = tol_T

    def detectar_anclas(
        self, 
        lab_pixeles: np.ndarray, 
        divisiones: int = 30, 
        umbral_porcentaje: float = 0.001, 
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Detecta colores ancla basados en la densidad del espacio CIELAB.
        """
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

    def obtener_paleta_mediana(self, lab_pixeles: np.ndarray, n_colores: int = 500) -> np.ndarray:
        """
        Reduce los colores de la matriz mediante el algoritmo Median Cut.
        """
        cajas = [lab_pixeles]
        
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
            
        paleta = [np.mean(caja, axis=0) for caja in cajas]
        return np.array(paleta)

    def agrupar_por_tolerancia_fija(self, centroides: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Agrupa centroides usando un sistema cartesiano local (Radial/Tangencial).
        Garantiza que ningún volumen de elipsoide se solape con otro mediante vectorización.
        """
        indices_restantes = list(range(len(centroides)))
        centros_finales: list[np.ndarray] = []
        etiquetas = np.full(len(centroides), -1)
        id_cluster = 0
        
        # Precalculamos los radios para evitar divisiones redundantes en el bucle
        r_L = self.tol_L / 2
        r_C = self.tol_C / 2
        r_T = self.tol_T / 2
        
        while len(indices_restantes) > 0:
            mejores_vecinos = []
            mejor_centro_idx = -1
            
            # Convertimos los centros aceptados a matriz para la validación vectorizada
            matriz_centros = np.array(centros_finales) if centros_finales else None
            
            if matriz_centros is not None:
                # Vectorización del cálculo de vectores directores para TODOS los centros aceptados
                L_g = matriz_centros[:, 0]
                a_g = matriz_centros[:, 1]
                b_g = matriz_centros[:, 2]
                C_g = np.sqrt(a_g**2 + b_g**2)
                
                # Máscaras para evitar división por cero en el origen cromático
                valido = C_g > 1e-5
                uR_ag = np.where(valido, a_g / np.where(valido, C_g, 1.0), 1.0)
                uR_bg = np.where(valido, b_g / np.where(valido, C_g, 1.0), 0.0)
                uT_ag = np.where(valido, -b_g / np.where(valido, C_g, 1.0), 0.0)
                uT_bg = np.where(valido, a_g / np.where(valido, C_g, 1.0), 1.0)
            
            for idx in indices_restantes:
                L0, a0, b0 = centroides[idx]
                
                # Reemplazo de la lógica repetida:
                uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centroides[idx])
                
                # --- FILTRO DE NO-SOLAPAMIENTO VECTORIZADO ---
                if matriz_centros is not None:
                    dL_g = (L0 - L_g) / r_L
                    da_g = a0 - a_g
                    db_g = b0 - b_g
                    dC_g = (da_g * uR_ag + db_g * uR_bg) / r_C
                    dT_g = (da_g * uT_ag + db_g * uT_bg) / r_T
                    
                    if np.any((dL_g**2 + dC_g**2 + dT_g**2) < 4.0):
                        continue
                
                # --- BUSQUEDA DE VECINOS DENTRO DEL RADIO DE TOLERANCIA ---
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
        
    def filtrar_pixeles_por_elipsoide(self, centro_lab: np.ndarray, lab_total: np.ndarray) -> np.ndarray:
        """
        Aplica la proyección ortogonal y geométrica del elipsoide rígido orientado.
        Devuelve los índices de los píxeles que pertenecen a la familia.
        """
        L0, a0, b0 = centro_lab
        
        # Uso de la función unificada
        uR_a, uR_b, uT_a, uT_b = AgrupadorColor.obtener_sistema_local(centro_lab)
        
        # Distancias normalizadas
        dL = (lab_total[:, 0] - L0) / (self.tol_L / 2)
        da = lab_total[:, 1] - a0
        db = lab_total[:, 2] - b0
        
        dC = (da * uR_a + db * uR_b) / (self.tol_C / 2)
        dT = (da * uT_a + db * uT_b) / (self.tol_T / 2)
        
        return np.where((dL**2 + dC**2 + dT**2) <= 1.0)[0]
    
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
    