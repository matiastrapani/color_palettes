import numpy as np
from dataclasses import dataclass
from scipy.ndimage import label
from skimage.measure import perimeter

@dataclass(frozen=True)
class ReporteMorfologicoColor:
    """Encapsula los resultados métricos del análisis espacial de una familia."""
    categoria: str               # "Dominante", "Acento" o "Textura"
    cobertura_global: float      # % de la superficie total de la imagen (0.0 a 100.0)
    cantidad_manchas: int        # Número de islas conexas significativas detectadas
    compacidad_media: float      # Promedio de la relación Área/Perímetro de las manchas
    manchas_totales: int         # Cantidad total de manchas incluyendo el ruido mínimo

class AnalizadorMorfologicoEspacial:
    """
    Librería pura encargada del análisis geométrico y distribución espacial 
    de las familias de color en una máscara binaria.
    """
    def __init__(self, umbral_ruido_pixeles: int = 4) -> None:
        """
        Parámetros:
            umbral_ruido_pixeles: Tamaño mínimo en píxeles para que una mancha 
                                  sea considerada significativa y no ruido térmico.
        """
        self.umbral_ruido = umbral_ruido_pixeles

    def analizar_mascara(self, mascara_2d: np.ndarray) -> ReporteMorfologicoColor:
        """
        Procesa una máscara binaria 2D (True/False o 1/0) para identificar sus 
        componentes conexos, calcular descriptores de forma y clasificar el rol del color.
        """
        if mascara_2d.ndim != 2:
            raise ValueError("La máscara debe tener exactamente 2 dimensiones (alto, ancho).")
            
        alto, ancho = mascara_2d.shape
        pixeles_totales = alto * ancho
        
        # 1. Cobertura Global
        pixeles_activos = int(np.sum(mascara_2d))
        cobertura_global = (pixeles_activos / pixeles_totales) * 100.0
        
        if pixeles_activos == 0:
            return ReporteMorfologicoColor(
                categoria="Inexistente",
                cobertura_global=0.0,
                cantidad_manchas=0,
                compacidad_media=0.0,
                manchas_totales=0
            )

        # 2. Segmentación de Componentes Conexos (Conectividad estructural de 8 vecinos)
        estructura_conectividad = np.ones((3, 3), dtype=int)
        resultado_label: tuple[np.ndarray, int] = label(mascara_2d, structure=estructura_conectividad)  # type: ignore
        mascara_etiquetada, n_componentes = resultado_label
        n_componentes: int = int(resultado_label[1])
        
        # 3. Extracción de propiedades por cada isla/mancha
        compacidades = []
        manchas_significativas = 0
        
        for i in range(1, n_componentes + 1):
            isla_binaria = (mascara_etiquetada == i)
            area_isla = int(np.sum(isla_binaria))
            
            # FILTRADO CORREGIDO: Descartar elementos menores al umbral de ruido
            if area_isla < self.umbral_ruido:
                continue
                
            manchas_significativas += 1
            
            # Cálculo del perímetro continuo usando Scikit-Image
            perimetro_isla = float(perimeter(isla_binaria, neighborhood=8))
            
            # Evitamos divisiones por cero si la geometría es anómala
            if perimetro_isla > 0:
                # Factor de compacidad estándar (Área / Perímetro)
                compacidad = area_isla / perimetro_isla
                compacidades.append(compacidad)
            else:
                compacidades.append(0.0)

        # Promedio del factor morfológico de las figuras válidas
        compacidad_media = float(np.mean(compacidades)) if compacidades else 0.0

        # 4. Árbol de decisión morfológica para asignación de rol cromático
        if cobertura_global >= 20.0:
            categoria = "Dominante"
        else:
            # Si hay muchas figuras fragmentadas o la compacidad promedio es muy baja, es trama/textura
            if manchas_significativas > 40 or (compacidad_media < 0.8 and manchas_significativas > 5):
                categoria = "Textura"
            else:
                categoria = "Acento"

        return ReporteMorfologicoColor(
            categoria=categoria,
            cobertura_global=cobertura_global,
            cantidad_manchas=manchas_significativas,
            compacidad_media=compacidad_media,
            manchas_totales=n_componentes
        )


if __name__ == "__main__":
    import os
    from PIL import Image

    print("=== TEST INDEPENDIENTE DEL ANALIZADOR MORFOLÓGICO ===")
    
    # 1. Definí acá la ruta de la máscara PNG que quieras testear
    # Podés arrastrar cualquier PNG en blanco y negro para probarlo
    ruta_test_png = "test_mascara_color_2.png"
    
    if not os.path.exists(ruta_test_png):
        print(f"\n[Aviso] No se encontró el archivo '{ruta_test_png}'.")
        print("Generando una máscara sintética de control (1000x1000 px) para validar...")
        
        # Creamos una matriz de prueba artificial (Fondo negro)
        mascara_prueba = np.zeros((1000, 1000), dtype=bool)
        
        # Caso Acento Sintético: Un cuadrado denso y concentrado en el centro
        mascara_prueba[400:600, 400:600] = True
        
        # Agregar un poco de ruido de textura aislado (píxeles sueltos)
        mascara_prueba[10, 10] = True
        mascara_prueba[900, 150] = True
        
        print("-> Máscara sintética de control creada con éxito.")
    else:
        print(f"Cargando máscara real desde: {ruta_test_png}")
        # Cargamos la imagen, la convertimos a escala de grises y luego a Booleana
        img_pil = Image.open(ruta_test_png).convert("L")
        # Consideramos píxeles activos aquellos que sean mayores a un umbral intermedio (128)
        mascara_prueba = np.array(img_pil) > 128
        print(f"-> Imagen cargada correctamente. Dimensiones: {mascara_prueba.shape}")

    # 2. Instanciar el analizador fijando el umbral de ruido en 4 píxeles
    analizador = AnalizadorMorfologicoEspacial(umbral_ruido_pixeles=4)
    
    # 3. Ejecutar el análisis morfológico puro
    print("\nProcesando matriz espacial y analizando islas conexas...")
    reporte = analizador.analizar_mascara(mascara_prueba)
    
    # 4. Desplegar los resultados en consola
    print("\n=============================================")
    print("           RESULTADO DEL DIAGNÓSTICO         ")
    print("=============================================")
    print(f" CATEGORÍA ASIGNADA  : {reporte.categoria.upper()}")
    print(f" Cobertura Superficie: {reporte.cobertura_global:.2f}%")
    print(f" Manchas Válidas     : {reporte.cantidad_manchas} islas")
    print(f" Manchas Totales (c/Ruido): {reporte.manchas_totales}")
    print(f" Compacidad Promedio : {reporte.compacidad_media:.4f}")
    print("=============================================\n")