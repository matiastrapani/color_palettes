import os
import matplotlib.pyplot as plt
from PIL import Image
from modulos.modelo_color import ProcesadorEspacioColor
from modulos.modelo_color import AgrupadorColor, NomencladorColor
from modulos.interfase_color import VentanaDetalleColor, VisualizadorEspacioColor

# Parámetros de Configuración Inmutables localizados
CONFIGURACION = {
    "tolerancias": {
        "L": 15.0,
        "C": 15.0,
        "T": 8.0
    },
    "n_paleta_mediana": 1000,
    "ancho_analisis": 200,
    "paso_submuestreo": 50,
    "radio_desplazamientos": 20
}



def ejecutar_analisis_color(filename: str, dir_entrada: str | None = None, dir_mascara: str | None = None):
    ruta_imagen = os.path.join(dir_entrada or "", filename)
    filename_masc = f"MASC_{os.path.splitext(filename)[0]}.png"
    ruta_masc = os.path.join(dir_mascara or "", filename_masc)


    # 1. El procesador solo carga la imagen original en resolución nativa HD
    procesador = ProcesadorEspacioColor()
    estado_original = procesador.cargar_desde_archivo(
        ruta_imagen, 
        ruta_mascara=ruta_masc if os.path.exists(ruta_masc) else None
    )
    
    if estado_original.mascara is not None:
        print(f"-> Máscara cargada correctamente: {estado_original.mascara.shape}")

    # 2. Vos decidís de forma independiente si querés una reducción de resolución
    # para trabajar más rápido en los filtros, sin destruir el estado_original.
    estado_reducido = procesador.reducir_resolucion(
        estado_original, 
        ancho_objetivo=CONFIGURACION["ancho_analisis"]
    )

    #estado_reducido = procesador.submuestrear_por_pasos(
    #    estado_original, 
    #    paso=CONFIGURACION["paso_submuestreo"]  # O el valor que quieras usar de tu configuración
    #)

    # El agrupador nace libre de tolerancias fijas, es una caja de herramientas puras
    agrupador = AgrupadorColor()

    # 3. FILTROS SECUENCIALES ESTILO PHOTOSHOP:
    # Aplicás filtros sobre el objeto y decidís qué variable se transforma o se genera.
    
    # PASO A: Si tenés ganas, aplicás el Mean-Shift sobre el estado de análisis.
    # Devuelve un nuevo estado con las manchas homogéneas y bordes limpios.
    estado_filtrado = agrupador.filtro_desplazamiento_media(estado_reducido, radio_color=CONFIGURACION["radio_desplazamientos"])
    
    
    # PASO B: Calculás las anclas. Elegís qué datos usar (ej: el estado filtrado)
    anclas = agrupador.detectar_anclas(estado_filtrado, divisiones=50, umbral_porcentaje=0.0005)
    
    # PASO C: Corrés el Median Cut. Te devuelve la paleta reducida.
    paleta_mediana = agrupador.filtro_paleta_mediana(estado_filtrado, n_colores=CONFIGURACION["n_paleta_mediana"])

    procesador.visualizar_estados_dinamico(
        estados=[estado_original, estado_filtrado, paleta_mediana],
        titulos=["1. Original HD", "2. Mean-Shift", "3. Median Cut"]
    )

    
    # PASO D: Construcción de familias por elipsoides fijas. 
    # Alimentamos este método con los centros reducidos de la paleta mediana.
    etiquetas_familias, familias = agrupador.agrupar_por_tolerancia_fija(
        estado=paleta_mediana, 
        tolerancias=CONFIGURACION["tolerancias"],
        estado_original=estado_original
        )
    
    
    ventana_detalle = VentanaDetalleColor()
    ventana_detalle.inicializar()
    nomenclador = NomencladorColor()
    
    visualizador = VisualizadorEspacioColor(
        tolerancias=CONFIGURACION["tolerancias"]
    )

    # El gráfico 3D renderiza la nube de puntos del estado que vos elijas inspeccionar (ej: el filtrado)
    visualizador.inicializar_grafico_3d(
        estado_analisis=estado_reducido,  # <- Enviamos el objeto de estado completo
        estado_original=estado_original,
        anclas=anclas,
        paleta_mediana=paleta_mediana,
        familias=familias,
        etiquetas_familias=etiquetas_familias,
        ventana_detalle=ventana_detalle,
        agrupador=agrupador,
        nomenclador=nomenclador
    )
    
    visualizador.configurar_titulo(ruta_imagen=ruta_imagen)
    
    # 2. Configuramos el título de la paleta lateral con sus muestras
    visualizador.configurar_titulo_paleta(
        cantidad_familias=len(familias),
        configuracion=CONFIGURACION,
        estado_original=estado_original
    )
    
    plt.show()

if __name__ == "__main__":
    dir_entrada = "entrada/isleta"
    dir_mascara = "salida/isleta/masc"
    #filename = "PIÑEYRO_ISLETA_260616_F1937_F1941.jpg"
    #filename = "PIÑEYRO_ISLETA_260616_F1971_2022.jpg"
    #filename = "PIÑEYRO_ISLETA_260616_F1995_2.jpg"
    #filename = "PIÑEYRO_ISLETA_260616_F1995_1.jpg"
    #filename = "PIÑEYRO_ISLETA_260616_F1921_2.jpg"
    filename = "pajaros.jpg"

    ruta_final = os.path.join(dir_entrada, filename)
    
    ejecutar_analisis_color(filename, dir_entrada, dir_mascara)