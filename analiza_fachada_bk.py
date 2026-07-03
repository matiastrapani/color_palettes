import os
import matplotlib.pyplot as plt
from PIL import Image
from modelo_color_bk import ProcesadorEspacioColor
from modelo_color_bk import AgrupadorColor, NomencladorColor, ContenedorMuestrasColor
from interfase_color_bk import VentanaDetalleColor, VisualizadorEspacioColor

# Parámetros de Configuración Inmutables localizados
CONFIGURACION = {
    "tol_L": 15.0,
    "tol_C": 15.0,
    "tol_T": 15.0,
    "n_paleta_mediana": 1000,
    "paso_submuestreo": 50
}

def ejecutar_analisis_color(ruta_imagen: str):
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró la imagen en {ruta_imagen}")
        return

    print("1. Cargando imagen y convirtiendo espacios de color...")
    procesador = ProcesadorEspacioColor(ruta_imagen, paso_submuestreo=CONFIGURACION["paso_submuestreo"])
    alto, ancho = procesador.obtener_dimensiones_reales()
    
    print("2. Calculando picos de densidad (Anclas) y paleta base...")
    agrupador = AgrupadorColor(tol_L=CONFIGURACION["tol_L"], tol_C=CONFIGURACION["tol_C"], tol_T=CONFIGURACION["tol_T"])
    anclas = agrupador.detectar_anclas(procesador.lab_pixeles_muestreo, divisiones=50, umbral_porcentaje=0.0005, sigma=1.0)
    paleta_mediana = agrupador.obtener_paleta_mediana(procesador.lab_pixeles_muestreo, n_colores=CONFIGURACION["n_paleta_mediana"])
    
    print("3. Agrupando elipsoides rígidas orientadas locales...")
    etiquetas_familias, familias = agrupador.agrupar_por_tolerancia_fija(paleta_mediana)
    
    print("4. Inicializando interfaces y visualización interactiva...")
    # Cargamos la referencia de imagen PIL limpia para pasarla al popup
    imagen_pil = Image.open(ruta_imagen).convert('RGB')
    
    contexto_imagen = ContenedorMuestrasColor(
        imagen_pil=imagen_pil,
        alto_real=alto,
        ancho_real=ancho,
        lab_total=procesador.lab_total,
        rgb_total=procesador.rgb_total
    )

    ventana_detalle = VentanaDetalleColor()
    ventana_detalle.inicializar()

    nomenclador = NomencladorColor()
    
    visualizador = VisualizadorEspacioColor(tol_L=CONFIGURACION["tol_L"], tol_C=CONFIGURACION["tol_C"], tol_T=CONFIGURACION["tol_T"])
    visualizador.inicializar_grafico_3d(
        lab_muestreo=procesador.lab_pixeles_muestreo,
        rgb_muestreo=procesador.rgb_pixeles_muestreo,
        anclas=anclas,
        paleta_mediana=paleta_mediana,
        familias=familias,
        etiquetas_familias=etiquetas_familias,
        ventana_detalle=ventana_detalle,
        contexto_imagen=contexto_imagen,
        agrupador=agrupador,
        nomenclador=nomenclador
    )
    
    visualizador.configurar_titulo(ruta_imagen=ruta_imagen)
    
    # 2. Configuramos el título de la paleta lateral con sus muestras
    visualizador.configurar_titulo_paleta(cantidad_familias=len(familias))
    
    plt.show()

if __name__ == "__main__":
    dir_salida = "salida/isleta"
    filename = "RES_PIÑEYRO_ISLETA_260616_F1937_F1941_.jpg"
    ruta_final = os.path.join(dir_salida, filename)
    
    ejecutar_analisis_color(ruta_final)