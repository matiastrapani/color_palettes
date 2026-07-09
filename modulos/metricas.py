from dataclasses import dataclass

@dataclass
class MetricasProcesamiento:
    alto_px: int = 0
    ancho_px: int = 0
    total_absoluto: int = 0
    muestras_utiles: int = 0
    muestras_enmascaradas: int = 0
    pixeles_cubiertos: int = 0
    cobertura_total_pct: float = 0.0
    cobertura_util_pct: float = 0.0
    ruido_pct: float = 0.0
    familias: dict | None = None # Inicializar en __post_init__ si es necesario

    def __post_init__(self):
        if self.familias is None:
            self.familias = {}