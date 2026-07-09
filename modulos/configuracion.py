from dataclasses import dataclass

@dataclass(frozen=True)
class ParametrosAnalisis:
    tolerancia_L: float
    tolerancia_C: float
    tolerancia_T: float
    n_paleta_mediana: int
    ancho_analisis: int
    paso_submuestreo: int
    radio_desplazamientos: int

    @property
    def tolerancias(self) -> dict[str, float]:
        return {"L": self.tolerancia_L, "C": self.tolerancia_C, "T": self.tolerancia_T}