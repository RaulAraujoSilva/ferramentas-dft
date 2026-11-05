"""Utilidades comuns: paleta ECG para gráficos, leitura de CSV com vírgula decimal, formatação pt-BR."""
from __future__ import annotations

import numpy as np
import pandas as pd

PALETA = {"marinho": "#0D2548", "azul": "#008DC6", "azul_escuro": "#00639A", "verde": "#70B83D",
          "verde_escuro": "#3F7F1F", "cinza": "#4D4D4D", "linha": "#C9D3DE", "claro": "#F2F2F2"}
SERIES = [PALETA["azul"], PALETA["marinho"], PALETA["verde_escuro"], PALETA["cinza"]]


def ler_csv(caminho, **kw) -> pd.DataFrame:
    """Lê CSV em pt-BR (separador ; e vírgula decimal) ou internacional, detectando automaticamente."""
    with open(caminho, encoding="utf-8-sig") as f:
        cab = f.readline()
    if ";" in cab:
        return pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig", **kw)
    return pd.read_csv(caminho, encoding="utf-8-sig", **kw)


def br(x: float, casas: int = 2) -> str:
    """Número no formato brasileiro: 1.234,56."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def estilo_graficos():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PALETA["linha"],
                         "axes.prop_cycle": plt.cycler(color=SERIES), "font.size": 11, "figure.dpi": 110,
                         "savefig.bbox": "tight"})
    return plt


def rng(semente: int | None = 20261105) -> np.random.Generator:
    """Gerador com semente fixa (resultados reprodutíveis em aula)."""
    return np.random.default_rng(semente)
