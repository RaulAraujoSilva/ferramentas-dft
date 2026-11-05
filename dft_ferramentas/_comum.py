"""Utilidades comuns: paleta ECG para gráficos, leitura de CSV com vírgula decimal, formatação pt-BR
(números, eixos de gráfico e meses)."""
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


MESES_BR = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")


def mes_br(rotulo) -> str:
    """Converte um mês em rótulo pt-BR: ``"2025-01"``, ``"2025-01-15"`` ou data → ``"jan/2025"``.

    Rótulos que não são datas voltam inalterados (como texto)."""
    if hasattr(rotulo, "year") and hasattr(rotulo, "month"):
        return f"{MESES_BR[int(rotulo.month) - 1]}/{int(rotulo.year)}"
    t = str(rotulo).strip()
    partes = t.replace("/", "-").split("-")
    eh_data = len(partes) >= 2 and partes[0].isdigit() and len(partes[0]) == 4 and partes[1].isdigit()
    if eh_data and 1 <= int(partes[1]) <= 12:
        return f"{MESES_BR[int(partes[1]) - 1]}/{partes[0]}"
    return t


def meses_seguintes(rotulo, quantidade: int) -> list[str]:
    """Os ``quantidade`` meses posteriores a ``rotulo`` (``"2025-12"`` → ``["jan/2026", ...]``); [] se não for data."""
    t = str(rotulo).strip().replace("/", "-").split("-")
    if not (len(t) >= 2 and t[0].isdigit() and t[1].isdigit()):
        return []
    ano, mes = int(t[0]), int(t[1])
    saida = []
    for _ in range(quantidade):
        mes += 1
        if mes > 12:
            ano, mes = ano + 1, 1
        saida.append(f"{MESES_BR[mes - 1]}/{ano}")
    return saida


def _formatador_br(casas: int | None = None):
    """Formatador de eixo matplotlib em pt-BR: vírgula decimal, ponto de milhar e casas decimais automáticas."""
    from matplotlib.ticker import Formatter

    class FormatadorBR(Formatter):
        def __init__(self):
            self.casas = casas
            self._auto = 0

        def set_locs(self, locs):
            super().set_locs(locs)
            if self.casas is None:
                vis = [float(v) for v in locs if np.isfinite(v)]
                c = 0
                while c < 4 and any(abs(v * 10 ** c - round(v * 10 ** c)) > 1e-6 * max(1.0, abs(v * 10 ** c))
                                    for v in vis):
                    c += 1
                self._auto = c

        def __call__(self, x, pos=None):
            c = self.casas if self.casas is not None else self._auto
            if abs(x) < 0.5 * 10 ** -c:
                x = 0.0
            return br(x, c).replace("-", "−")

    return FormatadorBR()


def formatar_eixos_br(alvo, casas: int | None = None):
    """Aplica o formato pt-BR (vírgula decimal, ponto de milhar) aos rótulos numéricos dos eixos.

    ``alvo`` pode ser um ``Axes`` ou uma ``Figure`` (formata todos os eixos, inclusive os gêmeos). Só substitui o
    formatador numérico padrão (``ScalarFormatter``); rótulos de texto definidos pelo autor (``set_xticks(x, nomes)``)
    e formatadores próprios são preservados. Devolve ``alvo``."""
    from matplotlib.ticker import ScalarFormatter
    eixos = alvo.get_axes() if hasattr(alvo, "get_axes") else [alvo]
    for ax in eixos:
        for eixo in (ax.xaxis, ax.yaxis):
            if type(eixo.get_major_formatter()) is ScalarFormatter:
                eixo.set_major_formatter(_formatador_br(casas))
    return alvo


def _registrar_work_sans(fm) -> bool:
    if any(f.name == "Work Sans" for f in fm.fontManager.ttflist):
        return True
    try:
        for caminho in fm.findSystemFonts():
            if "worksans" in caminho.lower().replace("-", "").replace("_", ""):
                fm.fontManager.addfont(caminho)
    except Exception:
        return False
    return any(f.name == "Work Sans" for f in fm.fontManager.ttflist)


def estilo_graficos():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PALETA["linha"],
                         "axes.prop_cycle": plt.cycler(color=SERIES), "font.size": 11, "figure.dpi": 110,
                         "savefig.bbox": "tight", "axes.formatter.use_locale": False})
    if _registrar_work_sans(font_manager):  # identidade ECG quando a fonte estiver instalada; senão, padrão
        plt.rcParams["font.family"] = ["Work Sans", "DejaVu Sans"]
    return plt


def rng(semente: int | None = 20261105) -> np.random.Generator:
    """Gerador com semente fixa (resultados reprodutíveis em aula)."""
    return np.random.default_rng(semente)
