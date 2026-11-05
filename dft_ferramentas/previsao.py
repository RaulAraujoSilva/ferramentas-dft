"""4. Previsão de demanda — Fase 4 do DFT (Estimativa de demanda).

O que este módulo faz (dossiê em ``docs/metodos/4_previsao.md``):

1. lê uma série mensal (CSV em pt-BR) e diagnostica o tamanho, a intermitência (ADI/CV²) e a força de
   tendência e de sazonalidade (decomposição STL);
2. escolhe os métodos candidatos por uma **árvore de decisão por tamanho de série**;
3. calcula os **métodos de referência** (média, ingênuo, ingênuo sazonal, drift), **SES**, **Holt**, **Holt
   amortecido**, **Holt-Winters** (aditivo e multiplicativo), **ARIMA automático** simplificado (Hyndman-Khandakar)
   e, para demanda intermitente, **Croston**, **SBA** e **TSB**;
4. compara os candidatos por **validação de origem móvel** (ou treino/teste), **sem embaralhar**, com MAE, RMSE,
   MAPE (com guarda de zero) e MASE; escolhe pelo menor RMSE e, em empate, o mais simples;
5. projeta o horizonte pedido com **intervalo de previsão por bootstrap dos resíduos**;
6. aplica os **vetores de demanda** (decisão D4 do curso): fatores multiplicativos com vigência, travas e trilha;
7. gera o relatório legível e, opcionalmente, o gráfico antes/depois.

Convenção numérica (idêntica ao motor TypeScript do Sistema DFT e ao dossiê L2, §5.4): SES, Holt e Holt amortecido
são ajustados por **busca em grade** (passo 0,01) minimizando o SSE de um passo, com ``ℓ₀ = y₁``, ``b₀ = y₂ − y₁`` e
SSE somado a partir de ``t = 2``. É essa convenção que reproduz o caso de teste CT-3 (Holt vence SES, RMSE 6,20 <
7,41; Σ projeção de 12 meses = 2.481,05; Σ ajustada pelos vetores = 2.637,19). Holt-Winters e ARIMA usam o
``statsmodels`` (máxima verossimilhança/SSE com estado inicial estimado).

Referência-mestra: HYNDMAN, R. J.; ATHANASOPOULOS, G. *Forecasting: principles and practice*. 3. ed. OTexts, 2021
(FPP3), §§3.6, 4.3, 5.2, 5.5, 5.8, 5.10, 6.7, 8.1–8.3, 9.1, 9.7 e 13.2. Demais fontes no dossiê do método.
"""
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ._comum import PALETA, br, estilo_graficos, formatar_eixos_br, ler_csv, mes_br, meses_seguintes, rng

# ----------------------------------------------------------------------------------------------------------------
# Constantes do método
# ----------------------------------------------------------------------------------------------------------------
PASSO_GRADE = 0.01
"""Passo da busca em grade de α, β (L2, §5.4): 101 avaliações no SES, 10.201 no Holt."""
FAIXA_PHI = tuple(round(0.80 + 0.02 * i, 2) for i in range(10))
"""Valores de φ testados no Holt amortecido: 0,80 a 0,98 (faixa recomendada por FPP3, §8.2)."""
N_SIM_BOOTSTRAP = 1000
SEMENTE_PREVISAO = 42
"""Semente padrão do bootstrap (a mesma do motor TypeScript; registrada no relatório)."""
LIMIAR_SAZONALIDADE = 0.64
"""Força sazonal F_S a partir da qual se considera a sazonalidade relevante (FPP3, §9.1, ``unitroot_nsdiffs``)."""
LIMIAR_ADI = 1.32
LIMIAR_CV2 = 0.49
"""Fronteiras da classificação de padrões de demanda (Syntetos, Boylan e Croston, 2005)."""
FAIXA_BLOQUEADA = (0.97, 1.03)
"""Vetor com 0,97 < fator < 1,03 é bloqueado: ajuste pequeno demais (D4; FPP3, §6.7; Fildes et al., 2009)."""
FAIXA_COMPOSICAO = (0.60, 1.60)
"""Fora desta faixa, o fator composto F(h) gera alerta: revise o modelo, não acrescente vetores (L2, §2.3)."""
MIN_JUSTIFICATIVA = 200
"""Comprimento mínimo da justificativa de um vetor (D4)."""
MIN_ORIGENS = 3
"""Número mínimo de origens de teste que um candidato precisa ter na validação de origem móvel."""


class ErroPrevisao(ValueError):
    """Erro de entrada ou de uso dos métodos de previsão."""


class ErroVetor(ValueError):
    """Vetor de demanda inválido — vetor inválido não é aplicável (D4)."""


def _serie(y: Sequence[float], minimo: int = 1, metodo: str = "O método") -> np.ndarray:
    """Converte para ``np.ndarray`` de floats e valida tamanho e finitude."""
    a = np.asarray(y, dtype=float).ravel()
    if a.size < minimo:
        raise ErroPrevisao(f"{metodo} exige ao menos {minimo} observações; recebidas {a.size}.")
    if not np.all(np.isfinite(a)):
        raise ErroPrevisao("A série histórica contém valor ausente ou não finito.")
    return a


def _horizonte(h: int) -> int:
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ErroPrevisao("O horizonte h precisa ser inteiro ≥ 1.")
    return int(h)


# ----------------------------------------------------------------------------------------------------------------
# 1. Métodos de referência (benchmarks) — FPP3, §5.2
# ----------------------------------------------------------------------------------------------------------------
def prever_media(y: Sequence[float], h: int) -> np.ndarray:
    """Método da média: ``ŷ_{T+h|T} = ȳ`` para todo h (FPP3, §5.2). Unidade: a da série."""
    a = _serie(y, 1, "A média")
    return np.full(_horizonte(h), a.mean())


def prever_ingenuo(y: Sequence[float], h: int) -> np.ndarray:
    """Método ingênuo (*naïve*): ``ŷ_{T+h|T} = y_T`` (FPP3, §5.2)."""
    a = _serie(y, 1, "O ingênuo")
    return np.full(_horizonte(h), a[-1])


def prever_sazonal_ingenuo(y: Sequence[float], h: int, m: int) -> np.ndarray:
    """Ingênuo sazonal: ``ŷ_{T+h|T} = y_{T+h−m(k+1)}``, com ``k = ⌊(h−1)/m⌋`` (FPP3, §5.2).

    Repete o último ciclo observado. Exige ao menos ``m`` observações.
    """
    a = _serie(y, int(m), "O ingênuo sazonal")
    h = _horizonte(h)
    T = a.size
    return np.array([a[T + i - m * ((i - 1) // m + 1) - 1] for i in range(1, h + 1)])


def prever_drift(y: Sequence[float], h: int) -> np.ndarray:
    """Método *drift*: ``ŷ_{T+h|T} = y_T + h·(y_T − y₁)/(T − 1)`` (FPP3, §5.2).

    Equivale a traçar a reta entre a primeira e a última observação — a "projeção linear simples" que o artigo-base
    critica na prática atual dos órgãos.
    """
    a = _serie(y, 2, "O drift")
    h = _horizonte(h)
    inclinacao = (a[-1] - a[0]) / (a.size - 1)
    return a[-1] + inclinacao * np.arange(1, h + 1)


def metodos_referencia(y: Sequence[float], h: int, m: int | None = None) -> dict[str, np.ndarray | None]:
    """Os quatro métodos de referência de FPP3, §5.2, de uma vez. ``sazonal_ingenuo`` é ``None`` sem ``m`` válido."""
    a = _serie(y, 2, "Os métodos de referência")
    saz = prever_sazonal_ingenuo(a, h, m) if m and 1 <= m <= a.size else None
    return {"media": prever_media(a, h), "ingenuo": prever_ingenuo(a, h), "sazonal_ingenuo": saz,
            "drift": prever_drift(a, h)}


# ----------------------------------------------------------------------------------------------------------------
# 2. Suavização exponencial por busca em grade — FPP3, §§8.1–8.2; L2, §5.4 (convenção do CT-3)
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class ModeloSuavizacao:
    """Modelo de suavização exponencial ajustado (SES, Holt ou Holt amortecido).

    Atributos: ``tipo`` (``"ses"``, ``"holt"``, ``"holt_amortecido"``), parâmetros ``alpha``, ``beta``, ``phi``,
    ``sse`` de um passo no ajuste, estado inicial (``l0``, ``b0``) e final (``lT``, ``bT``), ``n`` observações e
    ``alertas`` (ex.: parâmetro na fronteira da grade).
    """

    tipo: str
    alpha: float
    beta: float
    phi: float
    sse: float
    l0: float
    b0: float
    lT: float
    bT: float
    n: int
    alertas: list[str] = field(default_factory=list)

    def prever(self, h: int) -> np.ndarray:
        """Previsões para ``1..h``. SES: plana (``ℓ_T``); Holt: ``ℓ_T + h·b_T``; amortecido:
        ``ℓ_T + (φ + … + φ^h)·b_T`` (FPP3, §8.2)."""
        h = _horizonte(h)
        if self.tipo == "ses":
            return np.full(h, self.lT)
        k = np.arange(1, h + 1)
        if self.phi == 1.0:
            return self.lT + k * self.bT
        return self.lT + self.phi * (1 - self.phi ** k) / (1 - self.phi) * self.bT


def _grade(passo: float) -> np.ndarray:
    if not (0 < passo <= 1):
        raise ErroPrevisao("O passo da grade precisa estar em (0; 1].")
    passos = round(1 / passo)
    if abs(passos * passo - 1) > 1e-9:
        raise ErroPrevisao("O passo da grade precisa dividir o intervalo [0; 1].")
    return np.arange(passos + 1) / passos


def _alerta_fronteira(simbolo: str, valor: float, sugestao: str) -> str | None:
    if valor not in (0.0, 1.0):
        return None
    return (f"{simbolo}* = {br(valor)} — parâmetro na fronteira da grade — amostra possivelmente curta; {sugestao}")


def ajustar_ses(y: Sequence[float], passo: float = PASSO_GRADE) -> ModeloSuavizacao:
    """Suavização exponencial simples por busca em grade de α ∈ [0; 1] (FPP3, §8.1; L2, §5.4).

    ``ℓ_t = α·y_t + (1−α)·ℓ_{t−1}``, ``ℓ₀ = y₁``, SSE de um passo somado a partir de t = 2. Previsão plana.
    CT-3 (treino de 9 pontos): α* = 0,90; SSE = 837,4104; ℓ_T = 170,617133.
    """
    a = _serie(y, 2, "O SES")
    alphas = _grade(passo)
    nivel = np.full(alphas.size, a[0])
    sse = np.zeros(alphas.size)
    for t in range(a.size):
        if t >= 1:
            sse += (a[t] - nivel) ** 2
        nivel = alphas * a[t] + (1 - alphas) * nivel
    i = int(np.argmin(sse))
    alertas = [x for x in [_alerta_fronteira("α", float(alphas[i]), "colete mais histórico antes de adotar o modelo.")]
               if x]
    return ModeloSuavizacao("ses", float(alphas[i]), 0.0, 1.0, float(sse[i]), float(a[0]), 0.0, float(nivel[i]), 0.0,
                            a.size, alertas)


def _filtro_holt(a: np.ndarray, alphas: np.ndarray, betas: np.ndarray, phi: float):
    """Roda as equações do Holt (amortecido se φ < 1) para toda a grade α × β de uma vez (vetorizado)."""
    A = alphas[:, None]
    B = betas[None, :]
    forma = (alphas.size, betas.size)
    nivel = np.full(forma, a[0])
    tend = np.full(forma, a[1] - a[0])
    sse = np.zeros(forma)
    for t in range(a.size):
        yhat = nivel + phi * tend
        if t >= 1:
            sse += (a[t] - yhat) ** 2
        novo = A * a[t] + (1 - A) * (nivel + phi * tend)          # FPP3, §8.2
        tend = B * (novo - nivel) + (1 - B) * phi * tend
        nivel = novo
    return sse, nivel, tend


def ajustar_holt(y: Sequence[float], passo: float = PASSO_GRADE, phi: float = 1.0) -> ModeloSuavizacao:
    """Método de Holt (tendência linear; amortecida se ``phi < 1``) por busca em grade de α × β (FPP3, §8.2).

    ``ℓ₀ = y₁``, ``b₀ = y₂ − y₁``; SSE de um passo a partir de t = 2. Em empate, fica o primeiro da grade.
    CT-3 (treino de 9 pontos): α* = 0,35, β* = 1,00 (fronteira → alerta), SSE = 369,1182.
    CT-3 (12 pontos): α* = 0,34, β* = 0,98, SSE = 453,3634, ℓ_T = 182,165145, b_T = 3,782945.
    """
    a = _serie(y, 3, "O Holt")
    if not (0 < phi <= 1):
        raise ErroPrevisao("O amortecimento φ precisa estar em (0; 1].")
    g = _grade(passo)
    sse, nivel, tend = _filtro_holt(a, g, g, phi)
    i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
    alertas = [x for x in (
        _alerta_fronteira("α", float(g[i]), "colete mais histórico antes de adotar o modelo."),
        _alerta_fronteira("β", float(g[j]), "considere Holt amortecido (φ < 1) ou colete mais histórico.")) if x]
    return ModeloSuavizacao("holt" if phi == 1.0 else "holt_amortecido", float(g[i]), float(g[j]), float(phi),
                            float(sse[i, j]), float(a[0]), float(a[1] - a[0]), float(nivel[i, j]), float(tend[i, j]),
                            a.size, alertas)


def ajustar_holt_amortecido(y: Sequence[float], passo: float = PASSO_GRADE,
                            phis: Sequence[float] = FAIXA_PHI) -> ModeloSuavizacao:
    """Holt amortecido (Gardner; McKenzie, 1985): grade α × β para cada φ ∈ ``phis`` (0,80 a 0,98) e fica o menor SSE.

    A tendência decai geometricamente e a previsão converge para ``ℓ_T + φ·b_T/(1−φ)`` — escolha prudente para
    horizontes longos de quadro de pessoal (FPP3, §8.2; Gardner, 2006).
    """
    melhor: ModeloSuavizacao | None = None
    for phi in phis:
        mod = ajustar_holt(y, passo, float(phi))
        if melhor is None or mod.sse < melhor.sse:
            melhor = mod
    assert melhor is not None
    melhor.tipo = "holt_amortecido"
    return melhor


def residuos_um_passo(modelo: ModeloSuavizacao, y: Sequence[float]) -> np.ndarray:
    """Resíduos de um passo do ajuste (t ≥ 2) — base do bootstrap (FPP3, §5.5).

    Resíduo (treino, um passo) ≠ erro de previsão (teste, vários passos) — FPP3, §5.8.
    """
    a = _serie(y, 2, "O cálculo de resíduos")
    res = []
    nivel, tend = a[0], (a[1] - a[0] if modelo.tipo != "ses" else 0.0)
    for t in range(a.size):
        yhat = nivel if modelo.tipo == "ses" else nivel + modelo.phi * tend
        if t >= 1:
            res.append(a[t] - yhat)
        if modelo.tipo == "ses":
            nivel = modelo.alpha * a[t] + (1 - modelo.alpha) * nivel
        else:
            novo = modelo.alpha * a[t] + (1 - modelo.alpha) * (nivel + modelo.phi * tend)
            tend = modelo.beta * (novo - nivel) + (1 - modelo.beta) * modelo.phi * tend
            nivel = novo
    return np.array(res)


def desvio_residuos(residuos: Sequence[float], n_parametros: int = 0) -> float:
    """``σ̂ = √(Σe² / (m − K))``, m resíduos e K parâmetros estimados (FPP3, §5.5). CT-3: 11 resíduos, K = 2 → 7,0974."""
    r = np.asarray(residuos, dtype=float)
    gl = r.size - n_parametros
    if gl <= 0:
        raise ErroPrevisao("Graus de liberdade insuficientes para estimar o desvio dos resíduos.")
    return float(math.sqrt(float(np.sum(r ** 2)) / gl))


def intervalo_normal(previsao: float, desvio: float, c: float = 1.28) -> tuple[float, float]:
    """Intervalo ``ŷ ± c·σ̂`` (FPP3, §5.5): c = 1,28 (80%) ou 1,96 (95%). Válido para h = 1; para h > 1 use o
    bootstrap. CT-3: 185,948 ± 9,085 → [176,86; 195,03]."""
    return previsao - c * desvio, previsao + c * desvio


# ----------------------------------------------------------------------------------------------------------------
# 3. Decomposição STL e força de tendência/sazonalidade — FPP3, §§3.6 e 4.3
# ----------------------------------------------------------------------------------------------------------------
def decompor_stl(y: Sequence[float], m: int, robusto: bool = True) -> dict[str, np.ndarray | float]:
    """Decomposição STL (Cleveland et al., 1990) via ``statsmodels``: ``y = T + S + R`` (aditiva).

    Devolve ``tendencia``, ``sazonal``, ``residuo`` e as forças ``forca_tendencia`` e ``forca_sazonal`` de FPP3, §4.3:
    ``F_T = max(0, 1 − Var(R)/Var(T + R))`` e ``F_S = max(0, 1 − Var(R)/Var(S + R))``. Exige ao menos 2 ciclos.
    Para sazonalidade multiplicativa, aplique a função ao logaritmo da série (FPP3, §3.6).
    """
    a = _serie(y, 2 * int(m), "A decomposição STL")
    from statsmodels.tsa.seasonal import STL

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = STL(a, period=int(m), robust=robusto).fit()
    T, S, R = np.asarray(r.trend), np.asarray(r.seasonal), np.asarray(r.resid)
    vr = float(np.var(R))
    ft = max(0.0, 1 - vr / float(np.var(T + R))) if np.var(T + R) > 0 else 0.0
    fs = max(0.0, 1 - vr / float(np.var(S + R))) if np.var(S + R) > 0 else 0.0
    return {"tendencia": T, "sazonal": S, "residuo": R, "forca_tendencia": ft, "forca_sazonal": fs}


# ----------------------------------------------------------------------------------------------------------------
# 4. Holt-Winters e ARIMA automático (statsmodels)
# ----------------------------------------------------------------------------------------------------------------
def ajustar_holt_winters(y: Sequence[float], m: int, sazonal: str = "add", amortecido: bool = False):
    """Holt-Winters (FPP3, §8.3) via ``statsmodels.tsa.holtwinters.ExponentialSmoothing``, estado inicial estimado.

    ``sazonal="add"`` quando a amplitude sazonal é constante; ``"mul"`` quando cresce com o nível (exige y > 0).
    Devolve o objeto de resultados do statsmodels (``.forecast(h)``, ``.simulate(...)``).
    """
    a = _serie(y, 2 * int(m), "O Holt-Winters")
    if sazonal == "mul" and np.any(a <= 0):
        raise ErroPrevisao("Holt-Winters multiplicativo exige série estritamente positiva.")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(a, trend="add", damped_trend=amortecido, seasonal=sazonal,
                                    seasonal_periods=int(m), initialization_method="estimated").fit()


def numero_diferencas(y: Sequence[float], alfa: float = 0.05, max_d: int = 2) -> int:
    """Número de diferenças d por testes KPSS sucessivos (H₀: estacionária) — FPP3, §9.1 (``unitroot_ndiffs``)."""
    from statsmodels.tsa.stattools import kpss

    x = _serie(y, 3, "O teste KPSS")
    d = 0
    while d < max_d and x.size >= 6 and np.ptp(x) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = kpss(x, regression="c", nlags="auto")[1]
        if p >= alfa:
            break
        d += 1
        x = np.diff(x)
    return d


@dataclass
class ModeloArima:
    """ARIMA escolhido pela busca automática: ordens, constante, AICc e o resultado do statsmodels."""

    ordem: tuple[int, int, int]
    ordem_sazonal: tuple[int, int, int, int]
    constante: bool
    aicc: float
    resultado: object

    def prever(self, h: int) -> np.ndarray:
        return np.asarray(self.resultado.forecast(_horizonte(h)), dtype=float)

    def descricao(self) -> str:
        p, d, q = self.ordem
        P, D, Q, m = self.ordem_sazonal
        s = f"ARIMA({p},{d},{q})" + (f"({P},{D},{Q})[{m}]" if (P or D or Q) else "")
        return s + (" com constante/drift" if self.constante else "")


def _ajustar_sarima(a, p, d, q, P, D, Q, m, constante):
    from statsmodels.tsa.arima.model import ARIMA

    k = d + D
    if constante and k >= 2:
        return None
    trend = ("c" if k == 0 else "t") if constante else "n"
    sazonal = (P, D, Q, m) if (P or D or Q) else (0, 0, 0, 0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ARIMA(a, order=(p, d, q), seasonal_order=sazonal, trend=trend).fit()
        aicc = float(r.aicc)
    except Exception:
        return None
    if not np.isfinite(aicc):
        return None
    return ModeloArima((p, d, q), sazonal, constante, aicc, r)


def ajustar_arima_auto(y: Sequence[float], m: int | None = None, max_p: int = 3, max_q: int = 3) -> ModeloArima:
    """ARIMA automático simplificado, no espírito de Hyndman e Khandakar (2008) / FPP3, §9.7.

    1. D = 1 se a força sazonal F_S ≥ 0,64 (FPP3, §9.1) e houver ≥ 2 ciclos; d por KPSS sucessivos (máx. 2);
    2. modelos iniciais (2,d,2), (0,d,0), (1,d,0), (0,d,1) — e os sazonais (1,D,1), (0,D,0), (1,D,0), (0,D,1)
       quando D = 1 —, com constante se d + D < 2, mais (0,d,0) sem constante;
    3. busca passo a passo: p, q (e P, Q ∈ {0, 1}) ± 1 e liga/desliga constante, enquanto o AICc cair.
    É uma versão didática (sem as aproximações do pacote R ``forecast``); para uso pesado, prefira ``pmdarima``/``fable``.
    """
    a = _serie(y, 8, "O ARIMA automático")
    usar_saz = bool(m and m > 1 and a.size >= 2 * m)
    D = 0
    if usar_saz and decompor_stl(a, int(m))["forca_sazonal"] >= LIMIAR_SAZONALIDADE:
        D = 1
    mm = int(m) if usar_saz else 0
    base = a[mm:] - a[:-mm] if D else a
    d = numero_diferencas(base, max_d=2 if not D else 1)
    cs = d + D < 2
    iniciais = [(2, 2, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)] if D else [(2, 2, 0, 0), (0, 0, 0, 0),
                                                                                     (1, 0, 0, 0), (0, 1, 0, 0)]
    vistos: dict[tuple, ModeloArima | None] = {}

    def tentar(p, q, P, Q, c):
        chave = (p, q, P, Q, c)
        if chave not in vistos:
            vistos[chave] = _ajustar_sarima(a, p, d, q, P, D, Q, mm, c)
        return vistos[chave]

    candidatos = [tentar(p, q, P, Q, cs) for p, q, P, Q in iniciais] + [tentar(0, 0, 0, 0, False)]
    candidatos = [c for c in candidatos if c is not None]
    if not candidatos:
        raise ErroPrevisao("Nenhum modelo ARIMA pôde ser ajustado a esta série.")
    atual = min(candidatos, key=lambda c: c.aicc)
    melhorou = True
    while melhorou:
        melhorou = False
        p0, _, q0 = atual.ordem
        P0, _, Q0, _ = atual.ordem_sazonal
        vizinhos = []
        for dp, dq in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
            vizinhos.append((p0 + dp, q0 + dq, P0, Q0, atual.constante))
        if D:
            for dP, dQ in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                vizinhos.append((p0, q0, P0 + dP, Q0 + dQ, atual.constante))
        if cs:
            vizinhos.append((p0, q0, P0, Q0, not atual.constante))
        for p, q, P, Q, c in vizinhos:
            if not (0 <= p <= max_p and 0 <= q <= max_q and 0 <= P <= 1 and 0 <= Q <= 1):
                continue
            mod = tentar(p, q, P, Q, c)
            if mod is not None and mod.aicc < atual.aicc - 1e-9:
                atual, melhorou = mod, True
                break
    return atual


# ----------------------------------------------------------------------------------------------------------------
# 5. Demanda intermitente — Croston (1972), SBA (Syntetos; Boylan, 2005), TSB (Teunter; Syntetos; Babai, 2011)
# ----------------------------------------------------------------------------------------------------------------
def classificar_demanda(y: Sequence[float]) -> dict[str, float | str | None]:
    """Classifica o padrão de demanda por ADI (intervalo médio entre demandas) e CV² dos tamanhos não nulos.

    Fronteiras ADI = 1,32 e CV² = 0,49 (Syntetos, Boylan e Croston, 2005): **suave** (ADI < 1,32; CV² < 0,49),
    **errática** (ADI < 1,32; CV² ≥ 0,49), **intermitente** (ADI ≥ 1,32; CV² < 0,49) e **irregular** (*lumpy*,
    ADI ≥ 1,32; CV² ≥ 0,49). ``intermitente_ou_irregular`` indica quando usar Croston/SBA/TSB.
    """
    a = _serie(y, 1, "A classificação de demanda")
    nz = a[a > 0]
    prop_zeros = float(np.mean(a == 0))
    if nz.size == 0:
        return {"adi": None, "cv2": None, "proporcao_zeros": prop_zeros, "classe": "sem demanda",
                "intermitente_ou_irregular": True}
    adi = a.size / nz.size
    cv2 = float((nz.std() / nz.mean()) ** 2) if nz.size > 1 else 0.0
    classe = ("suave" if cv2 < LIMIAR_CV2 else "errática") if adi < LIMIAR_ADI else (
        "intermitente" if cv2 < LIMIAR_CV2 else "irregular")
    return {"adi": float(adi), "cv2": cv2, "proporcao_zeros": prop_zeros, "classe": classe,
            "intermitente_ou_irregular": adi >= LIMIAR_ADI}


def prever_croston(y: Sequence[float], h: int, alpha: float = 0.1, variante: str = "classico") -> np.ndarray:
    """Croston (1972): SES separado para o tamanho ``q`` das demandas não nulas e o intervalo ``a`` entre elas;
    previsão plana ``q̂/â`` (FPP3, §13.2). ``variante="sba"`` aplica a correção de viés ``(1 − α/2)`` de Syntetos e
    Boylan (2005). Inicialização de Croston: primeira demanda e primeiro intervalo; α = 0,1. Sem intervalo de previsão
    algébrico (o método não corresponde a um modelo estatístico — FPP3, §13.2).
    """
    a = _serie(y, 2, "O Croston")
    h = _horizonte(h)
    idx = np.flatnonzero(a > 0)
    if idx.size == 0:
        return np.zeros(h)
    q = a[idx[0]]
    intervalo = float(idx[0] + 1)
    ultimo = idx[0]
    for i in idx[1:]:
        q = alpha * a[i] + (1 - alpha) * q
        intervalo = alpha * (i - ultimo) + (1 - alpha) * intervalo
        ultimo = i
    f = q / intervalo
    if variante == "sba":
        f *= 1 - alpha / 2
    return np.full(h, f)


def prever_tsb(y: Sequence[float], h: int, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    """TSB (Teunter, Syntetos e Babai, 2011): suaviza a **probabilidade** de haver demanda em todo período (β) e o
    tamanho só quando há demanda (α); previsão ``p̂·ẑ``. Ao contrário do Croston, decai durante longos períodos sem
    demanda (obsolescência) e é não viesado. Inicialização: p₀ = proporção de períodos com demanda; z₀ = média dos
    tamanhos não nulos.
    """
    a = _serie(y, 2, "O TSB")
    h = _horizonte(h)
    nz = a[a > 0]
    if nz.size == 0:
        return np.zeros(h)
    p, z = nz.size / a.size, float(nz.mean())
    for v in a:
        if v > 0:
            p = beta * 1 + (1 - beta) * p
            z = alpha * v + (1 - alpha) * z
        else:
            p = (1 - beta) * p
    return np.full(h, p * z)


# ----------------------------------------------------------------------------------------------------------------
# 6. Medidas de erro — FPP3, §5.8; Hyndman e Koehler (2006)
# ----------------------------------------------------------------------------------------------------------------
def escala_mase(treino: Sequence[float], m: int | None = None) -> float | None:
    """Escala do MASE: média de |y_t − y_{t−m}| no treino (m = 1 se não sazonal ou série curta). ``None`` se 0."""
    a = np.asarray(treino, dtype=float)
    lag = int(m) if (m and m > 1 and a.size >= 2 * m) else 1
    if a.size <= lag:
        return None
    s = float(np.mean(np.abs(a[lag:] - a[:-lag])))
    return s if s > 0 else None


def medidas_erro(reais: Sequence[float], previstos: Sequence[float], treino: Sequence[float] | None = None,
                 m: int | None = None, escalas: Sequence[float] | None = None) -> dict:
    """MAE, RMSE, MAPE e MASE de ``e = y − ŷ`` (FPP3, §5.8).

    - RMSE é o **critério de seleção** (FPP3, §5.10); MAE é a leitura de negócio ("erra, em média, X por mês").
    - MAPE fica ``None`` (com alerta) se algum observado ≤ 0: explode perto de zero e pune assimetricamente.
    - MASE usa a escala do ingênuo (sazonal, se ``m``) no treino; ou ``escalas`` por ponto (origem móvel).
      MASE < 1: melhor que o ingênuo dentro da amostra (Hyndman; Koehler, 2006).
    """
    r = np.asarray(reais, dtype=float)
    p = np.asarray(previstos, dtype=float)
    if r.size == 0 or r.size != p.size:
        raise ErroPrevisao("Reais e previstos precisam ter o mesmo tamanho, não nulo.")
    e = r - p
    alertas = []
    mape = float(np.mean(np.abs(100 * e / r))) if np.all(r > 0) else None
    if mape is None:
        alertas.append("MAPE indefinido: há valor observado ≤ 0. Use MASE (FPP3, §5.8).")
    mase = None
    if escalas is not None:
        esc = np.asarray(escalas, dtype=float)
        if np.all(np.isfinite(esc)) and np.all(esc > 0):
            mase = float(np.mean(np.abs(e) / esc))
    elif treino is not None:
        s = escala_mase(treino, m)
        mase = float(np.mean(np.abs(e)) / s) if s else None
    if mase is None:
        alertas.append("MASE não calculado (escala ingênua nula ou treino ausente).")
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "mape": mape, "mase": mase,
            "n": int(e.size), "alertas": alertas}


# ----------------------------------------------------------------------------------------------------------------
# 7. Catálogo de métodos candidatos
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Metodo:
    """Método candidato: ``prever(y, h)`` devolve as previsões ``1..h`` ajustando o método em ``y``."""

    chave: str
    nome: str
    complexidade: int
    minimo: int
    prever: Callable[[np.ndarray, int], np.ndarray]


def catalogo(m: int | None = 12) -> dict[str, Metodo]:
    """Todos os métodos disponíveis, em ordem de complexidade (usada no desempate: vence o mais simples)."""
    mm = int(m) if m and m > 1 else None
    c: dict[str, Metodo] = {
        "media": Metodo("media", "Média", 1, 2, prever_media),
        "ingenuo": Metodo("ingenuo", "Ingênuo", 1, 2, prever_ingenuo),
        "drift": Metodo("drift", "Drift (reta 1.º–último)", 1, 3, prever_drift),
        "ses": Metodo("ses", "SES", 2, 3, lambda y, h: ajustar_ses(y).prever(h)),
        "croston": Metodo("croston", "Croston", 2, 6, lambda y, h: prever_croston(y, h)),
        "sba": Metodo("sba", "Croston-SBA", 2, 6, lambda y, h: prever_croston(y, h, variante="sba")),
        "tsb": Metodo("tsb", "TSB", 2, 6, lambda y, h: prever_tsb(y, h)),
        "holt": Metodo("holt", "Holt", 3, 4, lambda y, h: ajustar_holt(y).prever(h)),
        "holt_amortecido": Metodo("holt_amortecido", "Holt amortecido", 4, 4,
                                  lambda y, h: ajustar_holt_amortecido(y).prever(h)),
    }
    if mm:
        c["sazonal_ingenuo"] = Metodo("sazonal_ingenuo", "Ingênuo sazonal", 1, mm,
                                      lambda y, h: prever_sazonal_ingenuo(y, h, mm))
        c["hw_aditivo"] = Metodo("hw_aditivo", "Holt-Winters aditivo", 5, 2 * mm,
                                 lambda y, h: np.asarray(ajustar_holt_winters(y, mm, "add").forecast(h)))
        c["hw_multiplicativo"] = Metodo("hw_multiplicativo", "Holt-Winters multiplicativo", 5, 2 * mm,
                                        lambda y, h: np.asarray(ajustar_holt_winters(y, mm, "mul").forecast(h)))
    c["arima"] = Metodo("arima", "ARIMA automático", 6, max(16, 2 * mm if mm else 16),
                        lambda y, h: ajustar_arima_auto(y, mm).prever(h))
    return dict(sorted(c.items(), key=lambda kv: kv[1].complexidade))


# ----------------------------------------------------------------------------------------------------------------
# 8. Árvore de decisão por tamanho de série — L2, §1.10, ampliada
# ----------------------------------------------------------------------------------------------------------------
def recomendar_metodos(y: Sequence[float], m: int | None = 12) -> dict:
    """Árvore de decisão: quais métodos são candidatos para esta série.

    - n < 12 → **não modelar**: usar a quantidade planejada ou estimativa estruturada de gestores (artigo-base,
      Fase 4; FPP3, §6.2); os métodos de referência saem só como referência.
    - demanda intermitente/irregular (ADI ≥ 1,32) → média, ingênuo, Croston, SBA, TSB (medir por MAE/MASE).
    - 12 ≤ n < 2m + 3 → referências sem sazonal + SES + Holt + Holt amortecido (sazonalidade não é estimável com
      menos de dois ciclos).
    - n ≥ 2m + 3 → acrescenta ingênuo sazonal; se F_S ≥ 0,64, Holt-Winters aditivo (e multiplicativo se y > 0);
      ARIMA automático quando n ≥ 3m (sazonal) ou n ≥ 24 (não sazonal).
    Em todos os ramos a escolha final é por validação (menor RMSE; empate → o mais simples).
    """
    a = _serie(y, 1, "A recomendação")
    n = a.size
    mm = int(m) if m and m > 1 else None
    dem = classificar_demanda(a)
    just: list[str] = []
    out = {"n": n, "m": mm, "demanda": dem, "forca_sazonal": None, "forca_tendencia": None}
    if n < 12:
        just.append(f"Série com {n} pontos (< 12): não modelar. Use a quantidade definida em planejamento ou a "
                    "estimativa estruturada de gestores, com justificativa registrada (artigo-base, Fase 4; FPP3 §6.2).")
        return {**out, "ramo": "n < 12", "modelar": False, "metodos": ["media", "ingenuo"], "justificativa": just}
    if dem["intermitente_ou_irregular"] and dem["proporcao_zeros"] > 0:
        just.append(f"Demanda {dem['classe']} (ADI = {br(dem['adi'] or 0)}; CV² = {br(dem['cv2'] or 0)}): "
                    "métodos para demanda intermitente (Croston, SBA, TSB). MAPE não se aplica (há zeros).")
        return {**out, "ramo": "intermitente", "modelar": True,
                "metodos": ["media", "ingenuo", "croston", "sba", "tsb"], "justificativa": just}
    metodos = ["media", "ingenuo", "drift", "ses", "holt", "holt_amortecido"]
    if not mm or n < 2 * mm + 3:
        just.append(f"Série com {n} pontos (menos de dois ciclos completos e folga para teste): métodos de "
                    "referência + SES + Holt + Holt amortecido; sem Holt-Winters (L2 §1.10).")
        if n >= 24 and not mm:
            metodos.append("arima")
        return {**out, "ramo": "12 ≤ n < 2m+3", "modelar": True, "metodos": metodos, "justificativa": just}
    stl = decompor_stl(a, mm)
    out["forca_sazonal"], out["forca_tendencia"] = stl["forca_sazonal"], stl["forca_tendencia"]
    metodos.insert(2, "sazonal_ingenuo")
    if stl["forca_sazonal"] >= LIMIAR_SAZONALIDADE:
        metodos.append("hw_aditivo")
        if np.all(a > 0):
            metodos.append("hw_multiplicativo")
        just.append(f"Sazonalidade relevante (F_S = {br(stl['forca_sazonal'])} ≥ 0,64): Holt-Winters entra entre os "
                    "candidatos; aditivo × multiplicativo decidido pela validação.")
    else:
        just.append(f"Sazonalidade fraca (F_S = {br(stl['forca_sazonal'])} < 0,64): sem Holt-Winters.")
    if n >= 3 * mm:
        metodos.append("arima")
        just.append(f"Série com {n} pontos (≥ 3 ciclos): ARIMA automático também entra.")
    return {**out, "ramo": "n ≥ 2m+3", "modelar": True, "metodos": metodos, "justificativa": just}


# ----------------------------------------------------------------------------------------------------------------
# 9. Validação: treino/teste e origem móvel (sem embaralhar) — FPP3, §§5.8 e 5.10; Tashman (2000)
# ----------------------------------------------------------------------------------------------------------------
def _resolver(metodos, m) -> list[Metodo]:
    cat = catalogo(m)
    saida = []
    for x in metodos:
        if isinstance(x, Metodo):
            saida.append(x)
        elif x in cat:
            saida.append(cat[x])
        else:
            raise ErroPrevisao(f"Método desconhecido: {x}. Disponíveis: {', '.join(cat)}.")
    return saida


def avaliar_treino_teste(y: Sequence[float], n_teste: int, metodos: Sequence[str | Metodo],
                         m: int | None = None) -> dict[str, dict]:
    """Divisão única: treino = primeiros ``n − n_teste`` pontos; teste = os últimos (FPP3, §5.8).

    O teste deve ter ao menos o tamanho do horizonte que se quer avaliar. Devolve, por método, as previsões e as
    medidas de erro. CT-3 (9 + 3): SES RMSE 7,4145; Holt RMSE 6,2044.
    """
    a = _serie(y, n_teste + 2, "A validação treino/teste")
    treino, teste = a[:-n_teste], a[-n_teste:]
    out = {}
    for met in _resolver(metodos, m):
        prev = met.prever(treino, n_teste)
        out[met.chave] = {"nome": met.nome, "complexidade": met.complexidade, "previsoes": prev,
                          **medidas_erro(teste, prev, treino, m)}
    return out


def validacao_origem_movel(y: Sequence[float], metodos: Sequence[str | Metodo], m: int | None = None,
                           inicio: int | None = None, h_validacao: int = 1, passo: int = 1) -> dict:
    """Validação cruzada de séries temporais por **origem móvel** (FPP3, §5.10; Tashman, 2000).

    Para cada origem t = inicio, inicio + passo, …, n − 1: ajusta cada método **só** em ``y[0:t]`` e prevê
    ``y[t : t + h_validacao]`` (os passos que existirem). Nunca embaralha: todo índice de treino é anterior a todo
    índice de teste da mesma origem. Todos os métodos são avaliados nas mesmas origens. ``inicio`` padrão:
    o maior entre o mínimo exigido pelos métodos e ⌈n/2⌉ (limitado a n − 3). Devolve ``{"origens": [...],
    "metricas": {chave: {...}}}``; cada origem traz ``(fim_treino, indices_teste)``.
    """
    a = _serie(y, 5, "A validação de origem móvel")
    n = a.size
    mets = _resolver(metodos, m)
    if inicio is None:
        inicio = max(max(mt.minimo for mt in mets), math.ceil(n / 2))
        inicio = min(inicio, n - MIN_ORIGENS)
    if inicio < 2 or inicio >= n:
        raise ErroPrevisao("Origem inicial inválida para o tamanho da série.")
    origens = []
    for t in range(inicio, n, max(1, int(passo))):
        idx = list(range(t, min(t + h_validacao, n)))
        assert max(range(t)) < min(idx), "vazamento: treino precisa anteceder o teste"
        origens.append((t, idx))
    metricas = {}
    for mt in mets:
        if mt.minimo > inicio:
            metricas[mt.chave] = {"nome": mt.nome, "complexidade": mt.complexidade, "erro":
                                  f"exige {mt.minimo} pontos; a primeira origem tem {inicio}"}
            continue
        reais, prev, escalas = [], [], []
        try:
            for t, idx in origens:
                treino = a[:t]
                f = mt.prever(treino, len(idx))
                s = escala_mase(treino, m)
                reais.extend(a[idx])
                prev.extend(f[: len(idx)])
                escalas.extend([s if s else np.nan] * len(idx))
            metricas[mt.chave] = {"nome": mt.nome, "complexidade": mt.complexidade,
                                  **medidas_erro(reais, prev, escalas=escalas)}
        except Exception as exc:  # um método que falha não derruba a comparação
            metricas[mt.chave] = {"nome": mt.nome, "complexidade": mt.complexidade, "erro": str(exc)}
    return {"inicio": inicio, "h_validacao": h_validacao, "origens": origens, "metricas": metricas}


def escolher_metodo(metricas: dict[str, dict], criterio: str = "rmse") -> str:
    """Escolhe o método de menor ``criterio`` (RMSE por padrão, FPP3 §5.10); empate → menor complexidade.

    O modelo sofisticado só é escolhido se vencer os métodos de referência, porque eles estão entre os candidatos.
    """
    validos = {k: v for k, v in metricas.items() if v.get(criterio) is not None and "erro" not in v}
    if not validos:
        raise ErroPrevisao("Nenhum método produziu medida de erro válida.")
    return min(validos, key=lambda k: (round(validos[k][criterio], 9), validos[k]["complexidade"]))


# ----------------------------------------------------------------------------------------------------------------
# 10. Projeção com intervalo por bootstrap dos resíduos — FPP3, §5.5
# ----------------------------------------------------------------------------------------------------------------
def _percentis(caminhos: np.ndarray, nivel: float):
    cauda = (100 - nivel) / 2
    return np.percentile(caminhos, cauda, axis=0), np.percentile(caminhos, 100 - cauda, axis=0)


def projetar(y: Sequence[float], chave: str, h: int, m: int | None = None, nivel: float = 80,
             n_sim: int = N_SIM_BOOTSTRAP, semente: int = SEMENTE_PREVISAO) -> dict:
    """Ajusta o método ``chave`` na série inteira e projeta ``h`` passos com intervalo de ``nivel``%.

    Intervalo: **bootstrap dos resíduos de um passo** (FPP3, §5.5) — reamostra resíduos com reposição e propaga o
    caminho pelas equações do próprio método (SES/Holt: atualiza nível e tendência com o valor simulado; ingênuo:
    ``y_{T+i} = y_{T+i−1} + e*``; drift: idem + inclinação; ingênuo sazonal: ``y_{T+i−m} + e*``; média: ``ȳ + e*``).
    Holt-Winters: ``simulate(..., random_errors="bootstrap")`` do statsmodels. ARIMA: intervalo do modelo (normal).
    Croston/SBA/TSB: sem intervalo (FPP3, §13.2). Semente fixa → resultado reprodutível.
    """
    a = _serie(y, 2, "A projeção")
    h = _horizonte(h)
    mm = int(m) if m and m > 1 else None
    g = rng(semente)
    det: dict = {"chave": chave, "nivel": nivel, "semente": semente, "n_sim": n_sim, "alertas": [],
                 "tipo_intervalo": f"bootstrap dos resíduos, {n_sim} caminhos, semente {semente}"}
    inf = sup = None
    if chave in ("ses", "holt", "holt_amortecido"):
        mod = {"ses": ajustar_ses, "holt": ajustar_holt, "holt_amortecido": ajustar_holt_amortecido}[chave](a)
        prev = mod.prever(h)
        res = residuos_um_passo(mod, a)
        idx = g.integers(0, res.size, size=(n_sim, h))
        nivel_ = np.full(n_sim, mod.lT)
        tend = np.full(n_sim, mod.bT)
        cam = np.empty((n_sim, h))
        for i in range(h):
            yhat = nivel_ if mod.tipo == "ses" else nivel_ + mod.phi * tend
            ys = yhat + res[idx[:, i]]
            cam[:, i] = ys
            novo = mod.alpha * ys + (1 - mod.alpha) * (nivel_ if mod.tipo == "ses" else nivel_ + mod.phi * tend)
            if mod.tipo != "ses":
                tend = mod.beta * (novo - nivel_) + (1 - mod.beta) * mod.phi * tend
            nivel_ = novo
        inf, sup = _percentis(cam, nivel)
        k = 1 if mod.tipo == "ses" else (2 if mod.tipo == "holt" else 3)
        det.update(modelo=mod, residuos=res, desvio_residuos=desvio_residuos(res, k), alertas=list(mod.alertas))
    elif chave in ("media", "ingenuo", "drift", "sazonal_ingenuo"):
        if chave == "media":
            prev, res = prever_media(a, h), a - a.mean()
        elif chave == "ingenuo":
            prev, res = prever_ingenuo(a, h), np.diff(a)
        elif chave == "drift":
            c = (a[-1] - a[0]) / (a.size - 1)
            prev, res = prever_drift(a, h), np.diff(a) - c
        else:
            if not mm:
                raise ErroPrevisao("Ingênuo sazonal exige o período sazonal m.")
            prev, res = prever_sazonal_ingenuo(a, h, mm), a[mm:] - a[:-mm]
        idx = g.integers(0, res.size, size=(n_sim, h))
        e = res[idx]
        if chave == "media":
            cam = a.mean() + e
        elif chave in ("ingenuo", "drift"):
            c = (a[-1] - a[0]) / (a.size - 1) if chave == "drift" else 0.0
            cam = a[-1] + np.cumsum(e + c, axis=1)
        else:
            hist = np.tile(a, (n_sim, 1))
            for i in range(h):
                hist = np.column_stack([hist, hist[:, -mm] + e[:, i]])
            cam = hist[:, a.size:]
        inf, sup = _percentis(cam, nivel)
        det["residuos"] = res
    elif chave in ("hw_aditivo", "hw_multiplicativo"):
        if not mm:
            raise ErroPrevisao("Holt-Winters exige o período sazonal m.")
        r = ajustar_holt_winters(a, mm, "add" if chave == "hw_aditivo" else "mul")
        prev = np.asarray(r.forecast(h), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = np.asarray(r.simulate(h, repetitions=n_sim, error="add" if chave == "hw_aditivo" else "mul",
                                        random_errors="bootstrap", anchor="end", random_state=g), dtype=float)
        inf, sup = _percentis(sim.T, nivel)
        det["parametros"] = {k: float(v) for k, v in r.params.items() if np.isscalar(v) and v is not None
                             and k in ("smoothing_level", "smoothing_trend", "smoothing_seasonal")}
    elif chave == "arima":
        mod = ajustar_arima_auto(a, mm)
        fc = mod.resultado.get_forecast(h)
        prev = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=1 - nivel / 100), dtype=float)
        inf, sup = ci[:, 0], ci[:, 1]
        det.update(modelo=mod, descricao=mod.descricao(), tipo_intervalo="aproximação normal do modelo ARIMA")
    elif chave in ("croston", "sba", "tsb"):
        prev = prever_tsb(a, h) if chave == "tsb" else prever_croston(a, h, variante="sba" if chave == "sba" else
                                                                       "classico")
        det["alertas"].append("Croston/SBA/TSB não têm intervalo de previsão algébrico (FPP3, §13.2).")
        det["tipo_intervalo"] = "sem intervalo"
    else:
        raise ErroPrevisao(f"Método desconhecido para projeção: {chave}.")
    return {"previsao": np.asarray(prev, dtype=float), "inferior": None if inf is None else np.asarray(inf),
            "superior": None if sup is None else np.asarray(sup), "detalhes": det}


# ----------------------------------------------------------------------------------------------------------------
# 11. Vetores de demanda — decisão D4 do curso; L2, §§2.3 e 5.5; FPP3, §§6.2, 6.7 e 7.4
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Vetor:
    """Vetor de demanda: ajuste por julgamento documentado, aplicado **sobre** a projeção estatística.

    ``fator`` multiplicativo (1,12 = +12%); vigência ``inicio``..``fim`` em índices 1-based do horizonte (``fim``
    ``None`` = permanente); ``entregas`` a que se aplica (vazio = todas as da chamada); ``justificativa`` (≥ 200
    caracteres, com a evidência externa), ``fonte`` verificável, ``responsavel``, ``data_registro`` e, para fator > 1,
    ``aprovacao_segundo`` (anti-otimismo).
    """

    nome: str
    fator: float
    inicio: int
    justificativa: str
    fonte: str
    responsavel: str = ""
    data_registro: str = ""
    fim: int | None = None
    entregas: tuple[str, ...] = ()
    aprovacao_segundo: str = ""
    id: str = ""


def validar_vetor(v: Vetor) -> None:
    """Aplica as travas de D4; lança ``ErroVetor`` na primeira violação (vetor inválido não é aplicável)."""
    rot = f'"{v.nome}"' if v.nome else "(sem nome)"
    if not v.nome or not v.nome.strip():
        raise ErroVetor("O vetor precisa de um nome que o identifique.")
    if not np.isfinite(v.fator) or v.fator <= 0:
        raise ErroVetor(f"Vetor {rot}: o fator precisa ser um número finito maior que zero.")
    if FAIXA_BLOQUEADA[0] < v.fator < FAIXA_BLOQUEADA[1]:
        raise ErroVetor(f"Vetor {rot}: ajuste pequeno demais (fator {br(v.fator, 3)}). Ajustes entre −3% e +3% "
                        "tendem a piorar a acurácia (FPP3 §6.7; Fildes et al., 2009); a faixa 0,97 < fator < 1,03 é "
                        "bloqueada (D4).")
    if not v.justificativa or len(v.justificativa.strip()) < MIN_JUSTIFICATIVA:
        n = len(v.justificativa.strip()) if v.justificativa else 0
        raise ErroVetor(f"Vetor {rot}: a justificativa precisa ter ao menos {MIN_JUSTIFICATIVA} caracteres com a "
                        f"evidência externa (D4); recebidos {n}.")
    if not v.fonte or not v.fonte.strip():
        raise ErroVetor(f"Vetor {rot}: fonte verificável obrigatória (lei, decreto, ata, ofício, plano anual).")
    if not isinstance(v.inicio, (int, np.integer)) or v.inicio < 1:
        raise ErroVetor(f"Vetor {rot}: o início precisa ser um índice inteiro ≥ 1 do horizonte.")
    if v.fim is not None and (not isinstance(v.fim, (int, np.integer)) or v.fim < v.inicio):
        raise ErroVetor(f"Vetor {rot}: o fim precisa ser inteiro ≥ início.")


def _vigente(v: Vetor, h: int, entrega: str | None) -> bool:
    if h < v.inicio or (v.fim is not None and h > v.fim):
        return False
    return not (entrega is not None and v.entregas and entrega not in v.entregas)


def fator_composto(vetores: Sequence[Vetor], h: int, entrega: str | None = None) -> float:
    """``F(h) = Π fator_v`` dos vetores vigentes em h (composição multiplicativa, invariante à ordem). CT-3: 1,12 ×
    0,95 = 1,064 a partir de h = 7."""
    f = 1.0
    for v in vetores:
        if _vigente(v, h, entrega):
            f *= v.fator
    return f


def aplicar_vetores(base: Sequence[float], vetores: Sequence[Vetor], entrega: str | None = None,
                    inferior: Sequence[float] | None = None, superior: Sequence[float] | None = None) -> dict:
    """Aplica os vetores sobre a projeção base e devolve o **antes/depois** completo (nenhum número ajustado circula
    sem o número base ao lado — FPP3, §6.7; L2, §2.3).

    Se ``inferior``/``superior`` forem dados, o intervalo também é multiplicado por F(h) (o vetor é um efeito
    proporcional sobre o nível). CT-3: Σ base = 2.481,0515 → Σ ajustada = 2.637,1918 (Δ = +156,1404; +6,2933%).
    """
    b = np.asarray(base, dtype=float)
    if b.size == 0 or not np.all(np.isfinite(b)):
        raise ErroVetor("A projeção base precisa ser um vetor não vazio e finito.")
    for v in vetores:
        validar_vetor(v)
    fatores = np.array([fator_composto(vetores, i + 1, entrega) for i in range(b.size)])
    ajustada = b * fatores
    alertas = []
    for v in vetores:
        if v.fator > 1 and not v.aprovacao_segundo.strip():
            alertas.append(f'Alerta anti-otimismo: o vetor "{v.nome}" aumenta a demanda (fator {br(v.fator, 3)}) sem '
                           "aprovação de segundo responsável (FPP3 §6.7; D4).")
    if np.any(fatores < FAIXA_COMPOSICAO[0]) or np.any(fatores > FAIXA_COMPOSICAO[1]):
        alertas.append("Composição de vetores fora da faixa admitida [0,60; 1,60]: revise o modelo, não acrescente "
                       "vetores (L2, §2.3).")
    trilha = [{"id": v.id, "nome": v.nome, "fator": v.fator,
               "vigencia": f"h ≥ {v.inicio}" + (f" até h = {v.fim}" if v.fim else ", permanente"),
               "justificativa": v.justificativa, "fonte": v.fonte, "responsavel": v.responsavel,
               "data_registro": v.data_registro, "aprovacao_segundo": v.aprovacao_segundo} for v in vetores]
    tb, ta = float(b.sum()), float(ajustada.sum())
    return {"base": b, "ajustada": ajustada, "fatores": fatores, "total_base": tb, "total_ajustada": ta,
            "delta_absoluto": ta - tb, "delta_percentual": (ta / tb - 1) * 100 if tb else float("nan"),
            "inferior_ajustado": None if inferior is None else np.asarray(inferior) * fatores,
            "superior_ajustado": None if superior is None else np.asarray(superior) * fatores,
            "alertas": alertas, "trilha": trilha}


def ler_vetores_csv(caminho: str) -> list[Vetor]:
    """Lê vetores de um CSV (``;`` e vírgula decimal) com as colunas: id; nome; entregas (separadas por ``|``); fator;
    inicio; fim (vazio = permanente); fonte; responsavel; data_registro; aprovacao_segundo; justificativa."""
    df = ler_csv(caminho, dtype=str, keep_default_na=False)
    vs = []
    for _, r in df.iterrows():
        fim = str(r.get("fim", "")).strip()
        vs.append(Vetor(id=str(r.get("id", "")), nome=str(r["nome"]), fator=float(str(r["fator"]).replace(",", ".")),
                        inicio=int(r["inicio"]), fim=int(fim) if fim else None,
                        entregas=tuple(e.strip() for e in str(r.get("entregas", "")).split("|") if e.strip()),
                        fonte=str(r.get("fonte", "")), responsavel=str(r.get("responsavel", "")),
                        data_registro=str(r.get("data_registro", "")),
                        aprovacao_segundo=str(r.get("aprovacao_segundo", "")),
                        justificativa=str(r.get("justificativa", ""))))
    return vs


# ----------------------------------------------------------------------------------------------------------------
# 12. Gráfico e relatório
# ----------------------------------------------------------------------------------------------------------------
def grafico(y: Sequence[float], projecao: dict, caminho: str, vetores: Sequence[Vetor] = (), ajuste: dict | None = None,
            titulo: str = "Previsão de demanda", rotulos: Sequence[str] | None = None, unidade: str = "demanda/mês"):
    """Gráfico antes/depois: histórico, projeção base (tracejada), faixa do intervalo, série ajustada pelos vetores
    (sólida) e o início de vigência de cada vetor sombreado. Salva em ``caminho`` (PNG)."""
    plt = estilo_graficos()
    a = np.asarray(y, dtype=float)
    n, h = a.size, projecao["previsao"].size
    xh, xf = np.arange(1, n + 1), np.arange(n + 1, n + h + 1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(xh, a, color=PALETA["marinho"], lw=2, label="Histórico")
    if projecao.get("inferior") is not None:
        ax.fill_between(xf, projecao["inferior"], projecao["superior"], color=PALETA["azul"], alpha=0.15,
                        label=f"Intervalo de {br(projecao['detalhes']['nivel'], 0)}% (base)")
    ax.plot(np.r_[n, xf], np.r_[a[-1], projecao["previsao"]], color=PALETA["azul"], lw=2, ls="--",
            label="Projeção base (estatística)")
    if ajuste is not None:
        ax.plot(np.r_[n, xf], np.r_[a[-1], ajuste["ajustada"]], color=PALETA["verde_escuro"], lw=2.4,
                label="Projeção ajustada (vetores)")
    # folga no topo para as etiquetas dos vetores (horizontais, escalonadas: não se sobrepõem às curvas)
    nv = len(vetores) if ajuste is not None else 0
    etiquetas = []
    y0, y1 = ax.get_ylim()
    if nv:
        ax.set_ylim(y0, y0 + (y1 - y0) / max(0.55, 0.86 - 0.075 * nv))
        import matplotlib.transforms as mtransforms
        mistura = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for k, v in enumerate(vetores):
            fim = n + (v.fim if v.fim else h) + 0.5
            ax.axvspan(n + v.inicio - 0.5, fim, color=PALETA["verde"], alpha=0.07)
            nome = v.nome if len(v.nome) <= 38 else v.nome[:38].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
            etq = ax.text(n + v.inicio - 0.35, 0.975 - 0.075 * k, f"{v.id}: {nome} (×{br(v.fator, 2)})" if getattr(v, "id", "")
                    else f"{nome} (×{br(v.fator, 2)})", transform=mistura, fontsize=8.5, va="top", ha="left",
                    color=PALETA["verde_escuro"], clip_on=False,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
            etiquetas.append(etq)
    ax.axvline(n + 0.5, color=PALETA["linha"], lw=1)
    ax.set_xlim(0.3, n + h + 0.7)
    if etiquetas:  # etiqueta que passaria da borda direita é alinhada à direita, dentro do gráfico
        rend = fig.canvas.get_renderer()
        borda = ax.get_window_extent(rend).x1
        for etq in etiquetas:
            if etq.get_window_extent(rend).x1 > borda:
                etq.set_x(n + h + 0.6)
                etq.set_horizontalalignment("right")
    if rotulos is not None and len(rotulos) == n:
        todos = [mes_br(r) for r in rotulos] + (meses_seguintes(rotulos[-1], h) or [""] * h)
        passo = max(1, -(-(n + h) // 12))
        ticks = list(range(1, n + h + 1, passo))
        ax.set_xticks(ticks)
        ax.set_xticklabels([todos[i - 1] for i in ticks], rotation=45, ha="right", fontsize=9)
    else:
        ax.set_xlabel("período")
    ax.set_title(titulo, loc="left", color=PALETA["marinho"])
    ax.set_ylabel(unidade)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    formatar_eixos_br(fig)
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def _fmt(x, casas=2):
    return "—" if x is None else br(x, casas)


def tabela_metricas(metricas: dict[str, dict], escolhido: str | None = None) -> str:
    """Tabela de texto com MAE, RMSE, MAPE e MASE por método, ordenada por RMSE."""
    linhas = [f"  {'Método':<30}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}{'MASE':>8}"]
    ordem = sorted(metricas, key=lambda k: (metricas[k].get("rmse") is None, metricas[k].get("rmse") or 0))
    for k in ordem:
        v = metricas[k]
        marca = " ◄ escolhido" if k == escolhido else ""
        if "erro" in v:
            linhas.append(f"  {v['nome']:<30}(não avaliado: {v['erro']})")
            continue
        linhas.append(f"  {v['nome']:<30}{_fmt(v['mae']):>10}{_fmt(v['rmse']):>10}{_fmt(v['mape']):>10}"
                      f"{_fmt(v['mase']):>8}{marca}")
    return "\n".join(linhas)


# ----------------------------------------------------------------------------------------------------------------
# 13. Linha de comando
# ----------------------------------------------------------------------------------------------------------------
def _ler_serie(caminho: str, coluna: str | None):
    df = ler_csv(caminho)
    if coluna:
        if coluna not in df.columns:
            raise ErroPrevisao(f"Coluna '{coluna}' não encontrada. Colunas: {', '.join(df.columns)}.")
        col = coluna
    else:
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            raise ErroPrevisao("Nenhuma coluna numérica no CSV; informe --coluna.")
        col = num[-1]
    rot = df[df.columns[0]].astype(str).tolist() if df.columns[0] != col else None
    return df[col].to_numpy(dtype=float), col, rot


def cli(argv: list[str]) -> int:
    """``dft previsao serie.csv [--horizonte 12] [--periodo 12] [--vetores v.csv] [--grafico saida.png] ...``"""
    p = argparse.ArgumentParser(prog="dft previsao", description="Previsão de demanda (Fase 4 do DFT): métodos de "
                                "referência, suavização exponencial, Holt-Winters, ARIMA, validação de origem móvel, "
                                "intervalo por bootstrap e vetores de demanda.")
    p.add_argument("arquivo", help="CSV com a série mensal (1.ª coluna = mês; coluna numérica = demanda)")
    p.add_argument("--coluna", help="nome da coluna da demanda (padrão: última coluna numérica)")
    p.add_argument("--horizonte", type=int, default=12, help="meses a projetar (padrão 12)")
    p.add_argument("--periodo", type=int, default=12, help="período sazonal m (padrão 12; 0 = sem sazonalidade)")
    p.add_argument("--validacao", choices=["origem", "treino-teste"], default="origem",
                   help="origem móvel (padrão) ou divisão única treino/teste")
    p.add_argument("--n-teste", type=int, default=None, help="tamanho do teste na divisão única (padrão: ~20%%, ≥ 3)")
    p.add_argument("--h-validacao", type=int, default=1, help="passos à frente avaliados em cada origem (padrão 1)")
    p.add_argument("--metodos", help="lista separada por vírgula para restringir os candidatos (ex.: ses,holt)")
    p.add_argument("--forcar", help="usa este método na projeção, qualquer que seja o vencedor")
    p.add_argument("--nivel", type=float, default=80, help="cobertura do intervalo em %% (padrão 80)")
    p.add_argument("--semente", type=int, default=SEMENTE_PREVISAO, help="semente do bootstrap (padrão 42)")
    p.add_argument("--vetores", help="CSV de vetores de demanda (D4) a aplicar sobre a projeção")
    p.add_argument("--entrega", help="identificador da entrega, para filtrar os vetores que a alcançam")
    p.add_argument("--grafico", help="salva o gráfico antes/depois neste arquivo PNG")
    p.add_argument("--saida", help="salva a projeção (base, intervalo, fator, ajustada) neste CSV")
    a = p.parse_args(argv)
    m = a.periodo if a.periodo and a.periodo > 1 else None
    try:
        y, col, rot = _ler_serie(a.arquivo, a.coluna)
        rec = recomendar_metodos(y, m)
        print("=" * 78)
        print(f"PREVISÃO DE DEMANDA — Fase 4 do DFT   ·   série '{col}' ({y.size} meses)   ·   arquivo {a.arquivo}")
        print("=" * 78)
        dem = rec["demanda"]
        print(f"Diagnóstico: n = {y.size}; mínimo {br(y.min(), 1)}; máximo {br(y.max(), 1)}; média {br(y.mean(), 1)}; "
              f"padrão de demanda: {dem['classe']}" + (f" (ADI {br(dem['adi'])}; CV² {br(dem['cv2'])})"
                                                       if dem['adi'] else ""))
        if rec["forca_sazonal"] is not None:
            print(f"STL: força de tendência F_T = {br(rec['forca_tendencia'])}; força sazonal F_S = "
                  f"{br(rec['forca_sazonal'])} (limiar 0,64)")
        print(f"Árvore de decisão — ramo {rec['ramo']}:")
        for j in rec["justificativa"]:
            print(f"  • {j}")
        cands = [c.strip() for c in a.metodos.split(",")] if a.metodos else rec["metodos"]
        if not rec["modelar"] and not a.metodos:
            ref = metodos_referencia(y, a.horizonte, m)
            print("\nReferências (apenas informativas, NÃO usar no quadro):")
            print(f"  média = {br(ref['media'][0])}/mês · ingênuo = {br(ref['ingenuo'][0])}/mês")
            return 0
        print(f"\nCandidatos: {', '.join(catalogo(m)[c].nome for c in cands)}")
        if a.validacao == "origem":
            val = validacao_origem_movel(y, cands, m, h_validacao=a.h_validacao)
            met = val["metricas"]
            print(f"Validação de origem móvel (sem embaralhar): {len(val['origens'])} origens, da {val['inicio']}.ª "
                  f"à {y.size - 1}.ª observação; {a.h_validacao} passo(s) à frente.")
        else:
            nt = a.n_teste or max(3, round(0.2 * y.size))
            met = avaliar_treino_teste(y, nt, cands, m)
            print(f"Validação treino/teste: treino = {y.size - nt} primeiros meses; teste = {nt} últimos.")
        escolhido = escolher_metodo(met)
        print(tabela_metricas(met, escolhido))
        for k, v in met.items():
            for al in v.get("alertas", []):
                if "MAPE" in al and v.get("mape") is None and k == escolhido:
                    print(f"  ! {al}")
        chave = a.forcar or escolhido
        print(f"\nMétodo escolhido: {catalogo(m)[escolhido].nome} (menor RMSE; empate → mais simples)."
              + (f"  Projeção FORÇADA com: {catalogo(m)[chave].nome}." if a.forcar else ""))
        proj = projetar(y, chave, a.horizonte, m, a.nivel, semente=a.semente)
        det = proj["detalhes"]
        if "modelo" in det and isinstance(det["modelo"], ModeloSuavizacao):
            mo = det["modelo"]
            print(f"Parâmetros (série completa): α = {br(mo.alpha)}; β = {br(mo.beta)}; φ = {br(mo.phi)}; "
                  f"SSE = {br(mo.sse, 4)}; ℓ_T = {br(mo.lT, 4)}; b_T = {br(mo.bT, 4)}; σ̂ resíduos = "
                  f"{br(det['desvio_residuos'], 4)}")
        if det.get("descricao"):
            print(f"Modelo: {det['descricao']}")
        for al in det["alertas"]:
            print(f"  ! {al}")
        ajuste = None
        vetores: list[Vetor] = []
        if a.vetores:
            vetores = ler_vetores_csv(a.vetores)
            ajuste = aplicar_vetores(proj["previsao"], vetores, a.entrega, proj["inferior"], proj["superior"])
        print(f"\nProjeção de {a.horizonte} meses (intervalo de {br(a.nivel, 0)}%: {det['tipo_intervalo']}):")
        cab = f"  {'h':>3}{'base':>12}{'inferior':>12}{'superior':>12}"
        if ajuste:
            cab += f"{'fator':>8}{'ajustada':>12}"
        print(cab)
        for i in range(a.horizonte):
            lin = (f"  {i + 1:>3}{br(proj['previsao'][i]):>12}"
                   f"{_fmt(None if proj['inferior'] is None else proj['inferior'][i]):>12}"
                   f"{_fmt(None if proj['superior'] is None else proj['superior'][i]):>12}")
            if ajuste:
                lin += f"{br(ajuste['fatores'][i], 3):>8}{br(ajuste['ajustada'][i]):>12}"
            print(lin)
        print(f"  Σ base = {br(proj['previsao'].sum())}")
        if ajuste:
            print(f"  Σ ajustada = {br(ajuste['total_ajustada'])}   Δ = {br(ajuste['delta_absoluto'])} "
                  f"({br(ajuste['delta_percentual'])} %)")
            print("\nTrilha de auditoria dos vetores (D4):")
            for t in ajuste["trilha"]:
                print(f"  • {t['nome']} — fator {br(t['fator'], 3)}; {t['vigencia']}; fonte: {t['fonte']}; "
                      f"responsável: {t['responsavel'] or '—'} ({t['data_registro'] or 's/d'})")
            for al in ajuste["alertas"]:
                print(f"  ! {al}")
        if a.saida:
            import pandas as pd

            pd.DataFrame({"h": np.arange(1, a.horizonte + 1), "base": proj["previsao"],
                          "inferior": proj["inferior"], "superior": proj["superior"],
                          "fator": ajuste["fatores"] if ajuste else 1.0,
                          "ajustada": ajuste["ajustada"] if ajuste else proj["previsao"]}).to_csv(
                a.saida, sep=";", decimal=",", index=False, encoding="utf-8-sig")
            print(f"\nProjeção salva em {a.saida}")
        if a.grafico:
            grafico(y, proj, a.grafico, vetores, ajuste, f"Previsão de demanda — {col}", rot)
            print(f"Gráfico salvo em {a.grafico}")
        print("\nLembrete: o número que vai para o quadro é a projeção ajustada SE houver vetor documentado; sempre "
              "com a base ao lado (FPP3 §6.7).")
        return 0
    except (ErroPrevisao, ErroVetor, FileNotFoundError) as exc:
        print(f"Erro: {exc}")
        return 2
