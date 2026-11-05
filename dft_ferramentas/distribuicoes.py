"""2.1 Identificação da curva de distribuição de tempo das atividades (Fase 6 do DFT — estimativa de esforço).

Dois caminhos, conforme exista ou não registro de tempo:

(a) **Com dados** (cronoanálise, apontamento de horas, extração de sistema): procedimento usual de modelagem de
    entrada de simulação (LAW, 2015, cap. 6; NIST/SEMATECH, 2012, §1.3): estatística descritiva → hipóteses de
    famílias → estimação por máxima verossimilhança → histograma com densidades, gráfico Q-Q/P-P → testes de
    aderência (Kolmogorov-Smirnov, Anderson-Darling, qui-quadrado equiprovável) → escolha por AIC/BIC
    (AKAIKE, 1974; SCHWARZ, 1978; BURNHAM; ANDERSON, 2002). Inclui dados censurados à direita (processo ainda
    aberto na data de corte) e arredondados (tempo anotado em múltiplos de 15 min), além do tamanho de amostra
    para estimar a média com erro relativo desejado (cronoanálise, OIT; BARNES) e da amostragem do trabalho.

(b) **Sem dados** (elicitação de três pontos — mínimo, mais provável, máximo): beta-PERT pelo método de Vose
    (padrão do curso, decisão D3), beta-PERT clássica (MALCOLM et al., 1959), triangular (só sensibilidade) e
    log-normal deslocada (opção avançada do curso; parametrização própria, L2 §3.6). O "mais provável" é a
    **moda** (decisão D1). Reproduz os casos canônicos CT-1 (P80 = 9,074998 h) e CT-2 (P80 = 10,920644 h).

Todas as funções são puras (sem I/O); a leitura de CSV e a impressão ficam em :func:`cli`.
Unidade de tempo: a mesma dos dados (no curso, horas).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import optimize, stats

from ._comum import PALETA, SERIES, br, estilo_graficos, ler_csv, rng

__all__ = [
    "ErroAjuste", "estatisticas_descritivas", "tamanho_amostra_media", "ciclos_cronoanalise_oit",
    "tempo_padrao", "amostragem_trabalho_n", "amostragem_trabalho_proporcoes", "tempo_unitario_amostragem",
    "Ajuste", "CANDIDATAS_PADRAO", "CANDIDATAS_LIMITADAS", "ajustar_candidata", "ajustar_candidatas", "recomendar",
    "estatistica_anderson_darling", "intervalos_de_arredondamento", "ajustar_censurado", "ajustar_intervalar",
    "BetaPert", "LognormalDeslocada", "Triangular", "beta_pert", "lognormal_deslocada_tres_pontos",
    "triangular", "pert_classico", "resumo_tres_pontos", "moda_implicita_quantil", "erro_padrao_quantil",
    "grafico_ajuste", "grafico_tres_pontos", "cli",
]


class ErroAjuste(ValueError):
    """Entrada que não admite o ajuste pedido (mensagem em português, pronta para o usuário)."""


# =====================================================================================================================
# (a.1) Estatística descritiva, tamanho de amostra, cronoanálise e amostragem do trabalho
# =====================================================================================================================

def _z(confianca: float) -> float:
    """Quantil bilateral da normal padrão para o nível de confiança (0,95 → 1,959964)."""
    return float(stats.norm.ppf(0.5 + confianca / 2))


def estatisticas_descritivas(x: Sequence[float]) -> dict[str, float]:
    """Resumo usado na Atividade I de Law (2015, tab. 6.5): média, mediana, desvio, CV, assimetria, curtose e P80.

    Entradas: ``x`` — tempos observados (> 0). Saída: dicionário com ``n, media, mediana, desvio, cv, minimo,
    maximo, assimetria, curtose_excesso, p80_empirico``. O desvio é o amostral (``ddof = 1``). O P80 empírico é o
    quantil 0,80 pelo método de Hazen (interpolação linear entre estatísticas de ordem) — é a leitura direta da
    regra do artigo-base quando há registro: "o valor necessário a cobrir 80% dos menores tempos observados".
    Leitura: média > mediana e assimetria > 0 indicam cauda à direita (típico de tempo de tarefa); CV ≈ 1 sugere
    exponencial, CV < 1 gama/Weibull/log-normal com forma > 1 (LAW, 2015, cap. 6).
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        raise ErroAjuste("São necessários pelo menos dois tempos válidos.")
    media = float(a.mean())
    desvio = float(a.std(ddof=1))
    return {
        "n": int(a.size), "media": media, "mediana": float(np.median(a)), "desvio": desvio,
        "cv": desvio / media if media else float("nan"), "minimo": float(a.min()), "maximo": float(a.max()),
        "assimetria": float(stats.skew(a, bias=False)) if a.size > 2 else float("nan"),
        "curtose_excesso": float(stats.kurtosis(a, bias=False)) if a.size > 3 else float("nan"),
        "p80_empirico": float(np.quantile(a, 0.80, method="hazen")),
    }


def tamanho_amostra_media(cv: float, erro_relativo: float = 0.05, confianca: float = 0.95,
                          n_piloto: int | None = None) -> int:
    """Número de observações para estimar a **média** com erro relativo ``±erro_relativo`` (confiança dada).

    Fórmula usual de estudo de tempos e de Monte Carlo: ``n = (z · CV / E)²`` (BARNES, 1980; OIT, 1992; é também a
    eq. 6 de ICHSAN et al., 2025, com ``z = 3``). Com ``n_piloto``, usa o quantil t de Student com ``n_piloto − 1``
    graus de liberdade e itera ``n ← (t_{n−1} · CV / E)²`` até estabilizar — correção para amostra piloto pequena.
    Saída: inteiro arredondado para cima. Ex.: CV = 0,5; E = 5%; 95% → 385 observações.
    """
    if cv <= 0 or erro_relativo <= 0 or not 0 < confianca < 1:
        raise ErroAjuste("CV e erro relativo devem ser positivos; confiança entre 0 e 1.")
    z = _z(confianca)
    n = (z * cv / erro_relativo) ** 2
    if n_piloto is not None:
        gl = max(int(n_piloto) - 1, 1)
        for _ in range(100):
            t = float(stats.t.ppf(0.5 + confianca / 2, gl))
            novo = (t * cv / erro_relativo) ** 2
            if abs(novo - n) < 1e-9:
                break
            n = novo
            gl = max(math.ceil(n) - 1, 1)
    return int(math.ceil(n - 1e-12))


def ciclos_cronoanalise_oit(tempos_piloto: Sequence[float], erro_relativo: float = 0.05,
                            k: float = 2.0) -> int:
    """Número de ciclos a cronometrar pela fórmula clássica de estudo de tempos (OIT, 1992; BARNES, 1980).

    ``N' = [ (k/E) · √(n·Σx² − (Σx)²) / Σx ]²``. Com ``k = 2`` (≈ 95%) e ``E = 5%`` o fator é 40 — a forma
    tabelada nos manuais. É algebricamente ``(k · CV_pop / E)²``, com o desvio populacional da amostra piloto.
    Se ``N'`` ≤ número de ciclos já cronometrados, a amostra piloto basta.
    """
    x = np.asarray(tempos_piloto, dtype=float)
    n, s1, s2 = x.size, x.sum(), (x ** 2).sum()
    if n < 2 or s1 <= 0:
        raise ErroAjuste("A amostra piloto precisa de pelo menos dois tempos positivos.")
    nprime = ((k / erro_relativo) * math.sqrt(max(n * s2 - s1 ** 2, 0.0)) / s1) ** 2
    return int(math.ceil(nprime - 1e-12))


def tempo_padrao(tempos_observados: Sequence[float], fator_ritmo: float = 1.0,
                 tolerancias: float = 0.0) -> dict[str, float]:
    """Tempo-padrão da cronoanálise: ``TN = média × ritmo``; ``TP = TN × (1 + tolerâncias)`` (OIT, 1992; BARNES).

    ``fator_ritmo`` (avaliação de ritmo/"rating", 1,00 = ritmo normal) e ``tolerancias`` (fração para pausas
    pessoais, fadiga e esperas inevitáveis, ex. 0,15). **Atenção ao DFT:** se a disponibilidade efetiva (Fase 7)
    já desconta pausas e ausências, não as conte de novo aqui (dupla contagem).
    """
    x = np.asarray(tempos_observados, dtype=float)
    to = float(x.mean())
    tn = to * fator_ritmo
    return {"tempo_observado": to, "tempo_normal": tn, "tempo_padrao": tn * (1 + tolerancias)}


def amostragem_trabalho_n(p: float, erro_absoluto: float | None = None, erro_relativo: float | None = None,
                          confianca: float = 0.95) -> int:
    """Número de observações instantâneas na amostragem do trabalho (work sampling).

    Proporção ``p`` da categoria de interesse (estimativa piloto). Erro absoluto ``E`` (pontos de proporção):
    ``n = z²·p(1−p)/E²``; erro relativo ``e`` (fração de ``p``): ``n = z²·(1−p)/(e²·p)`` (BARNES, 1980; OIT, 1992,
    que tabela a forma ``n = 4p(100 − p)/L²`` para 95% com ``z ≈ 2``).
    """
    if not 0 < p < 1:
        raise ErroAjuste("A proporção piloto deve estar entre 0 e 1 (exclusive).")
    z = _z(confianca)
    if erro_absoluto is not None:
        n = z * z * p * (1 - p) / erro_absoluto ** 2
    elif erro_relativo is not None:
        n = z * z * (1 - p) / (erro_relativo ** 2 * p)
    else:
        raise ErroAjuste("Informe erro_absoluto ou erro_relativo.")
    return int(math.ceil(n - 1e-12))


def amostragem_trabalho_proporcoes(categorias: Iterable[str], confianca: float = 0.95) -> list[dict]:
    """Proporção de cada categoria observada, com intervalo de Wilson (melhor que o de Wald em ``p`` pequeno).

    Saída: lista (ordenada por proporção decrescente) de ``{categoria, contagem, p, li, ls}``.
    """
    cats = [str(c) for c in categorias]
    n = len(cats)
    if n == 0:
        raise ErroAjuste("Nenhuma observação de amostragem do trabalho.")
    z = _z(confianca)
    saida = []
    for c in sorted(set(cats)):
        k = cats.count(c)
        p = k / n
        den = 1 + z * z / n
        centro = (p + z * z / (2 * n)) / den
        meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        saida.append({"categoria": c, "contagem": k, "p": p, "li": max(centro - meia, 0.0),
                      "ls": min(centro + meia, 1.0)})
    return sorted(saida, key=lambda d: -d["p"])


def tempo_unitario_amostragem(p: float, horas_periodo: float, quantidade_produzida: float,
                              fator_ritmo: float = 1.0, tolerancias: float = 0.0) -> float:
    """Tempo-padrão unitário pela amostragem do trabalho: ``(p × horas × ritmo / quantidade) × (1 + tol.)``.

    ``p`` = fração das observações na atividade; ``horas_periodo`` = horas-pessoa trabalhadas no período
    observado; ``quantidade_produzida`` = entregas concluídas no período (BARNES, 1980; OIT, 1992).
    """
    if quantidade_produzida <= 0:
        raise ErroAjuste("A quantidade produzida deve ser positiva.")
    return p * horas_periodo * fator_ritmo / quantidade_produzida * (1 + tolerancias)


# =====================================================================================================================
# (a.2) Ajuste de distribuições candidatas por máxima verossimilhança e testes de aderência
# =====================================================================================================================

CANDIDATAS_PADRAO: tuple[str, ...] = ("exponencial", "gama", "lognormal", "weibull", "lognormal_deslocada", "normal")
"""Famílias do ranking padrão com dados (suporte ilimitado à direita, estimação regular ou perfilada)."""

CANDIDATAS_LIMITADAS: tuple[str, ...] = ("triangular", "beta")
"""Famílias de suporte limitado: disponíveis sob pedido (``--candidatas``), fora do ranking padrão. O MLE dos
limites é não regular — os limites "colam" nos extremos da amostra e a teoria do AIC não vale (CHENG; AMIN, 1983) —
e o máximo estimado trunca justamente a cauda que define o P80. São as curvas da elicitação **sem** dados."""

ROTULOS = {
    "exponencial": "Exponencial", "gama": "Gama", "lognormal": "Log-normal", "weibull": "Weibull",
    "lognormal_deslocada": "Log-normal deslocada (3p)", "triangular": "Triangular", "beta": "Beta (4p)",
    "normal": "Normal (referência)",
}


@dataclass
class Ajuste:
    """Resultado do ajuste de uma família a uma amostra."""

    nome: str
    rotulo: str
    dist: object                      # distribuição "congelada" do SciPy
    parametros: dict[str, float]
    k: int                            # nº de parâmetros estimados
    n: int
    loglik: float
    aic: float = float("nan")
    bic: float = float("nan")
    ks: float = float("nan")
    ks_p_nominal: float = float("nan")
    ks_p: float | None = None         # p-valor por bootstrap paramétrico (parâmetros estimados)
    ad: float = float("nan")
    ad_p: float | None = None
    qui2: float = float("nan")
    qui2_gl: int = 0
    qui2_p: float = float("nan")
    r_qq: float = float("nan")        # correlação do gráfico de probabilidade (PPCC, NIST §1.3.3.22)
    p80: float = float("nan")
    avisos: list[str] = field(default_factory=list)

    def quantil(self, q: float) -> float:
        return float(self.dist.ppf(q))


def estatistica_anderson_darling(x: Sequence[float], cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    """``A² = −n − (1/n)·Σ(2i − 1)[ln F(x₍ᵢ₎) + ln(1 − F(x₍ₙ₊₁₋ᵢ₎))]`` (ANDERSON; DARLING, 1954; NIST §1.3.5.14).

    Dá mais peso às caudas que o KS — relevante porque o P80/P90 está na cauda direita.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    f = np.clip(cdf(xs), 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(f) + np.log1p(-f[::-1]))) / n)


def _ll_lognormal_perfil(x: np.ndarray, lam: float) -> float:
    y = np.log(x - lam)
    s = y.std()
    if s <= 0:
        return -np.inf
    return float(-y.sum() - x.size * (math.log(s) + 0.5 * math.log(2 * math.pi) + 0.5))


def _ajustar_lognormal_deslocada(x: np.ndarray) -> tuple[float, float, float, bool]:
    """MLE da log-normal de 3 parâmetros por verossimilhança perfilada em λ ∈ [0, x₍₁₎).

    Para λ fixo, μ e σ têm forma fechada (média e desvio de ln(x − λ)). A verossimilhança é ilimitada quando
    λ → x₍₁₎ (degenerescência conhecida do modelo de 3 parâmetros), por isso a busca para a 1% da distância entre
    o mínimo e a média. Devolve (λ, μ, σ, no_limite).
    """
    xmin, xbar = float(x.min()), float(x.mean())
    topo = xmin - 0.01 * max(xbar - xmin, 1e-9)
    if topo <= 0:
        y = np.log(x)
        return 0.0, float(y.mean()), float(y.std()), False
    grade = np.linspace(0.0, topo, 201)
    lls = np.array([_ll_lognormal_perfil(x, g) for g in grade])
    j = int(np.nanargmax(lls))
    lo, hi = grade[max(j - 1, 0)], grade[min(j + 1, grade.size - 1)]
    if hi > lo:
        r = optimize.minimize_scalar(lambda l: -_ll_lognormal_perfil(x, l), bounds=(lo, hi), method="bounded",
                                     options={"xatol": 1e-10})
        lam = float(r.x) if -r.fun >= lls[j] else float(grade[j])
    else:
        lam = float(grade[j])
    y = np.log(x - lam)
    return lam, float(y.mean()), float(y.std()), j == grade.size - 1


def _congelar(nome: str, x: np.ndarray) -> tuple[object, dict[str, float], int, list[str]]:
    """Estimativas de máxima verossimilhança por família (tempos > 0; λ = 0 quando não deslocada)."""
    avisos: list[str] = []
    if nome == "exponencial":
        esc = float(x.mean())
        return stats.expon(scale=esc), {"media": esc}, 1, avisos
    if nome == "gama":
        a, _, esc = stats.gamma.fit(x, floc=0)
        return stats.gamma(a, scale=esc), {"forma": a, "escala": esc}, 2, avisos
    if nome == "lognormal":
        y = np.log(x)
        mu, s = float(y.mean()), float(y.std())
        return stats.lognorm(s, scale=math.exp(mu)), {"mu": mu, "sigma": s}, 2, avisos
    if nome == "weibull":
        c, _, esc = stats.weibull_min.fit(x, floc=0)
        return stats.weibull_min(c, scale=esc), {"forma": c, "escala": esc}, 2, avisos
    if nome == "lognormal_deslocada":
        lam, mu, s, limite = _ajustar_lognormal_deslocada(x)
        if limite:
            avisos.append("deslocamento λ no limite da busca (verossimilhança degenerada); prefira 2 parâmetros")
        return stats.lognorm(s, loc=lam, scale=math.exp(mu)), {"lambda": lam, "mu": mu, "sigma": s}, 3, avisos
    if nome == "normal":
        mu, dp = float(x.mean()), float(x.std())
        if mu - 2 * dp < 0:
            avisos.append("atribui probabilidade a tempos negativos")
        return stats.norm(mu, dp), {"media": mu, "desvio": dp}, 2, avisos
    if nome in ("triangular", "beta"):
        amp = float(x.max() - x.min())
        folga = amp / max(x.size, 2)
        a0, b0 = max(float(x.min()) - folga, 0.0), float(x.max()) + folga
        if nome == "triangular":
            moda0 = float(stats.mode(np.round(x, 6), keepdims=False).mode)
            c0 = min(max((moda0 - a0) / (b0 - a0), 0.05), 0.95)

            def nll(t):
                a_, b_, c_ = t[0], t[1], t[2]
                if not (0 <= a_ < x.min() and b_ > x.max() and 0 < c_ < 1):
                    return 1e300
                return -float(np.sum(stats.triang.logpdf(x, c_, loc=a_, scale=b_ - a_)))
            r = optimize.minimize(nll, [a0, b0, c0], method="Nelder-Mead",
                                  options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
            a_, b_, c_ = (float(v) for v in r.x)
            avisos.append("suporte limitado: o máximo estimado trunca a cauda; AIC não regular (use só para comparar)")
            return (stats.triang(c_, loc=a_, scale=b_ - a_),
                    {"minimo": a_, "moda": a_ + c_ * (b_ - a_), "maximo": b_}, 3, avisos)
        z = (x - a0) / (b0 - a0)
        m, v = float(z.mean()), float(z.var())
        comum = max(m * (1 - m) / v - 1, 0.5)
        p0, q0 = max(m * comum, 1.5), max((1 - m) * comum, 1.5)

        def nll(t):  # α = 1 + e^t₀, β = 1 + e^t₁ > 1: unimodal, densidade nula nos limites (como a beta-PERT)
            p_, q_, a_, b_ = 1 + math.exp(t[0]), 1 + math.exp(t[1]), t[2], t[3]
            if not (0 <= a_ < x.min() and b_ > x.max()):
                return 1e300
            return -float(np.sum(stats.beta.logpdf(x, p_, q_, loc=a_, scale=b_ - a_)))
        r = optimize.minimize(nll, [math.log(p0 - 1), math.log(q0 - 1), a0, b0], method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 6000})
        p_, q_, a_, b_ = 1 + math.exp(r.x[0]), 1 + math.exp(r.x[1]), float(r.x[2]), float(r.x[3])
        avisos.append("suporte limitado: o máximo estimado trunca a cauda; AIC não regular (use só para comparar)")
        return (stats.beta(p_, q_, loc=a_, scale=b_ - a_),
                {"alfa": p_, "beta": q_, "minimo": a_, "maximo": b_}, 4, avisos)
    raise ErroAjuste(f"Candidata desconhecida: {nome}. Use uma de {', '.join(CANDIDATAS_PADRAO)}.")


def _qui2_equiprovavel(x: np.ndarray, dist, k_param: int) -> tuple[float, int, float]:
    """Qui-quadrado com classes equiprováveis sob a distribuição ajustada (LAW, 2015): ``n·pⱼ ≥ 5``.

    Graus de liberdade ``k − 1 − nº de parâmetros`` (NIST §1.3.5.15); se ≤ 0, devolve p = NaN.
    """
    n = x.size
    k = int(min(20, max(3, n // 5)))
    bordas = dist.ppf(np.linspace(0, 1, k + 1)[1:-1])
    obs = np.bincount(np.searchsorted(bordas, x, side="right"), minlength=k)
    esp = n / k
    q2 = float(((obs - esp) ** 2 / esp).sum())
    gl = k - 1 - k_param
    return q2, gl, float(stats.chi2.sf(q2, gl)) if gl > 0 else float("nan")


def _r_qq(x: np.ndarray, dist) -> float:
    xs = np.sort(x)
    pp = (np.arange(1, xs.size + 1) - 0.5) / xs.size
    teo = dist.ppf(pp)
    if not np.all(np.isfinite(teo)) or np.std(teo) == 0:
        return float("nan")
    return float(np.corrcoef(xs, teo)[0, 1])


def ajustar_candidata(x: Sequence[float], nome: str) -> Ajuste:
    """Ajusta uma família por máxima verossimilhança e calcula AIC, BIC, KS, AD, qui-quadrado, r(Q-Q) e P80.

    ``AIC = 2k − 2 ln L`` (AKAIKE, 1974); ``BIC = k ln n − 2 ln L`` (SCHWARZ, 1978). O p-valor KS "nominal" supõe
    parâmetros conhecidos e é **otimista** quando os parâmetros vêm dos próprios dados (LILLIEFORS, 1967;
    NIST §1.3.5.16); para p-valores válidos use ``ajustar_candidatas(..., bootstrap=B)``.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 5:
        raise ErroAjuste("São necessários pelo menos 5 tempos para ajustar distribuições (idealmente ≥ 30).")
    if nome != "normal" and np.any(a <= 0):
        raise ErroAjuste("Há tempos ≤ 0; trate-os (erro de registro?) antes de ajustar famílias positivas.")
    dist, par, k, avisos = _congelar(nome, a)
    ll = float(np.sum(dist.logpdf(a)))
    aj = Ajuste(nome=nome, rotulo=ROTULOS.get(nome, nome), dist=dist, parametros=par, k=k, n=a.size,
                loglik=ll, avisos=avisos)
    if not np.isfinite(ll):
        aj.avisos.append("verossimilhança não finita (observação fora do suporte)")
        return aj
    if any("degenerada" in av for av in avisos):
        aj.avisos.append("excluída do ranking")
        return aj
    aj.aic = 2 * k - 2 * ll
    aj.bic = k * math.log(a.size) - 2 * ll
    ks = stats.kstest(a, dist.cdf)
    aj.ks, aj.ks_p_nominal = float(ks.statistic), float(ks.pvalue)
    aj.ad = estatistica_anderson_darling(a, dist.cdf)
    aj.qui2, aj.qui2_gl, aj.qui2_p = _qui2_equiprovavel(a, dist, k)
    aj.r_qq = _r_qq(a, dist)
    aj.p80 = float(dist.ppf(0.80))
    return aj


def _bootstrap(aj: Ajuste, B: int, semente: int | None) -> None:
    """p-valores de KS e AD por bootstrap paramétrico: simula da ajustada, reajusta, recalcula (STEPHENS, 1974)."""
    g = rng(semente)
    ks_b, ad_b = [], []
    for _ in range(B):
        xb = aj.dist.rvs(size=aj.n, random_state=g)
        if aj.nome != "normal":
            xb = np.clip(xb, 1e-9, None)
        try:
            d, _, _, _ = _congelar(aj.nome, xb)
        except Exception:  # pragma: no cover - réplica patológica
            continue
        ks_b.append(stats.kstest(xb, d.cdf).statistic)
        ad_b.append(estatistica_anderson_darling(xb, d.cdf))
    if ks_b:
        aj.ks_p = (1 + sum(v >= aj.ks for v in ks_b)) / (len(ks_b) + 1)
        aj.ad_p = (1 + sum(v >= aj.ad for v in ad_b)) / (len(ad_b) + 1)


def ajustar_candidatas(x: Sequence[float], candidatas: Sequence[str] = CANDIDATAS_PADRAO, bootstrap: int = 0,
                       semente: int | None = 20261105) -> list[Ajuste]:
    """Ajusta todas as candidatas e devolve a lista ordenada por AIC (menor primeiro).

    ``bootstrap`` = nº de réplicas para p-valores de KS e AD com parâmetros estimados (99–999; 0 = desligado).
    Candidatas que falham são omitidas silenciosamente apenas se a falha for de suporte; as demais propagam erro.
    """
    ajustes = []
    for nome in candidatas:
        try:
            aj = ajustar_candidata(x, nome)
        except (FloatingPointError, RuntimeError, np.linalg.LinAlgError):
            continue
        if bootstrap and np.isfinite(aj.aic):
            _bootstrap(aj, int(bootstrap), semente)
        ajustes.append(aj)
    return sorted(ajustes, key=lambda a: (not np.isfinite(a.aic), a.aic))


def recomendar(ajustes: Sequence[Ajuste], alfa: float = 0.05, delta_aic: float = 2.0) -> dict:
    """Escolhe a distribuição e redige a justificativa (critério explícito, auditável).

    Regra: (1) ordena por AIC; (2) "plausíveis" = ΔAIC ≤ 2 (BURNHAM; ANDERSON, 2002); (3) descarta plausíveis
    rejeitadas pelo AD (p bootstrap < α; sem bootstrap, usa o KS nominal apenas como alerta); (4) entre as que
    restam, prefere a de **menos parâmetros** (parcimônia) e, empatando, a de menor AIC. A normal nunca é
    recomendada para tempo de tarefa se houver alternativa (LAW, 2016, "armadilha 2"), nem uma família de suporte
    limitado (triangular, beta) se houver alternativa regular.
    Saída: ``{escolhida: Ajuste, plausiveis: [...], justificativa: str, faixa_p80: (min, max)}``.
    """
    todos = [a for a in ajustes if np.isfinite(a.aic)]
    if not todos:
        raise ErroAjuste("Nenhuma candidata pôde ser ajustada.")
    regulares = [a for a in todos if a.nome != "normal" and a.nome not in CANDIDATAS_LIMITADAS]
    validos = regulares or todos
    melhor = min(a.aic for a in validos)
    plaus = [a for a in validos if a.aic - melhor <= delta_aic]
    rejeitada = (lambda a: a.ad_p is not None and a.ad_p < alfa)
    aceitas = [a for a in plaus if not rejeitada(a)] or plaus
    nao_normal = [a for a in aceitas if a.nome != "normal" and a.nome not in CANDIDATAS_LIMITADAS] or aceitas
    escolhida = sorted(nao_normal, key=lambda a: (a.k, a.aic))[0]
    ordem = sorted(validos, key=lambda a: a.aic)
    seg = next((a for a in ordem if a is not escolhida), None)
    p80s = [a.p80 for a in plaus]
    partes = [
        f"{escolhida.rotulo}: AIC = {br(escolhida.aic, 1)} (ΔAIC = {br(escolhida.aic - melhor, 2)} em relação ao "
        f"menor AIC; BIC = {br(escolhida.bic, 1)})."
    ]
    if seg is not None:
        partes.append(f"Próxima candidata: {seg.rotulo} (ΔAIC = {br(seg.aic - escolhida.aic, 2)}).")
    if len(plaus) > 1:
        partes.append(f"{len(plaus)} candidatas têm ΔAIC ≤ {br(delta_aic, 0)} (suporte equivalente); entre elas "
                      f"prevalece a de menos parâmetros (parcimônia).")
    if escolhida.ad_p is not None:
        verbo = "não rejeita" if escolhida.ad_p >= alfa else "REJEITA"
        partes.append(f"Anderson-Darling {verbo} a aderência (A² = {br(escolhida.ad, 3)}; p bootstrap = "
                      f"{br(escolhida.ad_p, 3)}); KS D = {br(escolhida.ks, 3)} (p bootstrap = {br(escolhida.ks_p, 3)}).")
    else:
        partes.append(f"A² = {br(escolhida.ad, 3)}; KS D = {br(escolhida.ks, 3)} (p nominal = "
                      f"{br(escolhida.ks_p_nominal, 3)}, otimista porque os parâmetros vêm da amostra).")
    partes.append(f"r(Q-Q) = {br(escolhida.r_qq, 4)}. P80 ajustado = {br(escolhida.p80, 2)}; entre as plausíveis o "
                  f"P80 vai de {br(min(p80s), 2)} a {br(max(p80s), 2)}.")
    if escolhida.n < 30:
        partes.append("Amostra pequena (n < 30): os testes têm baixo poder — trate a escolha como provisória.")
    if all(rejeitada(a) for a in plaus):
        partes.append("ATENÇÃO: todas as plausíveis foram rejeitadas pelo AD — verifique mistura de entregas "
                      "diferentes (fronteira mal definida) ou use a distribuição empírica.")
    if escolhida.avisos:
        partes.append("Avisos: " + "; ".join(escolhida.avisos) + ".")
    return {"escolhida": escolhida, "plausiveis": plaus, "justificativa": " ".join(partes),
            "faixa_p80": (min(p80s), max(p80s))}


# ---- dados censurados e arredondados ---------------------------------------------------------------------------------

_FAMILIAS_CENSURA = {
    # nome: (construtor a partir de parâmetros livres em escala log, chute inicial a partir de x)
    "exponencial": (lambda t: stats.expon(scale=math.exp(t[0])), lambda x: [math.log(x.mean())]),
    "gama": (lambda t: stats.gamma(math.exp(t[0]), scale=math.exp(t[1])),
             lambda x: [math.log(max(x.mean() ** 2 / x.var(), 0.1)), math.log(x.var() / x.mean())]),
    "lognormal": (lambda t: stats.lognorm(math.exp(t[0]), scale=math.exp(t[1])),
                  lambda x: [math.log(max(np.log(x).std(), 0.05)), float(np.log(x).mean())]),
    "weibull": (lambda t: stats.weibull_min(math.exp(t[0]), scale=math.exp(t[1])),
                lambda x: [math.log(1.5), math.log(x.mean())]),
}


def _parametros_legiveis(nome: str, dist) -> dict[str, float]:
    a, kw = dist.args, dist.kwds
    if nome == "exponencial":
        return {"media": kw["scale"]}
    if nome == "lognormal":
        return {"mu": math.log(kw["scale"]), "sigma": a[0]}
    return {"forma": a[0], "escala": kw["scale"]}


def ajustar_censurado(tempos: Sequence[float], observado: Sequence[bool], familia: str = "lognormal") -> Ajuste:
    """MLE com **censura à direita**: processos ainda abertos na data de corte (tempo real ≥ tempo decorrido).

    ``ln L = Σ_obs ln f(tᵢ) + Σ_cens ln S(tᵢ)``, com ``S = 1 − F`` (NIST §8.1.3.1, censura tipo I; LAW, 2015).
    Descartar os abertos — ou tratá-los como concluídos — **subestima** o tempo (some justamente a cauda longa).
    Famílias: exponencial, gama, lognormal, weibull.
    """
    t = np.asarray(tempos, dtype=float)
    obs = np.asarray(observado, dtype=bool)
    if t.size != obs.size or t.size < 5 or np.any(t <= 0):
        raise ErroAjuste("Tempos e indicadores devem ter o mesmo tamanho (≥ 5) e tempos > 0.")
    if obs.sum() < 2:
        raise ErroAjuste("Há menos de dois tempos concluídos; não é possível estimar.")
    if familia not in _FAMILIAS_CENSURA:
        raise ErroAjuste(f"Família não suportada com censura: {familia}.")
    constr, chute = _FAMILIAS_CENSURA[familia]

    def nll(th):
        d = constr(th)
        v = d.logpdf(t[obs]).sum() + d.logsf(t[~obs]).sum()
        return -v if np.isfinite(v) else 1e300

    r = optimize.minimize(nll, chute(t[obs]), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12,
                                                                               "maxiter": 5000})
    d = constr(r.x)
    k = len(r.x)
    aj = Ajuste(nome=familia, rotulo=ROTULOS[familia] + " (censura)", dist=d, parametros=_parametros_legiveis(familia, d),
                k=k, n=t.size, loglik=-float(r.fun))
    aj.aic = 2 * k + 2 * r.fun
    aj.bic = k * math.log(t.size) + 2 * r.fun
    aj.p80 = float(d.ppf(0.8))
    return aj


def intervalos_de_arredondamento(x: Sequence[float], passo: float,
                                 piso: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Converte tempos anotados em múltiplos de ``passo`` (ex.: 0,25 h) em intervalos ``[x − passo/2, x + passo/2]``.

    ``piso`` = menor valor anotável (ex.: quem anota em blocos de 15 min nunca escreve 0): registros ``≤ piso``
    viram ``(0, piso + passo/2]``. O limite inferior é truncado em ``passo/1000`` para manter o suporte positivo.
    """
    a = np.asarray(x, dtype=float)
    lo = np.maximum(a - passo / 2, passo / 1000)
    if piso is not None:
        lo = np.where(a <= piso + 1e-12, passo / 1000, lo)
    return lo, a + passo / 2


def ajustar_intervalar(inferior: Sequence[float], superior: Sequence[float], familia: str = "lognormal") -> Ajuste:
    """MLE para dados **intervalares** (arredondados ou "entre 2 e 4 h"): ``ln L = Σ ln[F(sᵢ) − F(iᵢ)]``.

    Evita que os empates artificiais do arredondamento distorçam o ajuste e os testes (TRIETSCH et al., 2012
    documentam o arredondamento como distorção típica de tempos reais). Famílias: exponencial, gama, lognormal,
    weibull.
    """
    lo = np.asarray(inferior, dtype=float)
    hi = np.asarray(superior, dtype=float)
    if lo.size != hi.size or lo.size < 5 or np.any(hi <= lo) or np.any(lo < 0):
        raise ErroAjuste("Intervalos inválidos: exija 0 ≤ inferior < superior e pelo menos 5 registros.")
    if familia not in _FAMILIAS_CENSURA:
        raise ErroAjuste(f"Família não suportada para dados intervalares: {familia}.")
    constr, chute = _FAMILIAS_CENSURA[familia]
    meio = (lo + hi) / 2

    def nll(th):
        d = constr(th)
        p = d.cdf(hi) - d.cdf(lo)
        if np.any(p <= 0):
            return 1e300
        return -float(np.log(p).sum())

    r = optimize.minimize(nll, chute(meio), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12,
                                                                            "maxiter": 5000})
    d = constr(r.x)
    k = len(r.x)
    aj = Ajuste(nome=familia, rotulo=ROTULOS[familia] + " (intervalar)", dist=d,
                parametros=_parametros_legiveis(familia, d), k=k, n=lo.size, loglik=-float(r.fun))
    aj.aic = 2 * k + 2 * r.fun
    aj.bic = k * math.log(lo.size) + 2 * r.fun
    aj.p80 = float(d.ppf(0.8))
    return aj


# =====================================================================================================================
# (b) Sem dados: três pontos (mínimo, mais provável = MODA, máximo)
# =====================================================================================================================

def _validar_tres_pontos(a: float, m: float, b: float) -> None:
    """Mesmas regras do motor TypeScript do Sistema DFT (``validarTresPontos``)."""
    if not all(math.isfinite(v) for v in (a, m, b)):
        raise ErroAjuste("Os três pontos precisam ser números finitos.")
    if a <= 0:
        raise ErroAjuste("O mínimo precisa ser maior que zero.")
    if not a < b:
        raise ErroAjuste("O mínimo precisa ser estritamente menor que o máximo.")
    if not a <= m <= b:
        raise ErroAjuste("O valor mais provável precisa estar entre o mínimo e o máximo "
                         "(mínimo ≤ mais provável ≤ máximo).")


class _Base:
    dist: object

    def quantil(self, q: float) -> float:
        return float(self.dist.ppf(q))

    def cdf(self, x: float) -> float:
        return float(self.dist.cdf(x))

    def pdf(self, x: float) -> float:
        return float(self.dist.pdf(x))

    def media(self) -> float:
        return float(self.dist.mean())

    def desvio(self) -> float:
        return float(self.dist.std())


@dataclass(frozen=True)
class BetaPert(_Base):
    """beta-PERT reescalada para ``[a, b]``: ``X = a + (b − a)·Beta(α, β)``."""

    a: float
    m: float
    b: float
    alfa: float
    beta: float
    media_pert: float
    dp_pert: float
    metodo: str

    @property
    def dist(self):
        return stats.beta(self.alfa, self.beta, loc=self.a, scale=self.b - self.a)

    @property
    def rotulo(self) -> str:
        return "beta-PERT (Vose)" if self.metodo == "vose" else "beta-PERT (clássica)"


@dataclass(frozen=True)
class LognormalDeslocada(_Base):
    """``X = λ + exp(Y)``, ``Y ~ Normal(μ, σ²)``; mínimo e máximo lidos como quantis ``p`` e ``1 − p``."""

    lam: float
    mu: float
    sigma: float
    p: float

    @property
    def dist(self):
        return stats.lognorm(self.sigma, loc=self.lam, scale=math.exp(self.mu))

    rotulo = "log-normal deslocada"


@dataclass(frozen=True)
class Triangular(_Base):
    a: float
    m: float
    b: float

    @property
    def dist(self):
        return stats.triang((self.m - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)

    rotulo = "triangular (sensibilidade)"


def beta_pert(a: float, m: float, b: float, metodo: str = "vose", k: float = 4.0) -> BetaPert:
    """beta-PERT a partir de três pontos (L2 §3.5; documentação ``prevalence::betaPERT``).

    ``μ = (a + k·m + b)/(k + 2)``; ``sd = (b − a)/(k + 2)``.
    - Vose (padrão, D3 — preserva a moda ``m``): ``α = (μ − a)(2m − a − b) / [(m − μ)(b − a)]``;
      se ``m = μ``, ``α = 1 + k/2``.
    - Clássico (MALCOLM et al., 1959 — preserva ``sd``): ``α = [(μ − a)/(b − a)]·[(μ − a)(b − μ)/sd² − 1]``.
    - Em ambos, ``β = α(b − μ)/(μ − a)``.
    CT-1 (4; 6; 16): α = 1,666667; β = 4,333333; P80 = 9,074998 h.
    """
    _validar_tres_pontos(a, m, b)
    if metodo not in ("vose", "classico", "classic"):
        raise ErroAjuste("Método da beta-PERT deve ser 'vose' ou 'classico'.")
    mu = (a + k * m + b) / (k + 2)
    sd = (b - a) / (k + 2)
    if metodo in ("classico", "classic"):
        alfa = ((mu - a) / (b - a)) * (((mu - a) * (b - mu)) / (sd * sd) - 1)
        metodo = "classico"
    elif abs(m - mu) < 1e-12:
        alfa = 1 + k / 2
    else:
        alfa = ((mu - a) * (2 * m - a - b)) / ((m - mu) * (b - a))
    beta = alfa * (b - mu) / (mu - a)
    if not (alfa > 0 and beta > 0 and math.isfinite(alfa) and math.isfinite(beta)):
        raise ErroAjuste("Os três pontos não produzem uma beta-PERT válida (α ou β não positivo). "
                         "Use a triangular apenas para sensibilidade ou reveja a coleta.")
    return BetaPert(a, m, b, alfa, beta, mu, sd, metodo)


def lognormal_deslocada_tres_pontos(a: float, m: float, b: float, p: float = 0.05) -> LognormalDeslocada:
    """Log-normal deslocada a partir de três pontos — parametrização do curso (L2 §3.6, **derivação própria**).

    Condições: ``λ + e^{μ − zσ} = a`` (mínimo = quantil p), ``λ + e^{μ + zσ} = b`` (máximo = quantil 1 − p) e
    ``λ + e^{μ − σ²} = m`` (mais provável = moda), com ``z = Φ⁻¹(1 − p)``. Daí ``μ(λ) = ½ ln[(a − λ)(b − λ)]``,
    ``σ(λ) = ln[(b − λ)/(a − λ)]/(2z)`` e resolve-se ``g(λ) = λ + e^{μ(λ) − σ(λ)²} − m = 0`` em ``[0, a)``.
    Se ``g(0) ≤ 0`` (estimativa simétrica ou assimétrica à esquerda), recusa e recomenda a beta-PERT.
    CT-2 (4; 6; 16; p = 5%): λ = 1,545912; μ = 1,784366; σ = 0,539021; P80 = 10,920644 h.
    """
    _validar_tres_pontos(a, m, b)
    if not 0 < p < 0.5:
        raise ErroAjuste("A probabilidade de cauda p deve estar entre 0 e 0,5.")
    z = float(stats.norm.ppf(1 - p))

    def par(lam: float) -> tuple[float, float]:
        return 0.5 * math.log((a - lam) * (b - lam)), math.log((b - lam) / (a - lam)) / (2 * z)

    def g(lam: float) -> float:
        mu, s = par(lam)
        return lam + math.exp(mu - s * s) - m

    if g(0.0) <= 0:
        raise ErroAjuste("Os três pontos não admitem log-normal deslocada com λ ≥ 0 (a estimativa não é "
                         "assimétrica à direita o bastante). Use a beta-PERT para esta entrega.")
    lam = optimize.brentq(g, 0.0, a - 1e-9, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500)
    mu, s = par(lam)
    return LognormalDeslocada(float(lam), mu, s, p)


def triangular(a: float, m: float, b: float) -> Triangular:
    """Triangular ``Tri(a, m, b)`` — hipótese mais crua; no curso, **só para sensibilidade** (D3)."""
    _validar_tres_pontos(a, m, b)
    return Triangular(a, m, b)


def pert_classico(a: float, m: float, b: float) -> dict[str, float]:
    """Fórmulas originais do PERT (MALCOLM et al., 1959): ``média = (a + 4m + b)/6``, ``dp = (b − a)/6``.

    Inclui o "P80 normal" ``média + 0,8416·dp`` usado em cronogramas (aproximação normal do caminho crítico);
    é só referência — para uma entrega isolada e assimétrica, use o quantil da beta-PERT.
    """
    _validar_tres_pontos(a, m, b)
    media, dp = (a + 4 * m + b) / 6, (b - a) / 6
    return {"media": media, "desvio": dp, "p80_normal": media + float(stats.norm.ppf(0.8)) * dp}


def resumo_tres_pontos(a: float, m: float, b: float, percentis: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
                       p_cauda: float = 0.05, incluir_triangular: bool = False) -> dict:
    """Tabela comparativa exigida pela decisão D3: beta-PERT (Vose) e log-normal deslocada lado a lado.

    Saída: ``{"pert": {...}, "linhas": [{rotulo, parametros, media, desvio, P50, P80, ...}], "delta_p80": %,
    "avisos": [...]}``; ``delta_p80`` = (P80 log-normal / P80 beta-PERT − 1). Inclui a beta-PERT clássica como
    comparação e, se pedido, a triangular (sensibilidade).
    """
    dists: list[tuple[str, _Base, dict]] = []
    avisos = []
    bp = beta_pert(a, m, b, "vose")
    dists.append((bp.rotulo, bp, {"alfa": bp.alfa, "beta": bp.beta}))
    bc = beta_pert(a, m, b, "classico")
    dists.append((bc.rotulo, bc, {"alfa": bc.alfa, "beta": bc.beta}))
    ln = None
    try:
        ln = lognormal_deslocada_tres_pontos(a, m, b, p_cauda)
        dists.append((ln.rotulo, ln, {"lambda": ln.lam, "mu": ln.mu, "sigma": ln.sigma}))
    except ErroAjuste as e:
        avisos.append(str(e))
    if incluir_triangular:
        tr = triangular(a, m, b)
        dists.append((tr.rotulo, tr, {"a": a, "m": m, "b": b}))
    linhas = []
    for rot, d, par in dists:
        lin = {"rotulo": rot, "parametros": par, "media": d.media(), "desvio": d.desvio()}
        for q in percentis:
            lin[f"P{round(q * 100)}"] = d.quantil(q)
        linhas.append(lin)
    delta = (ln.quantil(0.8) / bp.quantil(0.8) - 1) if ln is not None else float("nan")
    return {"pert": pert_classico(a, m, b), "linhas": linhas, "delta_p80": delta, "avisos": avisos,
            "beta_pert": bp, "lognormal": ln}


def moda_implicita_quantil(a: float, valor_central: float, b: float, prob: float = 2 / 3) -> dict[str, float]:
    """Ilustra a decisão **D1** (moda × quantil): e se o valor central informado for o quantil 2/3, e não a moda?

    O artigo-base define o "mais provável" como "valor suficiente a concluir 2/3 das demandas" (um quantil), mas
    as fórmulas PERT o tratam como moda. Esta função acha a moda ``m*`` tal que a beta-PERT (Vose) ``(a, m*, b)``
    tenha ``F(valor_central) = prob`` e compara o P80 das duas leituras.
    Saída: ``{moda_implicita, p80_lido_como_moda, p80_lido_como_quantil, variacao_p80}``, com
    ``variacao_p80 = P80(quantil)/P80(moda) − 1``. Em tempos assimétricos à direita a moda fica abaixo do quantil
    2/3, logo ``m* < valor_central`` e a variação é negativa: **ler como moda um valor pensado como quantil
    superestima** o P80; **ler como quantil um valor típico (moda) subestima** — por isso a coleta pergunta pelo
    valor típico e o sistema o trata como moda (D1). Se nenhuma moda em ``[a, b]`` produz o quantil pedido (o valor
    está abaixo do quantil 2/3 mesmo com moda = mínimo), levanta :class:`ErroAjuste`.
    """
    _validar_tres_pontos(a, valor_central, b)

    def h(mm: float) -> float:
        return beta_pert(a, mm, b).cdf(valor_central) - prob

    lo, hi = a + 1e-9 * (b - a), b - 1e-9 * (b - a)
    if h(lo) * h(hi) > 0:
        raise ErroAjuste("Nenhuma moda entre o mínimo e o máximo torna esse valor o quantil pedido: com esses "
                         "extremos, o valor informado é baixo demais para ser o quantil 2/3 — trate-o como moda.")
    mm = optimize.brentq(h, lo, hi, xtol=1e-12)
    p_moda = beta_pert(a, valor_central, b).quantil(0.8)
    p_quant = beta_pert(a, mm, b).quantil(0.8)
    return {"moda_implicita": mm, "p80_lido_como_moda": p_moda, "p80_lido_como_quantil": p_quant,
            "variacao_p80": p_quant / p_moda - 1}


def erro_padrao_quantil(dist: _Base | Ajuste, q: float = 0.8, n: int = 10_000) -> float:
    """Erro-padrão de Monte Carlo do quantil ``q`` com ``n`` sorteios: ``√(q(1 − q)/n) / f(Q(q))`` (L2 §3.7).

    CT-1: f(9,075) = 0,107836 → SE ≈ 0,0371 h com N = 10.000.
    """
    d = dist.dist
    return math.sqrt(q * (1 - q) / n) / float(d.pdf(d.ppf(q)))


# =====================================================================================================================
# Gráficos (opcionais)
# =====================================================================================================================

def grafico_ajuste(x: Sequence[float], ajustes: Sequence[Ajuste], caminho: str, n_curvas: int = 3,
                   titulo: str = "Ajuste de distribuições aos tempos observados") -> str:
    """Três painéis: histograma + densidades (top ``n_curvas`` por AIC), Q-Q da melhor e P-P (FDA × empírica)."""
    plt = estilo_graficos()
    a = np.sort(np.asarray(x, dtype=float))
    top = [aj for aj in ajustes if np.isfinite(aj.aic)][:n_curvas]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].hist(a, bins="fd", density=True, color=PALETA["claro"], edgecolor=PALETA["linha"], label="observado")
    grade = np.linspace(max(a.min() - 0.1 * np.ptp(a), 0), a.max() + 0.2 * np.ptp(a), 400)
    for cor, aj in zip(SERIES, top):
        ax[0].plot(grade, aj.dist.pdf(grade), color=cor, lw=2, label=f"{aj.rotulo} (AIC {br(aj.aic, 1)})")
    p80e = float(np.quantile(a, 0.8, method="hazen"))
    ax[0].axvline(p80e, color=PALETA["verde"], ls="--", lw=1.5, label=f"P80 empírico = {br(p80e)}")
    ax[0].set(title="Histograma e densidades ajustadas", xlabel="tempo", ylabel="densidade")
    ax[0].legend(fontsize=8, frameon=False)
    pp = (np.arange(1, a.size + 1) - 0.5) / a.size
    for cor, aj in zip(SERIES, top):
        ax[1].plot(aj.dist.ppf(pp), a, "o", ms=3.5, color=cor, alpha=0.8, label=aj.rotulo)
        ax[2].plot(pp, aj.dist.cdf(a), "o", ms=3, color=cor, alpha=0.8, label=aj.rotulo)
    lim = [min(a.min(), top[0].dist.ppf(pp[0])), max(a.max(), top[0].dist.ppf(pp[-1]))]
    ax[1].plot(lim, lim, color=PALETA["cinza"], lw=1)
    ax[1].set(title="Gráfico Q-Q (quantis)", xlabel="quantil teórico", ylabel="quantil observado")
    ax[2].plot([0, 1], [0, 1], color=PALETA["cinza"], lw=1)
    ax[2].set(title="Gráfico P-P (probabilidades)", xlabel="probabilidade empírica", ylabel="FDA ajustada")
    ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle(titulo, color=PALETA["marinho"])
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_tres_pontos(a: float, m: float, b: float, caminho: str, p_cauda: float = 0.05,
                        titulo: str | None = None) -> str:
    """Densidades da beta-PERT (Vose) e da log-normal deslocada com o P80 de cada uma (decisão D3)."""
    plt = estilo_graficos()
    r = resumo_tres_pontos(a, m, b, p_cauda=p_cauda, incluir_triangular=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.linspace(0, b * 1.6, 600)
    cores = {"beta-PERT (Vose)": PALETA["azul"], "log-normal deslocada": PALETA["marinho"],
             "triangular (sensibilidade)": PALETA["linha"]}
    objs = [r["beta_pert"], r["lognormal"], triangular(a, m, b)]
    for d in objs:
        if d is None:
            continue
        cor = cores[d.rotulo]
        ax.plot(xs, d.dist.pdf(xs), color=cor, lw=2.2 if "triang" not in d.rotulo else 1.2,
                ls="-" if "triang" not in d.rotulo else ":", label=f"{d.rotulo}: P80 = {br(d.quantil(0.8))}")
        if "triang" not in d.rotulo:
            ax.axvline(d.quantil(0.8), color=cor, ls="--", lw=1)
    for v, nome in ((a, "mín."), (m, "mais provável"), (b, "máx.")):
        ax.axvline(v, color=PALETA["cinza"], lw=0.6, alpha=0.6)
        ax.text(v, ax.get_ylim()[1] * 0.97, nome, rotation=90, va="top", ha="right", fontsize=8,
                color=PALETA["cinza"])
    ax.set(xlabel="horas por entrega", ylabel="densidade",
           title=titulo or f"Três pontos ({br(a)}; {br(m)}; {br(b)}) — mesma entrada, duas hipóteses")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# =====================================================================================================================
# Interface de linha de comando
# =====================================================================================================================

def _num(s: str) -> float:
    return float(str(s).replace(",", "."))


def _relatorio_tres_pontos(a: float, m: float, b: float, p_cauda: float, demanda: float | None,
                           rotulo: str = "") -> list[str]:
    r = resumo_tres_pontos(a, m, b, p_cauda=p_cauda, incluir_triangular=True)
    L = [f"Três pontos{(' — ' + rotulo) if rotulo else ''}: mínimo = {br(a)}; mais provável (moda) = {br(m)}; "
         f"máximo = {br(b)}",
         f"  PERT clássico (Malcolm et al., 1959): média = {br(r['pert']['media'], 6)}; "
         f"desvio ≈ (b − a)/6 = {br(r['pert']['desvio'], 6)}"]
    L.append(f"  {'distribuição':<28}{'parâmetros':<48}{'média':>10}{'P50':>11}{'P80':>12}{'P90':>11}{'P95':>11}")
    for lin in r["linhas"]:
        par = ", ".join(f"{k}={br(v, 6)}" for k, v in lin["parametros"].items())
        L.append(f"  {lin['rotulo']:<28}{par:<48}{br(lin['media'], 4):>10}{br(lin['P50'], 4):>11}"
                 f"{br(lin['P80'], 6):>12}{br(lin['P90'], 4):>11}{br(lin['P95'], 4):>11}")
    for av in r["avisos"]:
        L.append(f"  Aviso: {av}")
    if r["lognormal"] is not None:
        bp, ln = r["beta_pert"], r["lognormal"]
        L.append(f"  Δ P80 (log-normal deslocada vs. beta-PERT) = {br(100 * r['delta_p80'], 1)}% "
                 f"— a escolha da distribuição é decisão de governança (D3): registre e justifique.")
        if demanda:
            dh = (ln.quantil(0.8) - bp.quantil(0.8)) * demanda
            L.append(f"  Com demanda de {br(demanda)} entregas/ano: diferença de {br(dh, 1)} h/ano.")
        L.append("  Pergunta-filtro (D3): existe caso, ainda que raro, que demore mais que o máximo informado? "
                 "Se sim → log-normal deslocada; se não → beta-PERT (Vose).")
    try:
        d1 = moda_implicita_quantil(a, m, b)
        L.append(f"  Leitura D1: se {br(m)} fosse o 'valor que cobre 2/3' (quantil), a moda implícita seria "
                 f"{br(d1['moda_implicita'], 3)} e o P80 {br(d1['p80_lido_como_quantil'], 3)} "
                 f"({br(100 * d1['variacao_p80'], 1)}%). Pergunte pelo valor típico e trate-o como moda.")
    except ErroAjuste:
        L.append(f"  Leitura D1: {br(m)} fica abaixo do quantil 2/3 mesmo com a moda no mínimo — só pode ser lido "
                 f"como moda (valor típico).")
    L.append(f"  Erro de Monte Carlo do P80 da beta-PERT com N = 10.000: ±{br(erro_padrao_quantil(r['beta_pert']), 4)}")
    return L


def _relatorio_amostra(x: np.ndarray, args, so_descritiva: bool = False) -> list[str]:
    d = estatisticas_descritivas(x)
    L = ["ESTATÍSTICA DESCRITIVA (Law, 2015, Atividade I)",
         f"  n = {d['n']}; média = {br(d['media'], 3)}; mediana = {br(d['mediana'], 3)}; desvio = "
         f"{br(d['desvio'], 3)}; CV = {br(d['cv'], 3)}",
         f"  mínimo = {br(d['minimo'], 3)}; máximo = {br(d['maximo'], 3)}; assimetria = {br(d['assimetria'], 3)}; "
         f"curtose (excesso) = {br(d['curtose_excesso'], 3)}; P80 empírico = {br(d['p80_empirico'], 3)}"]
    if d["media"] > d["mediana"] and d["assimetria"] > 0:
        L.append("  Leitura: cauda à direita (média > mediana) — candidatas naturais: gama, log-normal, Weibull.")
    n_nec = tamanho_amostra_media(d["cv"], args.erro_relativo, args.confianca, n_piloto=d["n"])
    n_oit = ciclos_cronoanalise_oit(x, args.erro_relativo, k=2.0)
    situacao = "a amostra BASTA" if d["n"] >= n_nec else f"faltam {n_nec - d['n']} registros"
    L += ["", f"TAMANHO DE AMOSTRA para a média com erro de ±{br(100 * args.erro_relativo, 1)}% e confiança "
              f"{br(100 * args.confianca, 0)}%",
          f"  n = (t·CV/E)² = {n_nec} observações (fórmula OIT/Barnes com k = 2: {n_oit}); {situacao}."]
    if so_descritiva:
        return L
    cand = args.candidatas.split(",") if args.candidatas else list(CANDIDATAS_PADRAO)
    ajustes = ajustar_candidatas(x, cand, bootstrap=args.bootstrap, semente=args.semente)
    rec = recomendar(ajustes, alfa=args.alfa)
    tit_p = "p(AD)boot" if args.bootstrap else "p(KS)nom."
    L += ["", "AJUSTE POR MÁXIMA VEROSSIMILHANÇA — ranking por AIC",
          f"  {'candidata':<28}{'k':>2}{'AIC':>10}{'ΔAIC':>8}{'BIC':>10}{'KS D':>8}{'A²':>8}{tit_p:>11}"
          f"{'qui² p':>8}{'r(QQ)':>8}{'P80':>9}"]
    melhor = ajustes[0].aic
    for aj in ajustes:
        if not np.isfinite(aj.aic):
            continue
        pv = aj.ad_p if args.bootstrap else aj.ks_p_nominal
        L.append(f"  {aj.rotulo:<28}{aj.k:>2}{br(aj.aic, 1):>10}{br(aj.aic - melhor, 2):>8}{br(aj.bic, 1):>10}"
                 f"{br(aj.ks, 3):>8}{br(aj.ad, 3):>8}{br(pv, 3) if pv is not None else '—':>11}"
                 f"{br(aj.qui2_p, 3) if np.isfinite(aj.qui2_p) else '—':>8}{br(aj.r_qq, 4):>8}{br(aj.p80, 3):>9}")
    excl = [aj for aj in ajustes if not np.isfinite(aj.aic)]
    for aj in excl:
        L.append(f"  {aj.rotulo}: excluída — {'; '.join(aj.avisos)}")
    if not args.bootstrap:
        L.append("  (p-valores KS nominais são otimistas com parâmetros estimados; use --bootstrap 199 para AD/KS válidos)")
    esc = rec["escolhida"]
    L += ["", "RECOMENDAÇÃO", "  " + rec["justificativa"],
          "  Parâmetros: " + ", ".join(f"{k} = {br(v, 4)}" for k, v in esc.parametros.items()),
          f"  Para o DFT (Fase 6): esforço unitário = P80 = {br(esc.p80, 3)} (ajustado) | {br(d['p80_empirico'], 3)} "
          f"(empírico, regra literal do artigo-base). Registre a escolha e a justificativa."]
    if args.grafico:
        grafico_ajuste(x, ajustes, args.grafico)
        L.append(f"  Gráfico salvo em {args.grafico}")
    return L


def cli(argv: list[str]) -> int:
    """``dft distribuicoes [arquivo.csv] [opções]`` — imprime relatório em português; ``--grafico saida.png``.

    Modos: (1) CSV de tempos (coluna ``--coluna``, padrão ``tempo_h``; ``--censura encerrado`` para censura à
    direita; ``--arredondamento 0.25`` para dados arredondados); (2) CSV de três pontos (colunas ``minimo;
    mais_provavel;maximo`` e opcional ``entrega``); (3) ``--tres-pontos 4 6 16`` sem arquivo;
    (4) ``--amostragem-trabalho categoria`` para proporções de work sampling.
    """
    ap = argparse.ArgumentParser(prog="dft distribuicoes",
                                 description="2.1 Curva de distribuição de tempo das atividades (Fase 6 do DFT).")
    ap.add_argument("arquivo", nargs="?", help="CSV (; e vírgula decimal) com tempos ou com três pontos")
    ap.add_argument("--coluna", default="tempo_h", help="coluna dos tempos (padrão: tempo_h)")
    ap.add_argument("--filtro", help="valor da coluna 'entrega' a analisar (quando o CSV mistura entregas)")
    ap.add_argument("--censura", help="coluna 0/1 (1 = concluído; 0 = ainda aberto na data de corte)")
    ap.add_argument("--arredondamento", type=_num, help="passo de arredondamento do registro (ex.: 0,25 h)")
    ap.add_argument("--candidatas", help="lista separada por vírgula (padrão: todas)")
    ap.add_argument("--bootstrap", type=int, default=0, help="réplicas para p-valores de KS/AD (ex.: 199)")
    ap.add_argument("--semente", type=int, default=20261105)
    ap.add_argument("--alfa", type=_num, default=0.05)
    ap.add_argument("--erro-relativo", dest="erro_relativo", type=_num, default=0.05)
    ap.add_argument("--confianca", type=_num, default=0.95)
    ap.add_argument("--tres-pontos", dest="tres_pontos", nargs=3, type=_num, metavar=("MIN", "MODA", "MAX"))
    ap.add_argument("--p-cauda", dest="p_cauda", type=_num, default=0.05,
                    help="probabilidade abaixo do mínimo/acima do máximo na log-normal deslocada (padrão 0,05)")
    ap.add_argument("--demanda", type=_num, help="entregas/ano, para traduzir a diferença de P80 em horas")
    ap.add_argument("--amostragem-trabalho", dest="amostragem", metavar="COLUNA",
                    help="coluna de categoria das observações instantâneas (work sampling)")
    ap.add_argument("--grafico", help="caminho do PNG de saída")
    args = ap.parse_args(argv)
    L = ["=" * 100, "DFT · 2.1 Identificação da curva de distribuição de tempo das atividades", "=" * 100]
    try:
        if args.tres_pontos:
            a, m, b = args.tres_pontos
            L += _relatorio_tres_pontos(a, m, b, args.p_cauda, args.demanda)
            if args.grafico:
                grafico_tres_pontos(a, m, b, args.grafico, args.p_cauda)
                L.append(f"  Gráfico salvo em {args.grafico}")
        elif not args.arquivo:
            ap.print_usage()
            print("Informe um arquivo CSV ou --tres-pontos MIN MODA MAX.")
            return 2
        else:
            df = ler_csv(args.arquivo)
            if args.filtro and "entrega" in df.columns:
                df = df[df["entrega"].astype(str) == args.filtro]
            cols = set(df.columns)
            if args.amostragem:
                props = amostragem_trabalho_proporcoes(df[args.amostragem].astype(str))
                n = len(df)
                L += [f"AMOSTRAGEM DO TRABALHO — {n} observações instantâneas (IC de Wilson, "
                      f"{br(100 * args.confianca, 0)}%)",
                      f"  {'categoria':<30}{'obs.':>6}{'p':>9}{'IC':>22}{'n p/ ±10% rel.':>16}"]
                for d in props:
                    nn = amostragem_trabalho_n(d["p"], erro_relativo=0.10, confianca=args.confianca) \
                        if 0 < d["p"] < 1 else 0
                    L.append(f"  {d['categoria']:<30}{d['contagem']:>6}{br(100 * d['p'], 1) + '%':>9}"
                             f"{'[' + br(100 * d['li'], 1) + '%; ' + br(100 * d['ls'], 1) + '%]':>22}{nn:>16}")
                L.append(f"  n para ±5 p.p. com p = 50% (pior caso): "
                         f"{amostragem_trabalho_n(0.5, erro_absoluto=0.05, confianca=args.confianca)}")
            elif {"minimo", "mais_provavel", "maximo"} <= cols:
                for _, lin in df.iterrows():
                    rot = str(lin["entrega"]) if "entrega" in cols else ""
                    try:
                        L += _relatorio_tres_pontos(float(lin["minimo"]), float(lin["mais_provavel"]),
                                                    float(lin["maximo"]), args.p_cauda, args.demanda, rot)
                    except ErroAjuste as e:
                        L.append(f"Três pontos — {rot}: ERRO: {e}")
                    L.append("")
                if args.grafico and len(df):
                    lin = df.iloc[0]
                    grafico_tres_pontos(float(lin["minimo"]), float(lin["mais_provavel"]), float(lin["maximo"]),
                                        args.grafico, args.p_cauda)
                    L.append(f"Gráfico (primeira entrega) salvo em {args.grafico}")
            else:
                if args.coluna not in cols:
                    raise ErroAjuste(f"Coluna '{args.coluna}' não encontrada. Colunas: {', '.join(map(str, cols))}.")
                x = df[args.coluna].astype(float).to_numpy()
                if args.censura:
                    obs = df[args.censura].astype(int).to_numpy() == 1
                    L += [f"DADOS CENSURADOS À DIREITA: {int((~obs).sum())} de {obs.size} registros ainda abertos",
                          f"  {'família':<30}{'AIC':>10}{'P80 (com censura)':>20}{'P80 (só concluídos)':>22}"]
                    res = []
                    for fam in ("exponencial", "gama", "lognormal", "weibull"):
                        aj = ajustar_censurado(x, obs, fam)
                        ing = ajustar_candidata(x[obs], fam)
                        res.append(aj)
                        L.append(f"  {aj.rotulo:<30}{br(aj.aic, 1):>10}{br(aj.p80, 3):>20}{br(ing.p80, 3):>22}")
                    melhor = min(res, key=lambda r: r.aic)
                    L.append(f"  Menor AIC: {melhor.rotulo}; P80 = {br(melhor.p80, 3)}. Ignorar os abertos "
                             f"subestima o tempo.")
                elif args.arredondamento:
                    lo, hi = intervalos_de_arredondamento(x, args.arredondamento, piso=args.arredondamento)
                    L += [f"DADOS ARREDONDADOS (passo {br(args.arredondamento)}): MLE intervalar",
                          f"  {'família':<30}{'AIC':>10}{'P80':>10}"]
                    res = sorted((ajustar_intervalar(lo, hi, f) for f in ("exponencial", "gama", "lognormal",
                                                                                         "weibull")), key=lambda r: r.aic)
                    for aj in res:
                        L.append(f"  {aj.rotulo:<30}{br(aj.aic, 1):>10}{br(aj.p80, 3):>10}")
                    esc = res[0]
                    L += [f"  Recomendação: {esc.rotulo} (menor AIC; ΔAIC para a 2ª = {br(res[1].aic - esc.aic, 2)}); "
                          f"P80 = {br(esc.p80, 3)}. Parâmetros: "
                          + ", ".join(f"{k} = {br(v, 4)}" for k, v in esc.parametros.items()) + ".",
                          "  Os empates criados pelo arredondamento invalidam KS/AD pontuais e podem degenerar a "
                          "log-normal de 3 parâmetros; por isso o ajuste é intervalar.", ""]
                    L += _relatorio_amostra(x, args, so_descritiva=True)
                    if args.grafico:
                        grafico_ajuste(x, res, args.grafico, titulo="Ajuste intervalar (dados arredondados)")
                        L.append(f"Gráfico salvo em {args.grafico}")
                else:
                    L += _relatorio_amostra(x, args)
    except ErroAjuste as e:
        L.append(f"ERRO: {e}")
        print("\n".join(L))
        return 1
    print("\n".join(L))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(cli(sys.argv[1:]))
