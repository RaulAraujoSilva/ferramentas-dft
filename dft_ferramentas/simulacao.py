"""5. Simulação de processos por Monte Carlo — esforço anual de um setor e fila de atendimento.

Duas ferramentas, ambas com semente fixa (reprodutíveis em auditoria):

(a) **Monte Carlo do esforço anual** (Fase 6 do DFT — RODRIGUES et al., 2021): cada entrega tem volume anual (fixo ou
    três pontos) e tempo por execução (três pontos → beta-PERT de Vose, log-normal deslocada ou triangular). Sorteiam-se
    N = 10.000 cenários (METROPOLIS; ULAM, 1949; RUBINSTEIN; KROESE, 2017), com correlação de postos opcional entre as
    entregas (cópula gaussiana ou Iman-Conover — IMAN; CONOVER, 1982), e lê-se P50/P80/P90 da carga total em horas e em
    servidores (carga ÷ disponibilidade efetiva). Reporta o erro de Monte Carlo do P80 (intervalo de confiança do
    quantil por estatísticas de ordem — DAVID; NAGARAJA, 2003), a margem "soma dos P80 − P80 da soma" e o tornado de
    sensibilidade (correlação de postos de Spearman e balanço P10–P90).

(b) **Simulação de eventos discretos** de um processo de atendimento (LAW, 2015; BANKS et al., 2010): chegadas de
    Poisson, etapas em série com c servidores e tempos estocásticos, retrabalho com retorno a uma etapa anterior,
    aquecimento e replicações independentes. Motor próprio (lista de eventos com heapq) ou SimPy, se instalado. Reporta
    tempo de atravessamento, fila média, utilização e nível de serviço para 1, 2, 3… servidores.

Unidades: horas (tempo por execução, carga anual, disponibilidade efetiva em h/servidor/ano). No simulador de fila, o
relógio corre em **horas úteis** contínuas (a noite e o fim de semana não existem no modelo).

As distribuições a partir de três pontos vêm de `dft_ferramentas.distribuicoes` (beta_pert, lognormal_deslocada_tres_pontos,
triangular) quando o módulo está presente; há funções privadas equivalentes aqui como reserva (parametrização do dossiê
L2, §3.5–§3.6; casos de teste CT-1 e CT-2).
"""
from __future__ import annotations

import argparse
import heapq
import math
import textwrap
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from ._comum import PALETA, br, estilo_graficos, formatar_eixos_br, ler_csv, rng

N_PADRAO = 10_000
DISPONIBILIDADE_CT3 = 1153.60      # h/servidor/ano — CT-3, passo 6 (L2, §6)
DISPONIBILIDADE_NOVE_PASSOS = 928.2  # h/servidor/ano — exemplo de nove passos (L1, §3.6)
DISTRIBUICOES = ("beta_pert", "lognormal_deslocada", "triangular")


# =====================================================================================================================
# 1. Distribuições a partir de três pontos (privadas; equivalentes às de `distribuicoes`)
# =====================================================================================================================

def _validar_tres_pontos(a: float, m: float, b: float) -> None:
    if not (a <= m <= b):
        raise ValueError(f"três pontos inválidos: exige mínimo ≤ mais provável ≤ máximo (recebido {a}; {m}; {b})")
    if a < 0:
        raise ValueError("tempo/volume não pode ser negativo")


def _beta_pert_parametros(a: float, m: float, b: float, metodo: str = "vose", k: float = 4.0) -> tuple[float, float]:
    """α e β da beta-PERT em [a, b]. Vose (2008) preserva a moda; o clássico (MALCOLM et al., 1959) fixa sd=(b−a)/6."""
    _validar_tres_pontos(a, m, b)
    mu = (a + k * m + b) / (k + 2)
    if metodo == "classico":
        sd = (b - a) / (k + 2)
        alfa = ((mu - a) / (b - a)) * (((mu - a) * (b - mu) / sd ** 2) - 1)
    elif abs(m - mu) < 1e-12:
        alfa = 1 + k / 2
    else:
        alfa = ((mu - a) * (2 * m - a - b)) / ((m - mu) * (b - a))
    beta = alfa * (b - mu) / (mu - a)
    return float(alfa), float(beta)


def _lognormal_deslocada_parametros(a: float, m: float, b: float, p: float = 0.05) -> tuple[float, float, float]:
    """(λ, μ, σ) da log-normal deslocada com mínimo = quantil p, máximo = quantil 1−p e mais provável = moda (L2, §3.6)."""
    _validar_tres_pontos(a, m, b)
    z = float(stats.norm.ppf(1 - p))

    def par(lam: float) -> tuple[float, float]:
        return 0.5 * math.log((a - lam) * (b - lam)), math.log((b - lam) / (a - lam)) / (2 * z)

    def g(lam: float) -> float:
        mu, s = par(lam)
        return lam + math.exp(mu - s * s) - m

    if g(0.0) <= 0:
        raise ValueError("três pontos não admitem log-normal deslocada com λ ≥ 0; use beta-PERT (D3)")
    lo, hi = 0.0, a - 1e-12
    for _ in range(200):
        meio = 0.5 * (lo + hi)
        if g(meio) > 0:
            lo = meio
        else:
            hi = meio
    lam = 0.5 * (lo + hi)
    mu, s = par(lam)
    return lam, mu, s


try:  # reutiliza o módulo `distribuicoes` quando presente; senão, usa as funções privadas equivalentes acima
    from . import distribuicoes as _distribuicoes
    _ = (_distribuicoes.beta_pert, _distribuicoes.lognormal_deslocada_tres_pontos, _distribuicoes.triangular)
except (ImportError, AttributeError):  # pragma: no cover - depende da presença do módulo irmão
    _distribuicoes = None


@lru_cache(maxsize=256)
def _dist_congelada(a: float, m: float, b: float, distribuicao: str):
    """Distribuição scipy congelada a partir dos três pontos (em cache: o ajuste é feito uma vez só)."""
    if distribuicao not in DISTRIBUICOES:
        raise ValueError(f"distribuição desconhecida: {distribuicao} (use {', '.join(DISTRIBUICOES)})")
    if _distribuicoes is not None:
        if distribuicao == "beta_pert":
            return _distribuicoes.beta_pert(a, m, b).dist
        if distribuicao == "lognormal_deslocada":
            return _distribuicoes.lognormal_deslocada_tres_pontos(a, m, b).dist
        return _distribuicoes.triangular(a, m, b).dist
    _validar_tres_pontos(a, m, b)
    if distribuicao == "beta_pert":
        al, be = _beta_pert_parametros(a, m, b)
        return stats.beta(al, be, loc=a, scale=b - a)
    if distribuicao == "lognormal_deslocada":
        lam, mu, s = _lognormal_deslocada_parametros(a, m, b)
        return stats.lognorm(s, loc=lam, scale=math.exp(mu))
    return stats.triang((m - a) / (b - a), loc=a, scale=b - a)


@dataclass(frozen=True)
class TresPontos:
    """Estimativa de três pontos (mínimo, mais provável = moda (D1), máximo) e a distribuição adotada (D3)."""
    minimo: float
    mais_provavel: float
    maximo: float
    distribuicao: str = "beta_pert"

    @property
    def degenerada(self) -> bool:
        return self.maximo - self.minimo < 1e-12

    def _congelada(self):
        return _dist_congelada(self.minimo, self.mais_provavel, self.maximo, self.distribuicao)

    def quantil(self, u):
        """Q(u) — transformada inversa (vetorizada). Para três pontos iguais, devolve o valor fixo."""
        u = np.asarray(u, dtype=float)
        if self.degenerada:
            return np.full_like(u, self.minimo)
        return self._congelada().ppf(u)

    def densidade(self, x):
        if self.degenerada:
            return np.full_like(np.asarray(x, dtype=float), np.inf)
        return self._congelada().pdf(x)

    def media(self) -> float:
        return self.minimo if self.degenerada else float(self._congelada().mean())

    def desvio(self) -> float:
        return 0.0 if self.degenerada else float(self._congelada().std())


def quantil_analitico(tp: TresPontos, p: float) -> float:
    """Quantil exato (sem simulação) — ex.: CT-1 P80 beta-PERT (4; 6; 16) = 9,074998 h; CT-2 log-normal = 10,920644 h."""
    return float(tp.quantil(p))


# =====================================================================================================================
# 2. Erro de Monte Carlo do percentil e número de replicações
# =====================================================================================================================

def erro_padrao_quantil(p: float, n: int, densidade_no_quantil: float) -> float:
    """SE(Q̂p) ≈ √(p(1−p)/n) / f(Qp) — variância assintótica do quantil amostral (DAVID; NAGARAJA, 2003).

    CT-1: p=0,8, n=10.000, f(9,075)=0,107836 → 0,0371 h.
    """
    return math.sqrt(p * (1 - p) / n) / densidade_no_quantil


def ic_quantil_ordem(amostras: Sequence[float], p: float, confianca: float = 0.95) -> tuple[float, float]:
    """IC do quantil p sem supor distribuição: estatísticas de ordem x(j) e x(k) com j, k da Binomial(n, p).

    O número de amostras ≤ Qp é Binomial(n, p); escolhem-se os postos j = B⁻¹(α/2), k = B⁻¹(1−α/2)+1
    (DAVID; NAGARAJA, 2003, §7.1; CONOVER, quantile test). Vale para qualquer distribuição de saída.
    """
    x = np.sort(np.asarray(amostras, dtype=float))
    n = len(x)
    alfa = 1 - confianca
    j = int(stats.binom.ppf(alfa / 2, n, p))
    k = int(stats.binom.ppf(1 - alfa / 2, n, p)) + 1
    j, k = max(j, 1), min(k, n)
    return float(x[j - 1]), float(x[k - 1])


def densidade_empirica(amostras: Sequence[float], x: float) -> float:
    """Estimativa de f(x) por núcleo gaussiano (regra de Scott) — usada no SE do quantil quando não há fórmula."""
    a = np.asarray(amostras, dtype=float)
    if np.ptp(a) < 1e-12:
        return float("inf")
    return float(stats.gaussian_kde(a)(x)[0])


def n_para_precisao_quantil(p: float, densidade_no_quantil: float, meia_largura: float,
                            confianca: float = 0.95) -> int:
    """N necessário para que o IC do quantil p tenha a meia-largura pedida: N = (z/h)² · p(1−p) / f².

    Inverte `erro_padrao_quantil` (LAW, 2015, cap. 9: número de replicações para precisão absoluta).
    """
    z = float(stats.norm.ppf(0.5 + confianca / 2))
    return int(math.ceil((z / meia_largura) ** 2 * p * (1 - p) / densidade_no_quantil ** 2))


def n_iteracoes_ichsan(desvio: float, media: float, erro_relativo: float) -> int:
    """Regra de Ichsan, Isvara e Karim (2025, eqs. 4–6): n = (3σ/ε)², ε = média·erro_relativo (para a MÉDIA)."""
    eps = media * erro_relativo
    return int(math.ceil((3 * desvio / eps) ** 2))


# =====================================================================================================================
# 3. Correlação entre entradas: cópula gaussiana e Iman-Conover
# =====================================================================================================================

def spearman_para_pearson_normal(rho_s):
    """Converte correlação de postos (Spearman) alvo em correlação de Pearson das normais: r = 2·sen(π·ρs/6)."""
    return 2 * np.sin(np.pi * np.asarray(rho_s, dtype=float) / 6)


def matriz_correlacao(k: int, rho: float | np.ndarray) -> np.ndarray:
    """Matriz k×k: se `rho` for número, correlação uniforme entre todos os pares; se matriz, valida e devolve."""
    if np.isscalar(rho):
        c = np.full((k, k), float(rho))
        np.fill_diagonal(c, 1.0)
    else:
        c = np.asarray(rho, dtype=float)
        if c.shape != (k, k) or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1):
            raise ValueError("matriz de correlação deve ser k×k, simétrica e com diagonal 1")
    if np.linalg.eigvalsh(c).min() < -1e-10:
        raise ValueError("matriz de correlação não é positiva semidefinida")
    return c


def _cholesky_psd(c: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError:          # ρ = 1 exato (comonotônico) ou matriz singular
        w, v = np.linalg.eigh(c)
        return v @ np.diag(np.sqrt(np.clip(w, 0, None)))


def uniformes_correlacionadas(n: int, rho_postos: float | np.ndarray, k: int,
                              gerador: np.random.Generator) -> np.ndarray:
    """n×k uniformes com correlação de postos alvo, por cópula gaussiana (NELSEN, 2006): U = Φ(Z·Lᵀ)."""
    c = matriz_correlacao(k, rho_postos)
    r = spearman_para_pearson_normal(c)
    np.fill_diagonal(r, 1.0)
    z = gerador.standard_normal((n, k)) @ _cholesky_psd(r).T
    return np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)


def iman_conover(amostras: np.ndarray, rho_postos: float | np.ndarray, gerador: np.random.Generator) -> np.ndarray:
    """Reordena cada coluna de `amostras` (n×k, independentes) para induzir a correlação de postos alvo.

    Iman e Conover (1982): escores de van der Waerden Φ⁻¹(i/(n+1)) permutados ao acaso; correção pela correlação
    amostral dos escores (S = P·Q⁻¹); cada coluna original assume os postos da coluna correspondente dos escores.
    Preserva exatamente as marginais (os valores sorteados não mudam, só o pareamento).
    """
    x = np.asarray(amostras, dtype=float)
    n, k = x.shape
    alvo = matriz_correlacao(k, rho_postos)
    base = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    escores = np.column_stack([gerador.permutation(base) for _ in range(k)])
    t = np.corrcoef(escores, rowvar=False)
    s = _cholesky_psd(alvo) @ np.linalg.inv(_cholesky_psd(t))
    alvo_escores = escores @ s.T
    saida = np.empty_like(x)
    for j in range(k):
        postos = (stats.rankdata(alvo_escores[:, j], method="ordinal") - 1).astype(np.intp)
        saida[:, j] = np.sort(x[:, j])[postos]
    return saida


# =====================================================================================================================
# 4. Monte Carlo do esforço anual de um setor
# =====================================================================================================================

@dataclass(frozen=True)
class Entrega:
    """Entrega do catálogo: volume anual (três pontos; iguais = fixo) × tempo por execução (três pontos, em horas)."""
    nome: str
    volume: TresPontos
    tempo: TresPontos

    @classmethod
    def simples(cls, nome: str, volume: float, tempo_min: float, tempo_mp: float, tempo_max: float,
                distribuicao: str = "beta_pert") -> "Entrega":
        return cls(nome, TresPontos(volume, volume, volume, "beta_pert"),
                   TresPontos(tempo_min, tempo_mp, tempo_max, distribuicao))


@dataclass
class ResultadoEsforco:
    """Saída do Monte Carlo do esforço. `horas` é n×k (por entrega); `total` é a carga anual do setor (n)."""
    nomes: list[str]
    horas: np.ndarray
    total: np.ndarray
    n: int
    semente: int | None
    correlacao: float | np.ndarray
    metodo_correlacao: str
    agregacao: str
    percentis: dict[int, float] = field(default_factory=dict)
    p80_individuais: dict[str, float] = field(default_factory=dict)

    @property
    def soma_p80_individuais(self) -> float:
        return float(sum(self.p80_individuais.values()))

    @property
    def p80_total(self) -> float:
        return self.percentis[80]


def simular_esforco(entregas: Sequence[Entrega], n: int = N_PADRAO, semente: int | None = 20261105,
                    correlacao: float | np.ndarray = 0.0, metodo_correlacao: str = "copula",
                    agregacao: str = "comum", percentis: Iterable[int] = (10, 50, 80, 90, 95)) -> ResultadoEsforco:
    """Carga anual (h) = Σ volume × tempo por execução, em `n` cenários.

    agregacao:
      * "comum" (padrão do curso): num cenário, todas as execuções da entrega levam o mesmo tempo sorteado — a incerteza
        dos três pontos é tratada como incerteza sobre o tempo típico do setor. Reproduz a regra demanda × esforço.
      * "independente": as V execuções são independentes entre si; a soma é aproximada por Normal(V·μ, V·σ²) (teorema
        central do limite; exata se V < 30, por soma direta). Mostra quanto da variância se compensa ao longo do ano.
    correlacao: correlação de POSTOS (Spearman) entre os tempos das entregas (número = uniforme; ou matriz k×k).
    metodo_correlacao: "copula" (gaussiana, por transformada inversa) ou "iman_conover" (reordenação).
    """
    if n < 100:
        raise ValueError("use n ≥ 100 (padrão 10.000)")
    g = rng(semente)
    k = len(entregas)
    if k == 0:
        raise ValueError("nenhuma entrega informada")
    # volumes: independentes entre si
    vol = np.column_stack([e.volume.quantil(np.clip(g.random(n), 1e-12, 1 - 1e-12)) for e in entregas])
    # tempos: correlação de postos opcional
    tem_corr = (np.any(np.asarray(correlacao) != 0) if not np.isscalar(correlacao) else correlacao != 0) and k > 1
    if metodo_correlacao not in ("copula", "iman_conover"):
        raise ValueError("metodo_correlacao deve ser 'copula' ou 'iman_conover'")
    if tem_corr and metodo_correlacao == "copula":
        u = uniformes_correlacionadas(n, correlacao, k, g)
    else:
        u = np.clip(g.random((n, k)), 1e-12, 1 - 1e-12)
    if agregacao == "comum":
        tempo = np.column_stack([e.tempo.quantil(u[:, j]) for j, e in enumerate(entregas)])
        if tem_corr and metodo_correlacao == "iman_conover":
            tempo = iman_conover(tempo, correlacao, g)
        horas = vol * tempo
    elif agregacao == "independente":
        horas = np.empty((n, k))
        for j, e in enumerate(entregas):
            mu, sd = e.tempo.media(), e.tempo.desvio()
            v = vol[:, j]
            if np.all(v >= 30) or sd == 0:
                z = stats.norm.ppf(u[:, j])
                horas[:, j] = v * mu + np.sqrt(v) * sd * z
            else:
                vi = np.rint(v).astype(int)
                horas[:, j] = [e.tempo.quantil(np.clip(g.random(int(c)), 1e-12, 1 - 1e-12)).sum() for c in vi]
        if tem_corr and metodo_correlacao == "iman_conover":
            horas = iman_conover(horas, correlacao, g)
    else:
        raise ValueError("agregacao deve ser 'comum' ou 'independente'")
    total = horas.sum(axis=1)
    res = ResultadoEsforco([e.nome for e in entregas], horas, total, n, semente, correlacao,
                           metodo_correlacao, agregacao)
    res.percentis = {int(p): float(np.percentile(total, p)) for p in sorted(set(percentis) | {50, 80, 90})}
    res.p80_individuais = {e.nome: float(np.percentile(horas[:, j], 80)) for j, e in enumerate(entregas)}
    return res


def soma_p80_analitica(entregas: Sequence[Entrega], p: float = 0.8) -> float:
    """Regra oficial do quadro (L2, §4.3): Σ volume_mais_provável × P80 analítico do tempo. CT-3: 28.432,5106 h."""
    return float(sum(e.volume.mais_provavel * quantil_analitico(e.tempo, p) for e in entregas))


def servidores(carga_horas: float, disponibilidade: float) -> tuple[float, int]:
    """Quantitativo = carga ÷ disponibilidade efetiva; inteiro sempre para cima (ceil). CT-3: 28.432,51/1.153,60 → 25."""
    if disponibilidade <= 0:
        raise ValueError("disponibilidade efetiva deve ser positiva")
    q = carga_horas / disponibilidade
    return q, int(math.ceil(q - 1e-9))


def erro_mc_p80(res: ResultadoEsforco, confianca: float = 0.95) -> dict[str, float]:
    """Erro de Monte Carlo do P80 da carga total: SE assintótico (densidade por núcleo) e IC por estatísticas de ordem."""
    p80 = res.percentis[80]
    f = densidade_empirica(res.total, p80)
    se = 0.0 if math.isinf(f) else erro_padrao_quantil(0.8, res.n, f)
    lo, hi = ic_quantil_ordem(res.total, 0.8, confianca)
    return {"p80": p80, "erro_padrao": se, "ic_inf": lo, "ic_sup": hi,
            "erro_relativo": se / p80 if p80 else 0.0}


def tornado(res: ResultadoEsforco) -> list[dict[str, float]]:
    """Sensibilidade da carga total a cada entrega, ordenada da maior para a menor influência.

    * rho_postos: correlação de Spearman entre as horas da entrega e o total (o "tornado de correlação" do @Risk/PMI);
    * contribuicao_variancia: ρ² normalizado (aproximação usual de "contribuição à variância");
    * baixo/alto: total quando a entrega vai ao seu P10/P90 e as demais ficam no P50 (balanço "um de cada vez").
    """
    med = np.percentile(res.horas, 50, axis=0)
    base = float(med.sum())
    linhas = []
    for j, nome in enumerate(res.nomes):
        col = res.horas[:, j]
        rho = 0.0 if np.ptp(col) < 1e-12 else float(stats.spearmanr(col, res.total)[0])
        p10, p90 = np.percentile(col, [10, 90])
        linhas.append({"entrega": nome, "rho_postos": rho, "baixo": base - med[j] + p10,
                       "alto": base - med[j] + p90, "balanco": float(p90 - p10)})
    soma_r2 = sum(l["rho_postos"] ** 2 for l in linhas) or 1.0
    for l in linhas:
        l["contribuicao_variancia"] = l["rho_postos"] ** 2 / soma_r2
        l["base"] = base
    return sorted(linhas, key=lambda l: l["balanco"], reverse=True)


def convergencia_p80(amostras: Sequence[float], bloco: int = 1000) -> list[tuple[int, float]]:
    """P80 acumulado a cada `bloco` amostras — diagnóstico visual/numérico de estabilização (L2, §3.7)."""
    a = np.asarray(amostras, dtype=float)
    return [(i, float(np.percentile(a[:i], 80))) for i in range(bloco, len(a) + 1, bloco)]


# =====================================================================================================================
# 5. Simulação de eventos discretos de um processo de atendimento
# =====================================================================================================================

@dataclass(frozen=True)
class Etapa:
    """Etapa do processo: c servidores em paralelo, fila única FIFO, tempo de serviço em horas.

    distribuicao: "beta_pert" | "lognormal_deslocada" | "triangular" (três pontos), "exponencial" (usa o mais provável
    como MÉDIA — para validar contra M/M/c) ou "fixo". Após concluir a etapa, com probabilidade `prob_retrabalho` o
    caso volta para a etapa `volta_para` (nome).
    """
    nome: str
    servidores: int
    tempo: TresPontos
    prob_retrabalho: float = 0.0
    volta_para: str | None = None

    def sortear(self, g: np.random.Generator) -> float:
        d = self.tempo.distribuicao
        if d == "exponencial":
            return float(g.exponential(self.tempo.mais_provavel))
        if d == "fixo" or self.tempo.degenerada:
            return float(self.tempo.mais_provavel)
        return float(self.tempo.quantil(min(max(g.random(), 1e-12), 1 - 1e-12)))

    def media(self) -> float:
        d = self.tempo.distribuicao
        if d in ("exponencial", "fixo") or self.tempo.degenerada:
            return float(self.tempo.mais_provavel)
        return self.tempo.media()


class _Amostrador:
    """Sorteia tempos de serviço de uma etapa em blocos vetorizados, consumidos em ordem (mesma sequência do sortear)."""

    def __init__(self, etapa: "Etapa", gerador: np.random.Generator, bloco: int = 2048):
        self.etapa, self.g, self.bloco = etapa, gerador, bloco
        self.buf = np.empty(0)
        self.i = 0

    def proximo(self) -> float:
        if self.i >= len(self.buf):
            e, d = self.etapa, self.etapa.tempo.distribuicao
            if d == "exponencial":
                self.buf = self.g.exponential(e.tempo.mais_provavel, self.bloco)
            elif d == "fixo" or e.tempo.degenerada:
                self.buf = np.full(self.bloco, float(e.tempo.mais_provavel))
            else:
                self.buf = e.tempo.quantil(np.clip(self.g.random(self.bloco), 1e-12, 1 - 1e-12))
            self.i = 0
        v = float(self.buf[self.i])
        self.i += 1
        return v


class _Fluxos:
    """Fluxos aleatórios separados por finalidade (chegadas; serviço e retrabalho por etapa) — números aleatórios comuns.

    Separar os fluxos faz com que dois motores (próprio e SimPy) e duas configurações (1, 2, 3… servidores) consumam
    os MESMOS números para os mesmos fins — técnica de números aleatórios comuns (LAW, 2015, cap. 11).
    """

    def __init__(self, semente: int | None, etapas: Sequence["Etapa"]):
        n_etapas = len(etapas)
        ss = np.random.SeedSequence(semente)
        filhos = ss.spawn(1 + 2 * n_etapas)
        self.chegada = np.random.default_rng(filhos[0])
        self.servico = [_Amostrador(e, np.random.default_rng(f)) for e, f in zip(etapas, filhos[1:1 + n_etapas])]
        self.retrabalho = [np.random.default_rng(f) for f in filhos[1 + n_etapas:]]


def _chegadas_poisson(taxa: float, horizonte: float, g: np.random.Generator) -> np.ndarray:
    """Instantes de chegada de um processo de Poisson homogêneo (intervalos exponenciais de média 1/taxa)."""
    tempos, t = [], 0.0
    while True:
        t += g.exponential(1.0 / taxa)
        if t >= horizonte:
            return np.array(tempos)
        tempos.append(t)


@dataclass
class _Coletor:
    """Estatísticas por média no tempo (fila, ocupação, casos no sistema) dentro da janela [aquecimento, horizonte]."""
    n_etapas: int
    ini: float
    fim: float

    def __post_init__(self):
        self.area_fila = [0.0] * self.n_etapas
        self.area_ocup = [0.0] * self.n_etapas
        self.area_wip = 0.0
        self.ultimo = 0.0
        self.esperas: list[list[float]] = [[] for _ in range(self.n_etapas)]

    def avancar(self, t: float, filas: Sequence[int], ocupados: Sequence[int], wip: int) -> None:
        a, b = max(self.ultimo, self.ini), min(t, self.fim)
        if b > a:
            d = b - a
            for i in range(self.n_etapas):
                self.area_fila[i] += filas[i] * d
                self.area_ocup[i] += ocupados[i] * d
            self.area_wip += wip * d
        self.ultimo = t


def _indices_retorno(etapas: Sequence[Etapa]) -> list[int | None]:
    nomes = [e.nome for e in etapas]
    out = []
    for e in etapas:
        if e.prob_retrabalho > 0:
            if e.volta_para not in nomes:
                raise ValueError(f"etapa '{e.nome}': volta_para '{e.volta_para}' não existe")
            out.append(nomes.index(e.volta_para))
        else:
            out.append(None)
    return out


def _replicacao_propria(etapas: Sequence[Etapa], taxa: float, horizonte: float, aquecimento: float,
                        semente: int | None, max_retrabalhos: int) -> dict:
    """Uma replicação com motor próprio: lista de eventos futuros em heap (avanço para o próximo evento)."""
    s = len(etapas)
    fx = _Fluxos(semente, etapas)
    retorno = _indices_retorno(etapas)
    chegadas = _chegadas_poisson(taxa, horizonte, fx.chegada)
    filas: list[deque] = [deque() for _ in range(s)]
    ocupados = [0] * s
    col = _Coletor(s, aquecimento, horizonte)
    eventos: list = []
    seq = 0
    for i, t in enumerate(chegadas):
        heapq.heappush(eventos, (t, seq, "chegada", i, 0))
        seq += 1
    entrada_fila: dict[tuple[int, int], float] = {}
    voltas = [0] * len(chegadas)
    saida = np.full(len(chegadas), np.nan)
    wip = 0

    def iniciar(t: float, caso: int, etapa: int) -> None:
        nonlocal seq
        ocupados[etapa] += 1
        espera = t - entrada_fila.pop((caso, etapa))
        if chegadas[caso] >= aquecimento:
            col.esperas[etapa].append(espera)
        heapq.heappush(eventos, (t + fx.servico[etapa].proximo(), seq, "fim", caso, etapa))
        seq += 1

    def entrar(t: float, caso: int, etapa: int) -> None:
        entrada_fila[(caso, etapa)] = t
        if ocupados[etapa] < etapas[etapa].servidores:
            iniciar(t, caso, etapa)
        else:
            filas[etapa].append(caso)

    while eventos:
        t, _, tipo, caso, etapa = heapq.heappop(eventos)
        col.avancar(t, [len(f) for f in filas], ocupados, wip)
        if tipo == "chegada":
            wip += 1
            entrar(t, caso, 0)
            continue
        ocupados[etapa] -= 1
        if filas[etapa]:
            iniciar(t, filas[etapa].popleft(), etapa)
        e = etapas[etapa]
        if retorno[etapa] is not None and fx.retrabalho[etapa].random() < e.prob_retrabalho \
                and voltas[caso] < max_retrabalhos:
            voltas[caso] += 1
            entrar(t, caso, retorno[etapa])
        elif etapa + 1 < s:
            entrar(t, caso, etapa + 1)
        else:
            saida[caso] = t
            wip -= 1
    return _resumo_replicacao(etapas, chegadas, saida, voltas, col, aquecimento, horizonte)


def _replicacao_simpy(etapas: Sequence[Etapa], taxa: float, horizonte: float, aquecimento: float,
                      semente: int | None, max_retrabalhos: int) -> dict:
    """A mesma replicação escrita em SimPy (processos-geradores e Resource) — usada para verificação cruzada."""
    import simpy

    s = len(etapas)
    fx = _Fluxos(semente, etapas)
    retorno = _indices_retorno(etapas)
    chegadas = _chegadas_poisson(taxa, horizonte, fx.chegada)
    env = simpy.Environment()
    recursos = [simpy.Resource(env, capacity=e.servidores) for e in etapas]
    col = _Coletor(s, aquecimento, horizonte)
    filas = [0] * s
    voltas = [0] * len(chegadas)
    saida = np.full(len(chegadas), np.nan)
    estado = {"wip": 0}

    def marcar():
        col.avancar(env.now, filas, [r.count for r in recursos], estado["wip"])

    def caso(i: int):
        yield env.timeout(chegadas[i])
        marcar()
        estado["wip"] += 1
        etapa = 0
        while True:
            marcar()
            filas[etapa] += 1
            t0 = env.now
            req = recursos[etapa].request()
            yield req
            marcar()
            filas[etapa] -= 1
            if chegadas[i] >= aquecimento:
                col.esperas[etapa].append(env.now - t0)
            yield env.timeout(fx.servico[etapa].proximo())
            marcar()
            recursos[etapa].release(req)
            e = etapas[etapa]
            if retorno[etapa] is not None and fx.retrabalho[etapa].random() < e.prob_retrabalho \
                    and voltas[i] < max_retrabalhos:
                voltas[i] += 1
                etapa = retorno[etapa]
            elif etapa + 1 < s:
                etapa += 1
            else:
                saida[i] = env.now
                estado["wip"] -= 1
                marcar()
                return

    for i in range(len(chegadas)):
        env.process(caso(i))
    env.run()
    return _resumo_replicacao(etapas, chegadas, saida, voltas, col, aquecimento, horizonte)


def _resumo_replicacao(etapas, chegadas, saida, voltas, col: _Coletor, aquecimento, horizonte) -> dict:
    janela = horizonte - aquecimento
    sel = chegadas >= aquecimento
    w = saida[sel] - chegadas[sel]
    return {
        "atravessamento": w,
        "w_medio": float(np.mean(w)) if len(w) else float("nan"),
        "w_p80": float(np.percentile(w, 80)) if len(w) else float("nan"),
        "w_p90": float(np.percentile(w, 90)) if len(w) else float("nan"),
        "casos": int(sel.sum()),
        "retrabalhos": int(np.asarray(voltas)[sel].sum()),
        "fila_media": [a / janela for a in col.area_fila],
        "espera_media": [float(np.mean(x)) if x else 0.0 for x in col.esperas],
        "utilizacao": [col.area_ocup[i] / (e.servidores * janela) for i, e in enumerate(etapas)],
        "wip_medio": col.area_wip / janela,
        "taxa_chegada_obs": float(sel.sum() / janela),
    }


@dataclass
class ResultadoFila:
    """Resumo de R replicações independentes (média e meia-largura do IC 95% pela t de Student)."""
    etapas: list[str]
    servidores: list[int]
    replicacoes: list[dict]
    prazo: float | None = None

    def _ic(self, valores) -> tuple[float, float]:
        v = np.asarray(valores, dtype=float)
        r = len(v)
        if r < 2:
            return float(v.mean()), float("nan")
        return float(v.mean()), float(stats.t.ppf(0.975, r - 1) * v.std(ddof=1) / math.sqrt(r))

    def resumo(self) -> dict:
        reps = self.replicacoes
        out = {k: self._ic([r[k] for r in reps]) for k in ("w_medio", "w_p80", "w_p90", "wip_medio")}
        out["fila_media"] = [self._ic([r["fila_media"][i] for r in reps]) for i in range(len(self.etapas))]
        out["espera_media"] = [self._ic([r["espera_media"][i] for r in reps]) for i in range(len(self.etapas))]
        out["utilizacao"] = [self._ic([r["utilizacao"][i] for r in reps]) for i in range(len(self.etapas))]
        out["casos_por_replicacao"] = float(np.mean([r["casos"] for r in reps]))
        out["retrabalho_por_caso"] = float(np.sum([r["retrabalhos"] for r in reps]) /
                                           max(1, np.sum([r["casos"] for r in reps])))
        if self.prazo is not None:
            out["nivel_servico"] = self._ic([float(np.mean(r["atravessamento"] <= self.prazo)) for r in reps])
        # verificação pela lei de Little (L = λ·W) — LITTLE, 1961
        lam = np.mean([r["taxa_chegada_obs"] for r in reps])
        out["little_l"] = out["wip_medio"][0]
        out["little_lambda_w"] = float(lam * out["w_medio"][0])
        return out


def simular_processo(etapas: Sequence[Etapa], taxa_chegada: float, horizonte: float = 2000.0,
                     aquecimento: float = 200.0, replicacoes: int = 10, semente: int | None = 20261105,
                     motor: str = "proprio", prazo: float | None = None,
                     max_retrabalhos: int = 20) -> ResultadoFila:
    """Simulação de eventos discretos do processo, com R replicações independentes (sementes derivadas).

    taxa_chegada: casos por hora útil (Poisson). horizonte e aquecimento em horas úteis; as estatísticas só contam a
    janela [aquecimento, horizonte] e os casos chegados nela (que são acompanhados até a saída). motor: "proprio" ou
    "simpy" (exige `pip install simpy`). prazo: meta de atravessamento (h) para o nível de serviço (% no prazo).
    """
    if not 0 <= aquecimento < horizonte:
        raise ValueError("exige 0 ≤ aquecimento < horizonte")
    if taxa_chegada <= 0:
        raise ValueError("taxa de chegada deve ser positiva")
    for e in etapas:
        if e.servidores < 1:
            raise ValueError(f"etapa '{e.nome}' precisa de pelo menos 1 servidor")
    if motor not in ("proprio", "simpy"):
        raise ValueError("motor deve ser 'proprio' ou 'simpy'")
    funcao = _replicacao_propria if motor == "proprio" else _replicacao_simpy
    sementes = np.random.SeedSequence(semente).generate_state(replicacoes)
    reps = [funcao(etapas, taxa_chegada, horizonte, aquecimento, int(sm), max_retrabalhos) for sm in sementes]
    return ResultadoFila([e.nome for e in etapas], [e.servidores for e in etapas], reps, prazo)


def carga_por_etapa(etapas: Sequence[Etapa], taxa_chegada: float) -> list[float]:
    """Intensidade de tráfego a = λ·visitas·E[S] por etapa (em servidores ocupados). Utilização = a / c.

    As visitas resolvem as equações de tráfego com o retrabalho (v = λ·(I − P)⁻¹) — rede de Jackson.
    """
    s = len(etapas)
    retorno = _indices_retorno(etapas)
    p = np.zeros((s, s))
    for i, e in enumerate(etapas):
        q = e.prob_retrabalho if retorno[i] is not None else 0.0
        if retorno[i] is not None:
            p[i, retorno[i]] += q
        if i + 1 < s:
            p[i, i + 1] += 1 - q
    entrada = np.zeros(s)
    entrada[0] = 1.0
    visitas = np.linalg.solve((np.eye(s) - p).T, entrada)
    return [float(taxa_chegada * visitas[i] * e.media()) for i, e in enumerate(etapas)]


def varrer_servidores(etapas: Sequence[Etapa], taxa_chegada: float, etapa_variar: str | None = None,
                      valores: Iterable[int] = (1, 2, 3, 4, 5), **kw) -> list[dict]:
    """Repete a simulação variando o nº de servidores de uma etapa (padrão: a de maior carga, o gargalo).

    Usa a mesma semente (números aleatórios comuns) para que as diferenças reflitam só a mudança de capacidade.
    Configurações com utilização teórica ≥ 1 são marcadas como instáveis (a fila cresce sem limite no longo prazo).
    """
    nomes = [e.nome for e in etapas]
    cargas = carga_por_etapa(etapas, taxa_chegada)
    if etapa_variar is None:
        etapa_variar = nomes[int(np.argmax([c / e.servidores for c, e in zip(cargas, etapas)]))]
    j = nomes.index(etapa_variar)
    linhas = []
    for c in valores:
        conf = [Etapa(e.nome, c if i == j else e.servidores, e.tempo, e.prob_retrabalho, e.volta_para)
                for i, e in enumerate(etapas)]
        rho_teorico = cargas[j] / c
        res = simular_processo(conf, taxa_chegada, **kw).resumo()
        linhas.append({"etapa": etapa_variar, "servidores": c, "rho_teorico": rho_teorico,
                       "instavel": rho_teorico >= 1, "resumo": res})
    return linhas


def erlang_c(taxa_chegada: float, tempo_medio: float, c: int) -> dict[str, float]:
    """M/M/c analítico (Erlang C): P(esperar), Wq, W, Lq — referência de validação (SHORTLE et al., 2018)."""
    a = taxa_chegada * tempo_medio
    rho = a / c
    if rho >= 1:
        return {"rho": rho, "p_espera": 1.0, "wq": float("inf"), "w": float("inf"), "lq": float("inf")}
    soma = sum(a ** k / math.factorial(k) for k in range(c))
    topo = a ** c / (math.factorial(c) * (1 - rho))
    pw = topo / (soma + topo)
    wq = pw * tempo_medio / (c * (1 - rho))
    return {"rho": rho, "p_espera": pw, "wq": wq, "w": wq + tempo_medio, "lq": taxa_chegada * wq}


def curva_welch(etapas: Sequence[Etapa], taxa_chegada: float, horizonte: float, replicacoes: int = 5,
                janela: int = 25, semente: int | None = 20261105) -> np.ndarray:
    """Procedimento de Welch para escolher o aquecimento: média entre replicações do atravessamento por ordem de
    chegada, suavizada por média móvel de `janela` casos (LAW, 2015, §9.5.1). Lê-se onde a curva se estabiliza."""
    sementes = np.random.SeedSequence(semente).generate_state(replicacoes)
    series = [_replicacao_propria(etapas, taxa_chegada, horizonte, 0.0, int(s), 20)["atravessamento"]
              for s in sementes]
    m = min(len(x) for x in series)
    media = np.mean([x[:m] for x in series], axis=0)
    if m < janela:
        return media
    return np.convolve(media, np.ones(janela) / janela, mode="valid")


# =====================================================================================================================
# 6. Gráficos
# =====================================================================================================================

def grafico_esforco(res: ResultadoEsforco, caminho: str, disponibilidade: float | None = None) -> str:
    """Histograma da carga total + curva acumulada (eixo secundário) com o P80 marcado; tornado ao lado."""
    plt = estilo_graficos()
    from matplotlib.ticker import FuncFormatter
    milhar = FuncFormatter(lambda v, _: br(v, 0))
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.75})
    ax.hist(res.total, bins=50, color=PALETA["azul"], alpha=0.85, density=True, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("carga anual do setor (h)")
    ax.set_ylabel("frequência relativa")
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(milhar)
    ax2 = ax.twinx()
    x = np.sort(res.total)
    ax2.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, color=PALETA["marinho"], lw=2)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("acumulada (%)")
    ax2.spines["right"].set_visible(True)
    p80 = res.percentis[80]
    ax.axvline(p80, color=PALETA["verde_escuro"], lw=2.2, ls="--")
    rot = f"P80 = {br(p80, 0)} h"
    if disponibilidade:
        rot += f"\n= {br(p80 / disponibilidade, 2)} servidores"
    ax2.text(p80, 45, " " + rot + "\n cobre 80% dos cenários", color=PALETA["verde_escuro"],
            va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    ax.set_title("Monte Carlo do esforço anual", loc="left", color=PALETA["marinho"])
    linhas = tornado(res)[::-1]
    base = linhas[0]["base"] if linhas else 0
    for i, l in enumerate(linhas):
        axt.barh(i, l["alto"] - base, left=base, color=PALETA["azul"])
        axt.barh(i, l["baixo"] - base, left=base, color=PALETA["marinho"])
    axt.set_yticks(range(len(linhas)), [textwrap.fill(l["entrega"], 24) for l in linhas], fontsize=9)
    axt.axvline(base, color=PALETA["cinza"], lw=1)
    axt.xaxis.set_major_formatter(milhar)
    axt.tick_params(axis="x", labelrotation=30)
    axt.set_xlabel("carga total (h) — entrega no P10 (escuro) e no P90 (claro)", fontsize=9)
    axt.set_title("Tornado de sensibilidade", loc="left", color=PALETA["marinho"])
    formatar_eixos_br(fig)
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_servidores(linhas: list[dict], caminho: str, prazo: float | None = None) -> str:
    """Atravessamento médio e P90 (com IC) e utilização do gargalo em função do nº de servidores."""
    plt = estilo_graficos()
    est = [l for l in linhas if not l["instavel"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    c = [l["servidores"] for l in est]
    for chave, rot, cor in (("w_medio", "médio", PALETA["azul"]), ("w_p90", "P90", PALETA["marinho"])):
        a1.errorbar(c, [l["resumo"][chave][0] for l in est], yerr=[l["resumo"][chave][1] for l in est],
                    marker="o", capsize=4, label=rot, color=cor)
    if prazo:
        a1.axhline(prazo, color=PALETA["verde_escuro"], ls="--", label=f"prazo {br(prazo, 0)} h")
    a1.set_xlabel(f"servidores na etapa '{linhas[0]['etapa']}'")
    a1.set_ylabel("tempo de atravessamento (h)")
    a1.legend(frameon=False)
    a1.set_xticks(c)
    cs = [l["servidores"] for l in linhas]
    a2.bar(cs, [min(l["rho_teorico"], 1.2) * 100 for l in linhas],
           color=[PALETA["cinza"] if l["instavel"] else PALETA["azul"] for l in linhas])
    a2.axhline(100, color=PALETA["marinho"], lw=1)
    for l in linhas:
        if l["instavel"]:
            a2.text(l["servidores"], min(l["rho_teorico"], 1.2) * 100 + 2, "instável", ha="center", fontsize=9,
                    color=PALETA["cinza"])
    a2.set_ylabel("utilização teórica do gargalo (%)")
    a2.set_xlabel("servidores")
    a2.set_xticks(cs)
    fig.suptitle("Quantos servidores mudam o nível de serviço", x=0.02, ha="left", color=PALETA["marinho"])
    formatar_eixos_br(fig)
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# =====================================================================================================================
# 7. Leitura de CSV e relatórios
# =====================================================================================================================

def _col(df, *nomes, padrao=None):
    for n in nomes:
        if n in df.columns:
            return df[n]
    if padrao is not None:
        return [padrao] * len(df)
    raise KeyError(f"coluna ausente: {nomes[0]}")


def ler_entregas(caminho: str) -> list[Entrega]:
    """CSV de entregas: entrega; volume (fixo) OU volume_min; volume_mp; volume_max; tempo_min; tempo_mp; tempo_max;
    distribuicao (beta_pert | lognormal_deslocada | triangular; padrão beta_pert)."""
    df = ler_csv(caminho)
    df.columns = [c.strip().lower() for c in df.columns]
    out = []
    dist = _col(df, "distribuicao", padrao="beta_pert")
    for i, r in df.reset_index(drop=True).iterrows():
        if "volume_min" in df.columns:
            v = TresPontos(float(r["volume_min"]), float(r["volume_mp"]), float(r["volume_max"]))
        else:
            v = TresPontos(float(r["volume"]), float(r["volume"]), float(r["volume"]))
        d = str(dist[i]).strip() if str(dist[i]).strip() not in ("", "nan") else "beta_pert"
        out.append(Entrega(str(r["entrega"]), v,
                           TresPontos(float(r["tempo_min"]), float(r["tempo_mp"]), float(r["tempo_max"]), d)))
    return out


def ler_etapas(caminho: str) -> list[Etapa]:
    """CSV do processo: etapa; servidores; tempo_min; tempo_mp; tempo_max; distribuicao; prob_retrabalho; volta_para."""
    df = ler_csv(caminho)
    df.columns = [c.strip().lower() for c in df.columns]
    out = []
    for _, r in df.iterrows():
        pr = float(r.get("prob_retrabalho", 0) or 0)
        vp = r.get("volta_para")
        vp = None if vp is None or (isinstance(vp, float) and math.isnan(vp)) or str(vp).strip() == "" else str(vp)
        dist = str(r.get("distribuicao", "beta_pert")).strip() or "beta_pert"
        out.append(Etapa(str(r["etapa"]), int(r["servidores"]),
                         TresPontos(float(r["tempo_min"]), float(r["tempo_mp"]), float(r["tempo_max"]), dist),
                         pr if vp else 0.0, vp))
    return out


def relatorio_esforco(res: ResultadoEsforco, entregas: Sequence[Entrega], disponibilidades: Sequence[float]) -> str:
    """Relatório legível do Monte Carlo do esforço."""
    l = [f"Monte Carlo do esforço anual — N = {br(res.n, 0)} cenários, semente = {res.semente}",
         f"Correlação de postos entre tempos: "
         f"{br(res.correlacao, 2) if np.isscalar(res.correlacao) else 'matriz informada'}"
         f"{' (' + res.metodo_correlacao + ')' if np.any(np.asarray(res.correlacao) != 0) else ''}; "
         f"agregação das execuções: {res.agregacao}",
         "", f"{'Entrega':<36}{'volume (mp)':>12}{'P80 tempo (h)':>15}{'P80 horas/ano':>15}"]
    for j, e in enumerate(entregas):
        l.append(f"{e.nome[:35]:<36}{br(e.volume.mais_provavel, 1):>12}{br(quantil_analitico(e.tempo, .8), 3):>15}"
                 f"{br(res.p80_individuais[e.nome], 1):>15}")
    l.append("")
    l.append("Carga total do setor (h/ano):  " + "   ".join(f"P{p} = {br(v, 1)}" for p, v in res.percentis.items()
                                                          if p in (50, 80, 90)))
    em = erro_mc_p80(res)
    l.append(f"Erro de Monte Carlo do P80: erro-padrão ≈ {br(em['erro_padrao'], 1)} h "
             f"({br(100 * em['erro_relativo'], 2)}%); IC 95% (estatísticas de ordem) = "
             f"[{br(em['ic_inf'], 1)}; {br(em['ic_sup'], 1)}] h")
    for h in disponibilidades:
        l.append(f"Servidores com disponibilidade efetiva de {br(h, 2)} h/ano:")
        for p in (50, 80, 90):
            q, qi = servidores(res.percentis[p], h)
            l.append(f"   P{p}: {br(q, 2)} → {qi}")
    soma = res.soma_p80_individuais
    margem = soma - res.p80_total
    l += ["", "Advertência da carteira (L2, §3.8): soma dos P80 > P80 da soma, salvo correlação perfeita.",
          f"   soma dos P80 por entrega = {br(soma, 1)} h   |   P80 da carga total = {br(res.p80_total, 1)} h",
          f"   margem de conservadorismo = {br(margem, 1)} h ({br(100 * margem / soma if soma else 0, 1)}%)"
          + (f" = {br(margem / disponibilidades[0], 2)} servidores" if disponibilidades else ""),
          f"   regra oficial do quadro (Σ volume × P80 analítico) = {br(soma_p80_analitica(entregas), 1)} h",
          "", "Tornado de sensibilidade (maior influência primeiro):",
          f"{'Entrega':<36}{'ρ postos':>10}{'contrib. var.':>15}{'P10→P90 (h)':>15}"]
    for t in tornado(res):
        l.append(f"{t['entrega'][:35]:<36}{br(t['rho_postos'], 2):>10}{br(100 * t['contribuicao_variancia'], 1):>14}%"
                 f"{br(t['balanco'], 0):>15}")
    l.append("\nO percentil (P80) é apetite ao risco declarado na Fase 1 (decisão D3/L2 §3.8), não resultado do modelo.")
    return "\n".join(l)


def relatorio_fila(linhas: list[dict], etapas: Sequence[Etapa], taxa: float, prazo: float | None) -> str:
    """Relatório legível da varredura de servidores."""
    cargas = carga_por_etapa(etapas, taxa)
    l = [f"Simulação de eventos discretos — chegadas Poisson de {br(taxa, 2)} casos/h útil",
         "Carga teórica por etapa (servidores ocupados em média = λ·visitas·tempo médio):"]
    for e, c in zip(etapas, cargas):
        ret = f"; retrabalho {br(100 * e.prob_retrabalho, 0)}% → {e.volta_para}" if e.prob_retrabalho else ""
        l.append(f"   {e.nome:<28} {br(c, 2)}  (c = {e.servidores}{ret})")
    l.append(f"\nVariando a etapa '{linhas[0]['etapa']}' (demais etapas fixas; números aleatórios comuns):")
    cab = f"{'c':>3}{'ρ teórico':>11}{'W médio (h)':>16}{'W P90 (h)':>12}{'fila média':>12}{'espera (h)':>12}"
    if prazo:
        cab += f"{'no prazo':>10}"
    l.append(cab)
    nomes = [e.nome for e in etapas]
    j = nomes.index(linhas[0]["etapa"])
    for x in linhas:
        r = x["resumo"]
        linha = (f"{x['servidores']:>3}{br(100 * x['rho_teorico'], 0) + '%':>11}"
                 f"{br(r['w_medio'][0], 2) + ' ± ' + br(r['w_medio'][1], 2):>16}{br(r['w_p90'][0], 2):>12}"
                 f"{br(r['fila_media'][j][0], 2):>12}{br(r['espera_media'][j][0], 2):>12}")
        if prazo:
            linha += f"{br(100 * r['nivel_servico'][0], 1) + '%':>10}"
        if x["instavel"]:
            linha += "   ← instável (ρ ≥ 1): a fila só cresce; números dependem do horizonte"
        l.append(linha)
    est = [x for x in linhas if not x["instavel"]]
    if est:
        r = est[0]["resumo"]
        l.append(f"\nVerificação (lei de Little, c = {est[0]['servidores']}): L = {br(r['little_l'], 2)} casos ≈ "
                 f"λ·W = {br(r['little_lambda_w'], 2)}")
    l.append("± = meia-largura do IC 95% entre replicações (t de Student).")
    return "\n".join(l)


def _intervalo_inteiros(txt: str) -> list[int]:
    if "-" in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in txt.split(",")]


def cli(argv: list[str]) -> int:
    """`dft simulacao <arquivo.csv> [opções]` — esforço (CSV de entregas) ou fila (CSV de etapas, detecta sozinho)."""
    ap = argparse.ArgumentParser(prog="dft simulacao", description="Simulação de Monte Carlo do esforço anual e "
                                 "simulação de eventos discretos de um processo de atendimento.")
    ap.add_argument("arquivo", help="CSV de entregas (coluna 'entrega') ou de etapas do processo (coluna 'etapa')")
    ap.add_argument("--modo", choices=("auto", "esforco", "fila"), default="auto")
    ap.add_argument("--n", type=int, default=N_PADRAO, help="cenários de Monte Carlo (padrão 10.000)")
    ap.add_argument("--semente", type=int, default=20261105)
    ap.add_argument("--disponibilidade", type=float, nargs="+", default=[DISPONIBILIDADE_CT3],
                    help="disponibilidade efetiva em h/servidor/ano (ex.: 1153.6 928.2)")
    ap.add_argument("--correlacao", type=float, default=0.0, help="correlação de postos entre tempos (−1 a 1)")
    ap.add_argument("--metodo-correlacao", choices=("copula", "iman_conover"), default="copula")
    ap.add_argument("--agregacao", choices=("comum", "independente"), default="comum")
    ap.add_argument("--chegadas", type=float, default=1.3, help="fila: casos por hora útil (Poisson)")
    ap.add_argument("--servidores", default="1-5", help="fila: valores a testar na etapa variada, ex.: 1-5 ou 2,3,4")
    ap.add_argument("--etapa", default=None, help="fila: etapa a variar (padrão: o gargalo)")
    ap.add_argument("--horizonte", type=float, default=2000.0, help="fila: horas úteis por replicação")
    ap.add_argument("--aquecimento", type=float, default=200.0, help="fila: horas descartadas no início")
    ap.add_argument("--replicacoes", type=int, default=10)
    ap.add_argument("--prazo", type=float, default=None, help="fila: meta de atravessamento (h) para o nível de serviço")
    ap.add_argument("--motor", choices=("proprio", "simpy"), default="proprio")
    ap.add_argument("--grafico", default=None, help="salva figura PNG neste caminho")
    a = ap.parse_args(argv)
    df = ler_csv(a.arquivo)
    cols = {c.strip().lower() for c in df.columns}
    modo = a.modo if a.modo != "auto" else ("fila" if "etapa" in cols else "esforco")
    if modo == "esforco":
        entregas = ler_entregas(a.arquivo)
        res = simular_esforco(entregas, n=a.n, semente=a.semente, correlacao=a.correlacao,
                              metodo_correlacao=a.metodo_correlacao, agregacao=a.agregacao)
        print(relatorio_esforco(res, entregas, a.disponibilidade))
        if a.agregacao == "comum":
            ind = simular_esforco(entregas, n=a.n, semente=a.semente, correlacao=a.correlacao,
                                  metodo_correlacao=a.metodo_correlacao, agregacao="independente")
            print(f"\nLeitura complementar: se as execuções de cada entrega fossem independentes entre si (sem causa "
                  f"comum), o P80 da carga seria {br(ind.p80_total, 1)} h — a variação entre casos se compensa ao "
                  f"longo do ano. O curso adota 'comum' (a incerteza é sobre o tempo típico do setor).")
        if a.grafico:
            grafico_esforco(res, a.grafico, a.disponibilidade[0])
            print(f"\nGráfico salvo em {a.grafico}")
    else:
        etapas = ler_etapas(a.arquivo)
        linhas = varrer_servidores(etapas, a.chegadas, a.etapa, _intervalo_inteiros(a.servidores),
                                   horizonte=a.horizonte, aquecimento=a.aquecimento, replicacoes=a.replicacoes,
                                   semente=a.semente, motor=a.motor, prazo=a.prazo)
        print(relatorio_fila(linhas, etapas, a.chegadas, a.prazo))
        if a.grafico:
            grafico_servidores(linhas, a.grafico, a.prazo)
            print(f"\nGráfico salvo em {a.grafico}")
    return 0
