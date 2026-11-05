"""1. Mapeamento de processos e análise de capacidade.

Converte um fluxo de atividades (com desvios e retrabalho) em carga por atividade e compara carga com capacidade
por etapa: capacidade teórica, efetiva e demonstrada; utilização; gargalo; takt time e tempo de ciclo; lei de
Little (WIP = vazão × lead time); e efeito da utilização sobre a fila (Kingman/VUT e Erlang C, M/M/c).

Convenções de unidade (valem para o módulo inteiro):
    * tempo de execução das atividades em MINUTOS por execução;
    * horas disponíveis em HORAS por pessoa no período (horas efetivas, já descontadas ausências e hora útil da
      Fase 7), salvo quando o parâmetro ``horas_totais=True`` indicar que a coluna já é o total da etapa;
    * demanda, capacidade e vazão em UNIDADES (casos, processos, atendimentos) por período;
    * lead time na mesma unidade de tempo do período (ex.: mês) quando vem da lei de Little.

Referências do método (detalhes e referências ABNT em docs/metodos/1_capacidade.md):
    LITTLE (1961; 2011) — L = λW; KINGMAN (1961) e HOPP; SPEARMAN (2008) — aproximação VUT;
    SHORTLE et al. (2018) e GANS; KOOLE; MANDELBAUM (2003) — M/M/c e Erlang C;
    DUMAS et al. (2018, cap. 7) — análise de fluxo com desvios (XOR), paralelismo (AND) e retrabalho;
    ABPMP (2013, §4.4.6–4.4.7) — análise de capacidade e gargalo; ARNOLD; CHAPMAN; CLIVE (2017) — capacidade
    nominal e demonstrada; ROTHER; SHOOK (1999) — takt time.
"""
from __future__ import annotations

import argparse
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._comum import PALETA, br, estilo_graficos, formatar_eixos_br, ler_csv

INICIO = "INICIO"
FIM = "FIM"

#: Limite de alerta de utilização (zona em que a fila cresce depressa). Convenção prática, não norma; ver dossiê.
LIMITE_ALERTA = 0.85


# ---------------------------------------------------------------------------------------------------------------
# 1. Grandezas básicas por etapa
# ---------------------------------------------------------------------------------------------------------------

def capacidade_etapa(pessoas: float, horas_disponiveis: float, tempo_min: float, horas_totais: bool = False) -> float:
    """Capacidade efetiva de uma etapa, em unidades por período.

    capacidade = (pessoas × horas por pessoa × 60) / tempo por execução (min).
    Com ``horas_totais=True``, ``horas_disponiveis`` já é o total da etapa e ``pessoas`` não multiplica.
    """
    if tempo_min <= 0:
        raise ValueError("tempo por execução deve ser positivo (minutos)")
    horas = horas_disponiveis if horas_totais else pessoas * horas_disponiveis
    return horas * 60.0 / tempo_min


def carga_horas(demanda: float, tempo_min: float) -> float:
    """Carga de trabalho da etapa no período, em horas: demanda × tempo (min) / 60."""
    return demanda * tempo_min / 60.0


def utilizacao(demanda: float, tempo_min: float, pessoas: float, horas_disponiveis: float,
               horas_totais: bool = False) -> float:
    """Utilização ρ = carga / horas disponíveis = demanda / capacidade (adimensional).

    ρ > 1 significa que a etapa não dá conta da demanda: a fila cresce sem limite (sistema instável).
    """
    horas = horas_disponiveis if horas_totais else pessoas * horas_disponiveis
    if horas <= 0:
        return math.inf
    return carga_horas(demanda, tempo_min) / horas


def tempo_ciclo_etapa(tempo_min: float, pessoas: float) -> float:
    """Tempo de ciclo da etapa (min/unidade): intervalo médio entre duas saídas com todos trabalhando em paralelo."""
    if pessoas <= 0:
        return math.inf
    return tempo_min / pessoas


def takt_time(tempo_disponivel_min: float, demanda: float) -> float:
    """Takt time (min/unidade) = tempo disponível no período / demanda do período (ROTHER; SHOOK, 1999).

    É o ritmo que a demanda impõe. Uma etapa cujo tempo de ciclo excede o takt não acompanha a demanda.
    Com o tempo disponível por pessoa, vale a identidade ρ = tempo de ciclo / takt.
    """
    if demanda <= 0:
        return math.inf
    return tempo_disponivel_min / demanda


def pessoas_necessarias(carga_h: float, horas_por_pessoa: float, utilizacao_alvo: float = 1.0) -> float:
    """Pessoas necessárias (fracionário) = carga / (horas por pessoa × utilização-alvo).

    Com ``utilizacao_alvo=1`` é exatamente o Q do DFT (Q = Σ Dᵢ·Eᵢ / H). Use alvo < 1 apenas quando houver
    justificativa de fila/nível de serviço (ver ``servidores_para_meta``); a hora útil da Fase 7 já desconta o
    tempo não produtivo, e descontar de novo é dupla contagem.
    """
    if not 0 < utilizacao_alvo <= 1:
        raise ValueError("utilização-alvo deve estar em (0, 1]")
    return carga_h / (horas_por_pessoa * utilizacao_alvo)


# ---------------------------------------------------------------------------------------------------------------
# 2. Tipos de capacidade
# ---------------------------------------------------------------------------------------------------------------

def capacidades(pessoas: float, horas_brutas_pessoa: float, tempo_min: float, fator_disponibilidade: float = 1.0,
                eficiencia: float = 1.0) -> dict[str, float]:
    """Capacidade teórica e efetiva (nominal) de uma etapa, em unidades por período.

    * teórica (de projeto): todas as horas brutas da jornada, ao tempo-padrão, sem perdas;
    * efetiva (nominal, *rated*): teórica × fator de disponibilidade (ausências, hora útil) × eficiência
      (ARNOLD; CHAPMAN; CLIVE, 2017: capacidade nominal = tempo disponível × utilização × eficiência).
    A capacidade demonstrada não se calcula: mede-se (``capacidade_demonstrada``).
    """
    teorica = capacidade_etapa(pessoas, horas_brutas_pessoa, tempo_min)
    return {"teorica": teorica, "efetiva": teorica * fator_disponibilidade * eficiencia}


def capacidade_demonstrada(producao_por_periodo: Iterable[float]) -> float:
    """Capacidade demonstrada = média da produção realizada em períodos anteriores.

    Atenção: só mede capacidade se a etapa esteve saturada (com fila) em todos os períodos; caso contrário mede a
    demanda atendida. Onde há fila, a série histórica é teto de capacidade, não demanda (dossiê L1, Fase 4).
    """
    x = np.asarray(list(producao_por_periodo), dtype=float)
    if x.size == 0:
        raise ValueError("série de produção vazia")
    return float(x.mean())


def indicadores_capacidade(producao_real: float, teorica: float, efetiva: float) -> dict[str, float]:
    """Indicadores clássicos de manual de operações: utilização = real/teórica; eficiência = real/efetiva.

    Não confundir esta "utilização" (sobre a capacidade teórica) com a utilização ρ das filas
    (carga/capacidade efetiva), usada no resto do módulo.
    """
    return {"utilizacao_sobre_teorica": producao_real / teorica, "eficiencia_sobre_efetiva": producao_real / efetiva}


# ---------------------------------------------------------------------------------------------------------------
# 3. Lei de Little e eficiência do ciclo
# ---------------------------------------------------------------------------------------------------------------

def little(wip: float | None = None, vazao: float | None = None, lead_time: float | None = None) -> float:
    """Lei de Little, L = λW (LITTLE, 1961): resolve a grandeza que faltar.

    ``wip`` (casos em andamento), ``vazao`` (casos/período), ``lead_time`` (períodos). Informe exatamente duas.
    Vale para médias de longo prazo em sistema estável; com demanda > capacidade o WIP cresce e o resultado é
    apenas uma fotografia do momento.
    """
    dados = [wip is None, vazao is None, lead_time is None]
    if sum(dados) != 1:
        raise ValueError("informe exatamente duas das três grandezas: wip, vazao, lead_time")
    if wip is None:
        return vazao * lead_time
    if vazao is None:
        return wip / lead_time if lead_time else math.inf
    return wip / vazao if vazao else math.inf


def eficiencia_ciclo(tempo_trabalho: float, lead_time: float) -> float:
    """Eficiência do ciclo do processo (PCE) = tempo de trabalho efetivo / lead time (mesma unidade)."""
    return tempo_trabalho / lead_time if lead_time > 0 else math.nan


# ---------------------------------------------------------------------------------------------------------------
# 4. Filas: Kingman/VUT e M/M/c (Erlang C)
# ---------------------------------------------------------------------------------------------------------------

def kingman_espera(u: float, ca2: float, ce2: float, te: float, c: int = 1) -> float:
    """Tempo médio em fila pela aproximação VUT (KINGMAN, 1961; HOPP; SPEARMAN, 2008).

    Wq ≈ V × U × T, com V = (ca² + ce²)/2, U = u^(√(2(c+1)) − 1) / (c(1 − u)) e T = te.
    ``u`` utilização (0 ≤ u < 1); ``ca2``/``ce2`` quadrados dos coeficientes de variação das chegadas e do
    serviço; ``te`` tempo médio de serviço; ``c`` servidores em paralelo. Com c = 1 e ca² = ce² = 1 é exata (M/M/1).
    """
    if u >= 1:
        return math.inf
    if u <= 0:
        return 0.0
    expo = math.sqrt(2 * (c + 1)) - 1
    return (ca2 + ce2) / 2 * (u ** expo / (c * (1 - u))) * te


def erlang_b(c: int, a: float) -> float:
    """Probabilidade de bloqueio de Erlang B por recursão estável: B(0)=1; B(k)=a·B/(k + a·B)."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """Probabilidade de esperar (Erlang C) num M/M/c com tráfego a = λ/μ (erlangs). Exige a < c."""
    if c <= 0:
        raise ValueError("c deve ser ≥ 1")
    if a >= c:
        return 1.0
    b = erlang_b(c, a)
    return c * b / (c - a * (1 - b))


def mmc(lam: float, mu: float, c: int) -> dict[str, float]:
    """Medidas de desempenho do M/M/c (SHORTLE et al., 2018). ``lam`` chegadas e ``mu`` atendimentos por servidor,
    na mesma unidade de tempo. Retorna rho, p_espera, Lq, Wq, W, L (Wq e W na unidade de tempo de lam)."""
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return {"rho": rho, "p_espera": 1.0, "Lq": math.inf, "Wq": math.inf, "W": math.inf, "L": math.inf}
    pw = erlang_c(c, a)
    wq = pw / (c * mu - lam)
    w = wq + 1 / mu
    return {"rho": rho, "p_espera": pw, "Lq": lam * wq, "Wq": wq, "W": w, "L": lam * w}


def nivel_servico(lam: float, mu: float, c: int, meta_espera: float) -> float:
    """Fração de casos atendidos com espera ≤ meta_espera no M/M/c: 1 − C(c,a)·exp(−(cμ − λ)·t)."""
    a = lam / mu
    if a >= c:
        return 0.0
    return 1 - erlang_c(c, a) * math.exp(-(c * mu - lam) * meta_espera)


def servidores_para_meta(lam: float, mu: float, meta_espera: float, nivel: float = 0.8, c_max: int = 500) -> int:
    """Menor número de servidores que atinge o nível de serviço (ex.: 80% atendidos em até 10 min) no M/M/c."""
    c = max(1, math.floor(lam / mu) + 1)
    while c <= c_max:
        if nivel_servico(lam, mu, c, meta_espera) >= nivel:
            return c
        c += 1
    raise ValueError("meta inatingível até c_max servidores")


# ---------------------------------------------------------------------------------------------------------------
# 5. Do fluxo à carga: desvios, paralelismo e retrabalho
# ---------------------------------------------------------------------------------------------------------------

def tempo_xor(tempos: Sequence[float], probabilidades: Sequence[float]) -> float:
    """Bloco de decisão exclusiva (XOR): T = Σ pᵢ·Tᵢ (DUMAS et al., 2018, cap. 7). Probabilidades somam 1."""
    if not math.isclose(sum(probabilidades), 1.0, abs_tol=1e-9):
        raise ValueError("as probabilidades de um XOR devem somar 1")
    return float(sum(p * t for p, t in zip(probabilidades, tempos)))


def tempo_and(tempos: Sequence[float]) -> dict[str, float]:
    """Bloco paralelo (AND): duração = max(Tᵢ); trabalho (esforço) = Σ Tᵢ. Para o DFT, conta o trabalho."""
    return {"duracao": float(max(tempos)), "trabalho": float(sum(tempos))}


def tempo_retrabalho(tempo: float, prob_retrabalho: float) -> float:
    """Bloco de retrabalho: T / (1 − r), pois o número esperado de execuções é 1/(1 − r) (DUMAS et al., 2018)."""
    if not 0 <= prob_retrabalho < 1:
        raise ValueError("probabilidade de retrabalho deve estar em [0, 1)")
    return tempo / (1 - prob_retrabalho)


def _transicoes_para_lista(transicoes) -> list[tuple[str, str, float]]:
    if isinstance(transicoes, pd.DataFrame):
        df = _normalizar_colunas(transicoes)
        return [(str(o).strip(), str(d).strip(), float(p))
                for o, d, p in zip(df["origem"], df["destino"], df["probabilidade"])]
    return [(str(o).strip(), str(d).strip(), float(p)) for o, d, p in transicoes]


def validar_fluxo(transicoes, inicio: str = INICIO, fim: str = FIM) -> list[str]:
    """Verifica o fluxo e devolve avisos (lista vazia = ok).

    Regras (alinhadas às diretrizes 7PMG: um início, um fim; e às regras de fluxo de sequência do BPMN 2.0):
    probabilidade em [0, 1]; toda atividade tem saída; soma das saídas = 1 (XOR) — soma > 1 só é aceitável em
    divisão paralela (AND), onde cada ramo recebe probabilidade 1; atividade alcançável a partir do início.
    """
    lista = _transicoes_para_lista(transicoes)
    avisos: list[str] = []
    nos = {o for o, _, _ in lista} | {d for _, d, _ in lista}
    if inicio not in nos:
        avisos.append(f"não há evento de início '{inicio}'")
    if fim not in nos:
        avisos.append(f"não há evento de fim '{fim}'")
    saidas: dict[str, float] = {}
    for o, d, p in lista:
        if not 0 <= p <= 1:
            avisos.append(f"probabilidade fora de [0, 1] em {o} → {d}: {p}")
        if d == inicio:
            avisos.append(f"fluxo entrando no início ({o} → {d})")
        if o == fim:
            avisos.append(f"fluxo saindo do fim ({o} → {d})")
        saidas[o] = saidas.get(o, 0.0) + p
    for n in sorted(nos - {fim}):
        s = saidas.get(n, 0.0)
        if s == 0:
            avisos.append(f"'{n}' não tem saída (beco sem saída)")
        elif s < 1 - 1e-6:
            avisos.append(f"saídas de '{n}' somam {s:.3f} < 1: casos se perdem")
        elif s > 1 + 1e-6:
            avisos.append(f"saídas de '{n}' somam {s:.3f} > 1: válido só se for divisão paralela (AND)")
    # alcançabilidade
    adj: dict[str, list[str]] = {}
    for o, d, p in lista:
        if p > 0:
            adj.setdefault(o, []).append(d)
    vistos, pilha = {inicio}, [inicio]
    while pilha:
        for d in adj.get(pilha.pop(), []):
            if d not in vistos:
                vistos.add(d)
                pilha.append(d)
    for n in sorted(nos - vistos):
        avisos.append(f"'{n}' não é alcançável a partir do início")
    return avisos


def razao_visitas(transicoes, inicio: str = INICIO, fim: str = FIM) -> dict[str, float]:
    """Número esperado de execuções de cada atividade por caso que entra no processo (razão de visitas).

    O fluxo é tratado como cadeia de Markov absorvente: v = s + Pᵀ·v ⇒ (I − Pᵀ)·v = s, em que P é a matriz de
    probabilidades entre atividades e s o vetor que sai do início. Generaliza as fórmulas de bloco de
    DUMAS et al. (2018): XOR (Σ pᵢTᵢ), retrabalho (T/(1 − r)) e laços aninhados. Em divisão paralela (AND),
    cada ramo recebe probabilidade 1 e o resultado continua sendo o número esperado de execuções.

    ``transicoes``: DataFrame com colunas origem; destino; probabilidade, ou lista de tuplas (origem, destino, p).
    """
    lista = _transicoes_para_lista(transicoes)
    atividades = sorted(({o for o, _, _ in lista} | {d for _, d, _ in lista}) - {inicio, fim})
    idx = {a: i for i, a in enumerate(atividades)}
    n = len(atividades)
    P = np.zeros((n, n))
    s = np.zeros(n)
    for o, d, p in lista:
        if d in (fim, inicio):
            continue
        if o == inicio:
            s[idx[d]] += p
        elif o != fim:
            P[idx[o], idx[d]] += p
    if n and max(abs(np.linalg.eigvals(P))) >= 1 - 1e-12:
        raise ValueError("o fluxo tem laço do qual os casos nunca saem (raio espectral ≥ 1); revise as probabilidades")
    v = np.linalg.solve(np.eye(n) - P.T, s) if n else np.zeros(0)
    return {a: float(v[idx[a]]) for a in atividades}


def carga_por_atividade(visitas: Mapping[str, float], tempos_min: Mapping[str, float],
                        demanda_processo: float) -> pd.DataFrame:
    """Carga por atividade a partir das razões de visita.

    demanda da atividade = demanda do processo × visitas; carga (h) = demanda da atividade × tempo (min) / 60.
    A soma de visitas × tempo é o esforço unitário da entrega (Fase 6 do DFT), já com desvios e retrabalho.
    """
    linhas = []
    for a, v in visitas.items():
        t = float(tempos_min[a])
        linhas.append({"atividade": a, "visitas": v, "tempo_min": t, "demanda": demanda_processo * v,
                       "carga_h": carga_horas(demanda_processo * v, t)})
    return pd.DataFrame(linhas)


def esforco_unitario(visitas: Mapping[str, float], tempos_min: Mapping[str, float]) -> float:
    """Esforço por caso (horas) = Σ visitasᵢ × tempoᵢ / 60 — o Eᵢ que o fluxo entrega à Fase 6."""
    return float(sum(v * float(tempos_min[a]) for a, v in visitas.items()) / 60.0)


# ---------------------------------------------------------------------------------------------------------------
# 6. Análise tabular (o que o CLI usa)
# ---------------------------------------------------------------------------------------------------------------

_ALIASES = {
    "recurso": "perfil", "recurso_perfil": "perfil", "recurso/perfil": "perfil", "cargo": "perfil",
    "tempo": "tempo_min", "tempo_medio_min": "tempo_min", "tempo_medio": "tempo_min", "tempo_min_execucao": "tempo_min",
    "n_pessoas": "pessoas", "pessoas_alocadas": "pessoas", "no_pessoas": "pessoas",
    "horas": "horas_disponiveis", "horas_disponiveis_pessoa": "horas_disponiveis", "horas_periodo": "horas_disponiveis",
    "demanda_periodo": "demanda", "prob": "probabilidade", "p": "probabilidade",
    "producao": "producao_realizada", "estoque": "wip", "em_andamento": "wip",
}


def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c: str) -> str:
        c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode().strip().lower()
        c = c.replace("º", "").replace(".", "").replace(" ", "_")
        return _ALIASES.get(c, c)
    return df.rename(columns={c: norm(c) for c in df.columns})


def ler_atividades(caminho) -> pd.DataFrame:
    """Lê o CSV de atividades (``;`` e vírgula decimal). Colunas: atividade; perfil; tempo_min; pessoas;
    horas_disponiveis; demanda — opcionais: id; wip; producao_realizada."""
    df = _normalizar_colunas(ler_csv(caminho))
    faltam = {"atividade", "tempo_min", "pessoas", "horas_disponiveis"} - set(df.columns)
    if faltam:
        raise ValueError(f"colunas obrigatórias ausentes: {sorted(faltam)}")
    if "perfil" not in df:
        df["perfil"] = "(não informado)"
    if "id" not in df:
        df["id"] = df["atividade"]
    df["id"] = df["id"].astype(str).str.strip()
    for col in ("tempo_min", "pessoas", "horas_disponiveis", "demanda", "wip", "producao_realizada"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    return df


def ler_fluxo(caminho) -> pd.DataFrame:
    """Lê o CSV de transições: origem; destino; probabilidade (use INICIO e FIM como eventos)."""
    df = _normalizar_colunas(ler_csv(caminho))
    faltam = {"origem", "destino", "probabilidade"} - set(df.columns)
    if faltam:
        raise ValueError(f"colunas obrigatórias ausentes no fluxo: {sorted(faltam)}")
    return df


def analisar_capacidade(atividades: pd.DataFrame, horas_totais: bool = False, utilizacao_alvo: float = 1.0,
                        fluxo: pd.DataFrame | None = None, demanda_processo: float | None = None) -> pd.DataFrame:
    """Tabela de capacidade por etapa.

    Se ``fluxo`` for dado, a demanda de cada atividade = ``demanda_processo`` × razão de visitas (a coluna
    ``demanda`` do CSV é ignorada). Saída: capacidade, carga_h, horas, utilizacao, tempo_ciclo_min, takt_min,
    pessoas_necessarias, situacao, gargalo (True na etapa de maior utilização) e, se houver WIP, vazao e lead_time.
    """
    df = _normalizar_colunas(atividades.copy())
    if "id" not in df:
        df["id"] = df["atividade"]
    if "perfil" not in df:
        df["perfil"] = "(não informado)"
    if fluxo is not None:
        if demanda_processo is None:
            raise ValueError("com fluxo, informe a demanda do processo (casos que entram no período)")
        v = razao_visitas(fluxo)
        faltam = set(v) - set(df["id"])
        if faltam:
            raise ValueError(f"atividades do fluxo sem linha no CSV de atividades: {sorted(faltam)}")
        df["visitas"] = df["id"].map(v).fillna(0.0)
        df["demanda"] = demanda_processo * df["visitas"]
    elif "demanda" not in df:
        raise ValueError("informe a coluna demanda ou um fluxo com a demanda do processo")
    horas = df["horas_disponiveis"] if horas_totais else df["pessoas"] * df["horas_disponiveis"]
    horas_pessoa = df["horas_disponiveis"] / df["pessoas"].where(df["pessoas"] > 0) if horas_totais \
        else df["horas_disponiveis"]
    df["horas"] = horas
    df["capacidade"] = horas * 60.0 / df["tempo_min"]
    df["carga_h"] = df["demanda"] * df["tempo_min"] / 60.0
    df["utilizacao"] = df["carga_h"] / horas.where(horas > 0)
    df["tempo_ciclo_min"] = df["tempo_min"] / df["pessoas"].where(df["pessoas"] > 0)
    df["takt_min"] = horas_pessoa * 60.0 / df["demanda"].where(df["demanda"] > 0)
    df["pessoas_necessarias"] = df["carga_h"] / (horas_pessoa * utilizacao_alvo)
    df["situacao"] = np.select([df["utilizacao"] > 1, df["utilizacao"] >= LIMITE_ALERTA],
                               ["acima da capacidade", "zona de fila"], "folga")
    df["gargalo"] = False
    if df["utilizacao"].notna().any():
        df.loc[df["utilizacao"].idxmax(), "gargalo"] = True
    if "wip" in df:
        df["vazao"] = np.minimum(df["demanda"], df["capacidade"])
        df["lead_time"] = df["wip"] / df["vazao"].where(df["vazao"] > 0)
    return df


def resumo_processo(tab: pd.DataFrame, demanda_processo: float | None = None) -> dict:
    """Indicadores do processo: gargalo, capacidade do processo, vazão, WIP, lead time (Little) e esforço/caso.

    capacidade do processo = demanda do processo / ρ_max (a etapa de maior utilização limita o todo).
    Sem fluxo, a demanda do processo é a da primeira atividade (porta de entrada).
    """
    d = float(demanda_processo if demanda_processo is not None else tab["demanda"].iloc[0])
    g = tab.loc[tab["gargalo"]].iloc[0]
    rho_max = float(g["utilizacao"])
    cap_proc = d / rho_max if rho_max > 0 else math.inf
    vazao = min(d, cap_proc)
    res = {"demanda_processo": d, "gargalo": str(g["atividade"]), "utilizacao_gargalo": rho_max,
           "capacidade_processo": cap_proc, "vazao": vazao, "deficit_por_periodo": max(0.0, d - cap_proc),
           "carga_total_h": float(tab["carga_h"].sum()),
           "esforco_por_caso_h": float(tab["carga_h"].sum() / d) if d > 0 else math.nan}
    if "wip" in tab:
        wip = float(tab["wip"].sum())
        res.update({"wip_total": wip, "lead_time": little(wip=wip, vazao=vazao)})
    return res


def por_perfil(tab: pd.DataFrame) -> pd.DataFrame:
    """Agrega carga e horas por perfil (tipo de recurso — ponte com a Fase 5 do DFT)."""
    g = tab.groupby("perfil", sort=False).agg(carga_h=("carga_h", "sum"), horas=("horas", "sum"),
                                              pessoas=("pessoas", "sum"),
                                              pessoas_necessarias=("pessoas_necessarias", "sum")).reset_index()
    g["utilizacao"] = g["carga_h"] / g["horas"]
    return g


def relatorio(tab: pd.DataFrame, resumo: dict, periodo: str = "período", dias_periodo: float | None = None) -> str:
    """Relatório legível em português."""
    pct = lambda x: f"{br(100 * x, 1)}%" if pd.notna(x) and math.isfinite(x) else "—"
    linhas = [f"ANÁLISE DE CAPACIDADE — unidades por {periodo}", "=" * 96,
              f"{'Etapa':<34}{'Perfil':<20}{'Demanda':>9}{'Capac.':>9}{'Carga h':>9}{'Util.':>8}"
              f"{'Pess.nec.':>10}  Situação"]
    for _, r in tab.iterrows():
        marca = " ◄ GARGALO" if r["gargalo"] else ""
        linhas.append(f"{str(r['atividade'])[:33]:<34}{str(r['perfil'])[:19]:<20}{br(r['demanda'], 1):>9}"
                      f"{br(r['capacidade'], 1):>9}{br(r['carga_h'], 1):>9}{pct(r['utilizacao']):>8}"
                      f"{br(r['pessoas_necessarias'], 2):>10}  {r['situacao']}{marca}")
    linhas += ["-" * 96, "Takt × tempo de ciclo (min/unidade, sobre o tempo disponível por pessoa):"]
    for _, r in tab.iterrows():
        ok = "acompanha" if r["tempo_ciclo_min"] <= r["takt_min"] else "NÃO acompanha"
        linhas.append(f"  {str(r['atividade'])[:40]:<42} ciclo {br(r['tempo_ciclo_min'], 1):>7}  takt "
                      f"{br(r['takt_min'], 1):>7}  → {ok}")
    pp = por_perfil(tab)
    linhas += ["-" * 96, "Por perfil (tipo de recurso):"]
    for _, r in pp.iterrows():
        linhas.append(f"  {str(r['perfil'])[:30]:<32} carga {br(r['carga_h'], 1):>8} h  horas {br(r['horas'], 1):>8} h"
                      f"  util. {pct(r['utilizacao']):>7}  pessoas {br(r['pessoas'], 0)} → necessárias "
                      f"{br(r['pessoas_necessarias'], 2)}")
    linhas += ["-" * 96, "Processo:",
               f"  Demanda do processo: {br(resumo['demanda_processo'], 1)} casos/{periodo}",
               f"  Gargalo: {resumo['gargalo']} (utilização {pct(resumo['utilizacao_gargalo'])})",
               f"  Capacidade do processo: {br(resumo['capacidade_processo'], 1)} casos/{periodo}"
               f"  → vazão sustentável {br(resumo['vazao'], 1)}",
               f"  Carga total: {br(resumo['carga_total_h'], 1)} h/{periodo}; esforço por caso "
               f"(com desvios e retrabalho): {br(resumo['esforco_por_caso_h'], 3)} h"]
    if resumo["deficit_por_periodo"] > 0:
        linhas.append(f"  ATENÇÃO: demanda acima da capacidade — o estoque cresce cerca de "
                      f"{br(resumo['deficit_por_periodo'], 1)} casos por {periodo} (Little deixa de valer em regime).")
    if "lead_time" in resumo:
        lt = resumo["lead_time"]
        extra = f" ≈ {br(lt * dias_periodo, 1)} dias" if dias_periodo else ""
        linhas.append(f"  Lei de Little: WIP {br(resumo['wip_total'], 0)} ÷ vazão {br(resumo['vazao'], 1)} = lead time "
                      f"{br(lt, 2)} × {periodo}{extra}")
    linhas.append(f"Legenda: folga < {pct(LIMITE_ALERTA)} ≤ zona de fila ≤ 100% < acima da capacidade.")
    return "\n".join(linhas)


def grafico_capacidade(tab: pd.DataFrame, caminho, titulo: str = "Capacidade × demanda por etapa") -> None:
    """Barras de capacidade e demanda por etapa (painel 1) e utilização com faixas de 85% e 100% (painel 2)."""
    plt = estilo_graficos()
    nomes = [str(a) if len(str(a)) <= 36 else str(a)[:35] + "…" for a in tab["atividade"]]
    y = np.arange(len(nomes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.55 * len(nomes) + 2.2), sharey=True,
                                   gridspec_kw={"width_ratios": [3, 2]})
    ax1.barh(y + 0.2, tab["capacidade"], 0.4, color=PALETA["azul"], label="Capacidade efetiva")
    ax1.barh(y - 0.2, tab["demanda"], 0.4, color=PALETA["marinho"], label="Demanda")
    ax1.set_yticks(y, nomes)
    ax1.invert_yaxis()
    ax1.set_xlabel("unidades por período")
    ax1.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=13)
    cores = [PALETA["marinho"] if g else (PALETA["cinza"] if u < LIMITE_ALERTA else PALETA["azul_escuro"])
             for u, g in zip(tab["utilizacao"], tab["gargalo"])]
    ax2.barh(y, 100 * tab["utilizacao"], 0.55, color=cores)
    ax2.axvline(100 * LIMITE_ALERTA, color=PALETA["verde_escuro"], ls="--", lw=1)
    ax2.axvline(100, color=PALETA["marinho"], lw=1.2)
    for yi, u, g in zip(y, tab["utilizacao"], tab["gargalo"]):
        ax2.text(100 * u + 1.5, yi, f"{br(100 * u, 0)}%" + (" gargalo" if g else ""), va="center", fontsize=9)
    ax2.set_xlabel("utilização (%)")
    ax2.set_xlim(0, max(115, 100 * float(tab["utilizacao"].max()) + 25))
    ax2.set_title("Utilização (tracejado = 85%; contínuo = 100%)", loc="left", fontsize=11)
    formatar_eixos_br(fig)
    fig.savefig(caminho)
    plt.close(fig)


def grafico_curva_fila(caminho, ca2: float = 1.0, ce2: float = 1.0, servidores: Sequence[int] = (1, 2, 5)) -> None:
    """Curva de espera relativa (Wq/te) × utilização pela VUT: o "taco de hóquei" das filas."""
    plt = estilo_graficos()
    u = np.linspace(0.05, 0.97, 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in servidores:
        ax.plot(100 * u, [kingman_espera(x, ca2, ce2, 1.0, c) for x in u], label=f"{c} servidor(es)")
    ax.axvline(100 * LIMITE_ALERTA, color=PALETA["verde_escuro"], ls="--", lw=1)
    ax.set_ylim(0, 20)
    ax.set_xlabel("utilização (%)")
    ax.set_ylabel("espera em fila ÷ tempo de serviço")
    ax.set_title("Quanto mais perto de 100%, mais a fila explode", loc="left", fontsize=12)
    ax.legend(frameon=False)
    formatar_eixos_br(fig)
    fig.savefig(caminho)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# 7. Linha de comando
# ---------------------------------------------------------------------------------------------------------------

def cli(argv: list[str]) -> int:
    """``dft capacidade atividades.csv [--fluxo fluxo.csv --demanda N] [--grafico saida.png]``."""
    ap = argparse.ArgumentParser(prog="dft capacidade",
                                 description="Capacidade por etapa, utilização, gargalo, takt e lei de Little.")
    ap.add_argument("arquivo", help="CSV de atividades (atividade; perfil; tempo_min; pessoas; horas_disponiveis; "
                                    "demanda [; id; wip])")
    ap.add_argument("--fluxo", help="CSV de transições (origem; destino; probabilidade) com INICIO e FIM")
    ap.add_argument("--demanda", type=float, help="casos que entram no processo no período (obrigatório com --fluxo)")
    ap.add_argument("--horas-totais", action="store_true",
                    help="a coluna horas_disponiveis já é o total da etapa (não multiplica por pessoas)")
    ap.add_argument("--alvo", type=float, default=1.0,
                    help="utilização-alvo para pessoas necessárias (padrão 1,0 = Q do DFT)")
    ap.add_argument("--periodo", default="mês", help="rótulo do período (padrão: mês)")
    ap.add_argument("--dias-periodo", type=float, help="dias úteis por período, para exprimir o lead time em dias")
    ap.add_argument("--grafico", help="salva o gráfico capacidade × demanda neste arquivo PNG")
    a = ap.parse_args(argv)
    try:
        ativ = ler_atividades(a.arquivo)
        fluxo = ler_fluxo(a.fluxo) if a.fluxo else None
        if fluxo is not None:
            avisos = validar_fluxo(fluxo)
            for av in avisos:
                print(f"Aviso no fluxo: {av}")
            if a.demanda is None:
                raise ValueError("com --fluxo, informe --demanda (casos que entram no período)")
        tab = analisar_capacidade(ativ, horas_totais=a.horas_totais, utilizacao_alvo=a.alvo, fluxo=fluxo,
                                  demanda_processo=a.demanda)
        res = resumo_processo(tab, a.demanda)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"Erro: {e}")
        return 2
    if fluxo is not None:
        print("Razão de visitas (execuções esperadas por caso):")
        for _, r in tab.iterrows():
            print(f"  {r['id']:<8} {str(r['atividade'])[:50]:<52} {br(r['visitas'], 3)}")
        print()
    print(relatorio(tab, res, a.periodo, a.dias_periodo))
    if a.grafico:
        grafico_capacidade(tab, a.grafico)
        print(f"\nGráfico salvo em {a.grafico}")
    return 0

