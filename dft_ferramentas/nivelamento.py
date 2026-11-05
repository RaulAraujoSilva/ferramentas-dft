"""3. Nivelamento de recursos e Teoria das Restrições (TOC).

Duas ferramentas complementares para a passagem "carga de trabalho → quadro de pessoal":

Parte A — **suavização e nivelamento de recursos** (PMI, PMBOK Guide, 6. ed., §6.5.2.3 "Resource Optimization"):
    a partir de entregas/atividades com duração (semanas), precedência, recurso (perfil), carga (servidores/semana)
    e janela (semana mais cedo / semana limite), calcula o cronograma pelo método do caminho crítico (CPM), o perfil
    de demanda por recurso (histograma), aplica a **suavização** (desloca atividades dentro da folga, sem mudar o
    prazo, minimizando a soma dos quadrados da demanda — método de Burgess e Killebrew, 1962) e o **nivelamento**
    (respeita a capacidade de cada recurso, atrasando o necessário — esquema serial de geração de cronograma com
    regra de prioridade de menor folga; Davis e Patterson, 1975; Kolisch e Hartmann, 2006). Também calcula o
    pulmão de projeto da corrente crítica (Goldratt, 1997; corte de 50 % ou raiz da soma dos quadrados).

Parte B — **Teoria das Restrições** (Goldratt e Cox, *A Meta*; Goldratt, 1990): dado um fluxo de etapas com
    servidores, tempo por caso, fração de roteamento e demanda, aplica os 5 passos de focalização — identificar a
    restrição, explorá-la, subordinar o resto a ela (tambor-pulmão-corda), elevá-la e reiniciar — e mostra o efeito
    de +1 servidor na restrição × na não-restrição, inclusive pela contabilidade de ganhos (T, I, OE), o
    dimensionamento das etapas pela restrição (capacidade protetiva) e o tamanho do pulmão.

Convenções de tempo (Parte A): semanas inteiras, numeradas a partir de 1. Internamente, uma atividade que "começa
em s" (s = 0, 1, …) ocupa os períodos s, s+1, …, s+d−1; no relatório, início = s+1 e término = s+d.
Unidade de carga: servidores (equivalentes de tempo integral) por semana. Área do histograma = servidor-semanas.

Todas as funções são puras (sem leitura/escrita escondida); a CLI (`cli`) faz a leitura do CSV e a impressão.
Dados de exemplo sempre fictícios.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._comum import PALETA, br, estilo_graficos, ler_csv

__all__ = [
    # Parte A
    "Atividade", "atividades_de_dataframe", "ordem_topologica", "cpm", "caminho_critico", "perfil_recursos",
    "indicadores_perfil", "soma_quadrados", "suavizar", "nivelar", "tabela_cronograma", "pulmao_projeto",
    # Parte B
    "Etapa", "etapas_de_dataframe", "analise_fluxo", "vazao_sistema", "explorar", "realocar", "cenarios_elevar",
    "cenarios_realocar", "dimensionar_pela_restricao", "servidores_necessarios", "pulmao_tambor",
    "efeito_subordinar", "cli",
]


# =====================================================================================================================
# PARTE A — suavização e nivelamento de recursos
# =====================================================================================================================
@dataclass(frozen=True)
class Atividade:
    """Atividade/entrega do cronograma.

    id: código curto e único; nome: descrição; duracao: semanas inteiras (≥ 1); recurso: perfil que executa
    (ex.: "Fiscal"); carga: servidores/semana enquanto a atividade dura; predecessoras: ids que precisam terminar
    antes (relação término-início); inicio_min: semana mais cedo permitida (1 = primeira); fim_max: semana-limite
    de término (inclusiva) ou None.
    """
    id: str
    nome: str
    duracao: int
    recurso: str
    carga: float
    predecessoras: tuple[str, ...] = ()
    inicio_min: int = 1
    fim_max: int | None = None

    @property
    def liberacao(self) -> int:
        """Primeiro período (0-based) em que pode começar."""
        return max(0, int(self.inicio_min) - 1)


def _vazio(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def atividades_de_dataframe(df: pd.DataFrame) -> list[Atividade]:
    """Converte um DataFrame (colunas `id`, `entrega` ou `atividade`, `duracao`, `predecessoras`, `recurso`, `carga`,
    opcionais `inicio_min`, `fim_max`) em lista de `Atividade`. Predecessoras separadas por vírgula ou espaço."""
    cols = {c.lower().strip(): c for c in df.columns}
    nome_col = cols.get("entrega") or cols.get("atividade") or cols.get("nome") or cols["id"]
    out = []
    for _, r in df.iterrows():
        pred = r.get(cols.get("predecessoras", "predecessoras"), None)
        preds = () if _vazio(pred) else tuple(p for p in str(pred).replace(",", " ").split() if p)
        imin = r.get(cols.get("inicio_min", "inicio_min"), None)
        fmax = r.get(cols.get("fim_max", "fim_max"), None)
        dur = float(r[cols["duracao"]])
        if dur != int(dur) or dur < 1:
            raise ValueError(f"Duração da atividade {r[cols['id']]} deve ser inteira e ≥ 1 semana (veio {dur}).")
        out.append(Atividade(id=str(r[cols["id"]]).strip(), nome=str(r[nome_col]).strip(), duracao=int(dur),
                             recurso=str(r[cols["recurso"]]).strip(), carga=float(r[cols["carga"]]),
                             predecessoras=preds, inicio_min=1 if _vazio(imin) else int(imin),
                             fim_max=None if _vazio(fmax) else int(fmax)))
    ids = [a.id for a in out]
    if len(set(ids)) != len(ids):
        raise ValueError("Há ids de atividade repetidos.")
    return out


def _mapa(ativs: Sequence[Atividade]) -> dict[str, Atividade]:
    return {a.id: a for a in ativs}


def _sucessoras(ativs: Sequence[Atividade]) -> dict[str, list[str]]:
    suc: dict[str, list[str]] = {a.id: [] for a in ativs}
    for a in ativs:
        for p in a.predecessoras:
            if p not in suc:
                raise ValueError(f"Predecessora desconhecida '{p}' na atividade {a.id}.")
            suc[p].append(a.id)
    return suc


def ordem_topologica(ativs: Sequence[Atividade]) -> list[str]:
    """Ordem topológica (algoritmo de Kahn, estável pela ordem de entrada). Levanta ValueError se houver ciclo."""
    suc = _sucessoras(ativs)
    grau = {a.id: len(a.predecessoras) for a in ativs}
    fila = [a.id for a in ativs if grau[a.id] == 0]
    ordem: list[str] = []
    while fila:
        x = fila.pop(0)
        ordem.append(x)
        for s in suc[x]:
            grau[s] -= 1
            if grau[s] == 0:
                fila.append(s)
    if len(ordem) != len(ativs):
        raise ValueError("A rede de precedências tem ciclo.")
    return ordem


def cpm(ativs: Sequence[Atividade], prazo: int | None = None) -> pd.DataFrame:
    """Método do caminho crítico com janelas (liberação e prazo por atividade).

    Retorna DataFrame indexado por id com ES, EF, LS, LF (períodos 0-based; EF/LF exclusivos), folga_total,
    folga_livre e critica (folga_total == 0). `prazo` (semanas) é o término exigido do projeto; se None, usa o
    término mais cedo calculado (folga zero no caminho crítico). Folga negativa indica janela inviável.
    Ref.: PMI (2017), §6.5.2.2; Kelley e Walker (1959).
    """
    m = _mapa(ativs)
    ordem = ordem_topologica(ativs)
    suc = _sucessoras(ativs)
    es, ef = {}, {}
    for i in ordem:
        a = m[i]
        es[i] = max([a.liberacao] + [ef[p] for p in a.predecessoras])
        ef[i] = es[i] + a.duracao
    fim = max(ef.values()) if prazo is None else int(prazo)
    ls, lf = {}, {}
    for i in reversed(ordem):
        a = m[i]
        lim = [fim] + ([a.fim_max] if a.fim_max is not None else []) + [ls[s] for s in suc[i]]
        lf[i] = min(lim)
        ls[i] = lf[i] - a.duracao
    linhas = []
    for i in ordem:
        livre = (min(es[s] for s in suc[i]) if suc[i] else fim) - ef[i]
        linhas.append({"id": i, "ES": es[i], "EF": ef[i], "LS": ls[i], "LF": lf[i], "folga_total": ls[i] - es[i],
                       "folga_livre": livre, "critica": ls[i] - es[i] == 0})
    return pd.DataFrame(linhas).set_index("id")


def caminho_critico(ativs: Sequence[Atividade], tabela_cpm: pd.DataFrame | None = None) -> list[str]:
    """Uma cadeia de atividades críticas, do início ao fim (segue predecessoras com EF == ES do sucessor)."""
    t = cpm(ativs) if tabela_cpm is None else tabela_cpm
    m = _mapa(ativs)
    atual = t["EF"].idxmax()
    cadeia = [atual]
    while True:
        cand = [p for p in m[atual].predecessoras if t.loc[p, "EF"] == t.loc[atual, "ES"] and t.loc[p, "critica"]]
        if not cand:
            break
        atual = cand[0]
        cadeia.append(atual)
    return cadeia[::-1]


def perfil_recursos(ativs: Sequence[Atividade], inicios: Mapping[str, int], horizonte: int | None = None
                    ) -> pd.DataFrame:
    """Histograma de recursos: demanda (servidores) por semana e por recurso, dado o início (0-based) de cada
    atividade. Índice = semana (1, 2, …); colunas = recursos."""
    fim = max(inicios[a.id] + a.duracao for a in ativs)
    T = max(fim, horizonte or 0)
    recursos = sorted({a.recurso for a in ativs})
    mat = {r: np.zeros(T) for r in recursos}
    for a in ativs:
        s = inicios[a.id]
        mat[a.recurso][s:s + a.duracao] += a.carga
    df = pd.DataFrame(mat, index=pd.RangeIndex(1, T + 1, name="semana"))
    return df


def soma_quadrados(perfil: pd.DataFrame | np.ndarray) -> float:
    """Σ r(t)² — métrica de Burgess: para área fixa, é mínima quando o perfil é o mais plano possível."""
    return float(np.square(np.asarray(perfil, dtype=float)).sum())


def indicadores_perfil(perfil: pd.DataFrame, capacidades: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Indicadores por recurso: área (servidor-semanas), pico, média no horizonte, razão pico/média, soma dos
    quadrados, semanas acima da capacidade e excesso máximo sobre a capacidade."""
    capacidades = capacidades or {}
    linhas = []
    for r in perfil.columns:
        x = perfil[r].to_numpy(dtype=float)
        cap = capacidades.get(r, math.inf)
        media = x.sum() / len(x)
        linhas.append({"recurso": r, "area": x.sum(), "pico": x.max(), "media": media,
                       "pico_sobre_media": x.max() / media if media else float("nan"), "soma_quadrados": soma_quadrados(x),
                       "capacidade": cap, "semanas_acima": int((x > cap + 1e-9).sum()),
                       "excesso_max": max(0.0, float(x.max() - cap)) if math.isfinite(cap) else 0.0})
    return pd.DataFrame(linhas).set_index("recurso")


def _janela(a: Atividade, inicios: Mapping[str, int], suc: Mapping[str, list[str]], m: Mapping[str, Atividade],
            fim: int) -> tuple[int, int]:
    lo = max([a.liberacao] + [inicios[p] + m[p].duracao for p in a.predecessoras])
    lim_fim = min([fim] + ([a.fim_max] if a.fim_max is not None else []) + [inicios[s] for s in suc[a.id]])
    return lo, lim_fim - a.duracao


def suavizar(ativs: Sequence[Atividade], prazo: int | None = None, max_passadas: int = 50) -> dict[str, int]:
    """Suavização de recursos (resource smoothing) pelo método de Burgess e Killebrew (1962).

    Mantém o prazo do projeto (o término do CPM, ou `prazo`) e as janelas; percorre as atividades da última para a
    primeira (ordem topológica reversa) e desloca cada uma, dentro da folga que as vizinhas deixam, para a posição
    que minimiza a soma dos quadrados do perfil do seu recurso (empate: menor pico; depois, o início mais cedo).
    Só muda de posição se a soma dos quadrados diminuir estritamente; repete passadas até não haver melhora.
    Heurística: dá bom resultado, não garante o ótimo global (Neumann e Zimmermann, 2000).
    Retorna {id: início 0-based}.
    """
    m = _mapa(ativs)
    suc = _sucessoras(ativs)
    base = cpm(ativs, prazo)
    if (base["folga_total"] < 0).any():
        raise ValueError("Janelas inviáveis (folga negativa): ajuste prazos antes de suavizar.")
    fim = int(base["EF"].max()) if prazo is None else int(prazo)
    inicios = {i: int(base.loc[i, "ES"]) for i in base.index}
    ordem = ordem_topologica(ativs)
    perfis = {r: np.zeros(fim) for r in {a.recurso for a in ativs}}
    for a in ativs:
        perfis[a.recurso][inicios[a.id]:inicios[a.id] + a.duracao] += a.carga
    for _ in range(max_passadas):
        mudou = False
        for i in reversed(ordem):
            a = m[i]
            lo, hi = _janela(a, inicios, suc, m, fim)
            if hi <= lo:
                continue
            p = perfis[a.recurso]
            s0 = inicios[i]
            p[s0:s0 + a.duracao] -= a.carga
            melhor = None
            for t in range(lo, hi + 1):
                q = p.copy()
                q[t:t + a.duracao] += a.carga
                chave = (round(float(np.square(q).sum()), 9), round(float(q.max()), 9), t)
                if melhor is None or chave < melhor[0]:
                    melhor = (chave, t)
            q0 = p.copy()
            q0[s0:s0 + a.duracao] += a.carga
            ss_atual = round(float(np.square(q0).sum()), 9)
            t_novo = melhor[1] if melhor[0][0] < ss_atual else s0
            p[t_novo:t_novo + a.duracao] += a.carga
            if t_novo != s0:
                inicios[i] = t_novo
                mudou = True
        if not mudou:
            break
    return inicios


_REGRAS = {
    "folga": lambda t, i: (t.loc[i, "folga_total"], t.loc[i, "LS"], t.loc[i, "ES"]),   # MINSLK
    "lft": lambda t, i: (t.loc[i, "LF"], t.loc[i, "folga_total"], t.loc[i, "ES"]),     # menor término tardio
    "lst": lambda t, i: (t.loc[i, "LS"], t.loc[i, "folga_total"], t.loc[i, "ES"]),     # menor início tardio
}


def nivelar(ativs: Sequence[Atividade], capacidades: Mapping[str, float], regra: str = "folga") -> dict[str, int]:
    """Nivelamento de recursos (resource leveling) com limite de capacidade.

    Esquema serial de geração de cronograma (serial SGS): a cada passo, entre as atividades elegíveis (todas as
    predecessoras já programadas), escolhe a de maior prioridade e a coloca no primeiro período ≥ liberação e
    ≥ término das predecessoras em que a carga cabe na capacidade restante do recurso durante toda a duração.
    Regras de prioridade: "folga" (menor folga total do CPM — MINSLK; desempate pelo menor LS), "lft" (menor
    término tardio) ou "lst" (menor início tardio). Recursos sem capacidade informada são ilimitados.
    O prazo pode aumentar — é a diferença essencial para a suavização (PMI, 2017, §6.5.2.3).
    Ref.: Davis e Patterson (1975); Kolisch e Hartmann (2006). Retorna {id: início 0-based}.
    """
    if regra not in _REGRAS:
        raise ValueError(f"Regra desconhecida: {regra}. Use {sorted(_REGRAS)}.")
    m = _mapa(ativs)
    for a in ativs:
        cap = capacidades.get(a.recurso, math.inf)
        if a.carga > cap + 1e-9:
            raise ValueError(f"A atividade {a.id} pede {br(a.carga, 1)} {a.recurso}(s)/semana, acima da capacidade "
                             f"{br(cap, 1)}: divida a atividade ou eleve a capacidade.")
    t = cpm(ativs)
    prio = {i: _REGRAS[regra](t, i) + (i,) for i in t.index}
    uso: dict[str, list[float]] = {r: [] for r in {a.recurso for a in ativs}}
    inicios: dict[str, int] = {}
    pendentes = set(m)
    while pendentes:
        eleg = [i for i in pendentes if all(p in inicios for p in m[i].predecessoras)]
        i = min(eleg, key=lambda x: prio[x])
        a = m[i]
        cap = capacidades.get(a.recurso, math.inf)
        s = max([a.liberacao] + [inicios[p] + m[p].duracao for p in a.predecessoras])
        u = uso[a.recurso]
        while True:
            if len(u) < s + a.duracao:
                u.extend([0.0] * (s + a.duracao - len(u)))
            if all(u[k] + a.carga <= cap + 1e-9 for k in range(s, s + a.duracao)):
                break
            s += 1
        for k in range(s, s + a.duracao):
            u[k] += a.carga
        inicios[i] = s
        pendentes.remove(i)
    return inicios


def tabela_cronograma(ativs: Sequence[Atividade], **cronogramas: Mapping[str, int]) -> pd.DataFrame:
    """Tabela legível: para cada cronograma nomeado (ex.: cedo=…, suavizado=…), início e término em semanas
    (1-based) e, quando houver `fim_max`, a indicação de atraso em relação ao limite."""
    linhas = []
    for a in ativs:
        lin = {"id": a.id, "entrega": a.nome, "recurso": a.recurso, "carga": a.carga, "duracao": a.duracao}
        for nome, ini in cronogramas.items():
            lin[f"ini_{nome}"] = ini[a.id] + 1
            lin[f"fim_{nome}"] = ini[a.id] + a.duracao
            if a.fim_max is not None:
                lin[f"atraso_{nome}"] = max(0, ini[a.id] + a.duracao - a.fim_max)
        linhas.append(lin)
    return pd.DataFrame(linhas).set_index("id")


def pulmao_projeto(duracoes_seguras: Iterable[float], fator_agressivo: float = 0.5, metodo: str = "corte"
                   ) -> dict[str, float]:
    """Pulmão de projeto da corrente crítica (Critical Chain — Goldratt, 1997).

    Cada estimativa "segura" (com margem embutida) é reduzida a uma estimativa agressiva = fator × segura; a margem
    retirada é agregada num único pulmão no fim da corrente:
      - "corte" (cut and paste, Goldratt/Leach): pulmão = 50 % da corrente agressiva;
      - "rss" (raiz da soma dos quadrados, Newbold, 1998): pulmão = √Σ (segura − agressiva)².
    Herroelen e Leus (2001) mostram que a regra dos 50 % tende a superdimensionar o pulmão em correntes longas.
    Retorna corrente_segura, corrente_agressiva, pulmao, prazo_com_pulmao (mesma unidade das durações).
    """
    seg = np.asarray(list(duracoes_seguras), dtype=float)
    agr = seg * fator_agressivo
    if metodo == "corte":
        pul = 0.5 * agr.sum()
    elif metodo == "rss":
        pul = float(np.sqrt(np.square(seg - agr).sum()))
    else:
        raise ValueError("metodo deve ser 'corte' ou 'rss'.")
    return {"corrente_segura": float(seg.sum()), "corrente_agressiva": float(agr.sum()), "pulmao": float(pul),
            "prazo_com_pulmao": float(agr.sum() + pul)}


# =====================================================================================================================
# PARTE B — Teoria das Restrições
# =====================================================================================================================
@dataclass(frozen=True)
class Etapa:
    """Etapa de um fluxo de atendimento (em série, na ordem da lista).

    nome; servidores (pode ser fracionário: 0,5 = meio servidor dedicado); tempo_h: horas de trabalho por caso
    *que passa pela etapa*; fracao: proporção dos casos do sistema que passam pela etapa (roteamento; 1 = todos);
    grupo: competência/cargo (realocação só dentro do mesmo grupo); horas: horas efetivas por servidor por semana
    (None = usa o valor geral); atravessamento_dias: tempo médio observado de fila + execução na etapa (dias
    úteis); cv: coeficiente de variação desse atravessamento.
    """
    nome: str
    servidores: float
    tempo_h: float
    fracao: float = 1.0
    grupo: str = ""
    horas: float | None = None
    atravessamento_dias: float | None = None
    cv: float | None = None


def etapas_de_dataframe(df: pd.DataFrame) -> list[Etapa]:
    """Converte DataFrame (colunas `etapa`, `servidores`, `tempo_h`; opcionais `fracao`, `grupo`,
    `horas_servidor_semana`, `atravessamento_dias`, `cv_atravessamento`, `ordem`) em lista de `Etapa`."""
    d = df.copy()
    d.columns = [c.lower().strip() for c in d.columns]
    if "ordem" in d.columns:
        d = d.sort_values("ordem")
    def opc(r, c, padrao=None):
        return padrao if c not in r or _vazio(r[c]) else r[c]
    return [Etapa(nome=str(r["etapa"]).strip(), servidores=float(r["servidores"]), tempo_h=float(r["tempo_h"]),
                  fracao=float(opc(r, "fracao", 1.0)), grupo=str(opc(r, "grupo", "")).strip(),
                  horas=None if opc(r, "horas_servidor_semana") is None else float(r["horas_servidor_semana"]),
                  atravessamento_dias=None if opc(r, "atravessamento_dias") is None else float(r["atravessamento_dias"]),
                  cv=None if opc(r, "cv_atravessamento") is None else float(r["cv_atravessamento"]))
            for _, r in d.iterrows()]


def _h(e: Etapa, horas: float) -> float:
    return e.horas if e.horas is not None else horas


def analise_fluxo(etapas: Sequence[Etapa], horas_por_servidor: float, demanda: float | None = None
                  ) -> tuple[pd.DataFrame, dict]:
    """Passo 1 da TOC — identificar a restrição.

    capacidade_i (casos do sistema/semana) = servidores × H / (tempo_h × fracao); carga_i (h/semana) =
    demanda × fracao × tempo_h; utilização_i = carga_i / (servidores × H). A restrição interna é a etapa de menor
    capacidade (= maior utilização). A vazão do sistema é min(demanda, menor capacidade). Se a demanda é menor que
    todas as capacidades, a restrição é externa (o "mercado": a própria demanda).
    Retorna (tabela por etapa, resumo com vazao, restricao, restricao_externa, acumulo_semanal).
    """
    linhas = []
    for e in etapas:
        H = _h(e, horas_por_servidor)
        cap = e.servidores * H / (e.tempo_h * e.fracao) if e.tempo_h * e.fracao > 0 else math.inf
        lin = {"etapa": e.nome, "grupo": e.grupo, "servidores": e.servidores, "tempo_h": e.tempo_h,
               "fracao": e.fracao, "horas_servidor": H, "capacidade": cap}
        if demanda is not None:
            carga = demanda * e.fracao * e.tempo_h
            lin["carga_h"] = carga
            lin["utilizacao"] = carga / (e.servidores * H) if e.servidores > 0 else math.inf
        linhas.append(lin)
    tab = pd.DataFrame(linhas).set_index("etapa")
    restr = str(tab["capacidade"].idxmin())
    cap_min = float(tab["capacidade"].min())
    vazao = cap_min if demanda is None else min(demanda, cap_min)
    resumo = {"restricao": restr, "capacidade_restricao": cap_min, "vazao": vazao,
              "restricao_externa": demanda is not None and demanda <= cap_min,
              "acumulo_semanal": 0.0 if demanda is None else max(0.0, demanda - cap_min)}
    tab["restricao"] = tab.index == restr
    return tab, resumo


def vazao_sistema(etapas: Sequence[Etapa], horas_por_servidor: float, demanda: float | None = None) -> float:
    """Vazão (casos/semana) do fluxo em série: a menor capacidade entre as etapas, limitada pela demanda."""
    return analise_fluxo(etapas, horas_por_servidor, demanda)[1]["vazao"]


def explorar(etapas: Sequence[Etapa], restricao: str, reducao: float, destino: str | None = None) -> list[Etapa]:
    """Passo 2 — explorar a restrição: tirar dela todo trabalho que não precisa ser feito ali.

    Reduz o tempo por caso da etapa `restricao` em `reducao` (fração, ex.: 0,15 = −15 %: retrabalho por processo
    incompleto, tarefas administrativas, esperas). Se `destino` for informado, o trabalho é transferido (não
    eliminado) para essa etapa — ex.: conferência de "kit completo" na análise documental (Ronen, 1992) —,
    acrescentando ao destino as mesmas horas por caso do sistema (ajustadas pelas frações de roteamento).
    """
    if not 0 <= reducao < 1:
        raise ValueError("reducao deve estar em [0, 1).")
    mapa = {e.nome: e for e in etapas}
    if restricao not in mapa or (destino is not None and destino not in mapa):
        raise ValueError("Etapa de restrição ou de destino inexistente.")
    r = mapa[restricao]
    horas_por_caso_sistema = r.tempo_h * reducao * r.fracao
    out = []
    for e in etapas:
        if e.nome == restricao:
            e = replace(e, tempo_h=e.tempo_h * (1 - reducao))
        elif e.nome == destino:
            e = replace(e, tempo_h=e.tempo_h + horas_por_caso_sistema / e.fracao)
        out.append(e)
    return out


def realocar(etapas: Sequence[Etapa], origem: str, destino: str, n: float = 1.0) -> list[Etapa]:
    """Move `n` servidores da etapa `origem` para `destino` (polivalência / remanejamento interno)."""
    out = []
    for e in etapas:
        if e.nome == origem:
            if e.servidores - n < 0:
                raise ValueError("Servidores insuficientes na origem.")
            e = replace(e, servidores=e.servidores - n)
        elif e.nome == destino:
            e = replace(e, servidores=e.servidores + n)
        out.append(e)
    return out


def cenarios_elevar(etapas: Sequence[Etapa], horas_por_servidor: float, demanda: float | None = None,
                    incremento: float = 1.0, custo_servidor: float | None = None,
                    ganho_por_caso: float | None = None) -> pd.DataFrame:
    """Passo 4 — elevar: efeito de +`incremento` servidor(es) em cada etapa, uma de cada vez.

    Mostra a vazão resultante, o ganho de vazão (casos/semana) e a nova restrição. Se `custo_servidor`
    (R$/semana) e `ganho_por_caso` (R$ — ganho no sentido da contabilidade de ganhos: receita menos custos
    totalmente variáveis; no setor público, pode ser a taxa líquida arrecadada ou um valor público convencionado)
    forem informados, acrescenta ΔT (ganho), ΔOE (despesa operacional) e Δ(T − OE) — o critério de Goldratt
    (1990) e Corbett (1998): uma decisão é boa se aumenta T mais do que aumenta OE.
    """
    _, base = analise_fluxo(etapas, horas_por_servidor, demanda)
    linhas = [{"cenario": "atual", "vazao": base["vazao"], "delta_vazao": 0.0, "nova_restricao": base["restricao"]}]
    for e in etapas:
        novo = [replace(x, servidores=x.servidores + incremento) if x.nome == e.nome else x for x in etapas]
        _, r = analise_fluxo(novo, horas_por_servidor, demanda)
        restr = "demanda (externa)" if r["restricao_externa"] else r["restricao"]
        linhas.append({"cenario": f"+{br(incremento, 0)} em {e.nome}", "vazao": r["vazao"],
                       "delta_vazao": r["vazao"] - base["vazao"], "nova_restricao": restr})
    tab = pd.DataFrame(linhas).set_index("cenario")
    if custo_servidor is not None and ganho_por_caso is not None:
        tab["delta_T"] = tab["delta_vazao"] * ganho_por_caso
        tab["delta_OE"] = [0.0] + [custo_servidor * incremento] * len(etapas)
        tab["delta_T_menos_OE"] = tab["delta_T"] - tab["delta_OE"]
    return tab


def cenarios_realocar(etapas: Sequence[Etapa], horas_por_servidor: float, demanda: float | None = None,
                      n: float = 1.0) -> pd.DataFrame:
    """Balanceamento por polivalência: testa mover `n` servidor(es) de cada etapa do MESMO grupo de competência
    para a restrição atual e informa a vazão resultante e a nova restrição (pode surgir uma restrição nova)."""
    _, base = analise_fluxo(etapas, horas_por_servidor, demanda)
    alvo = next(e for e in etapas if e.nome == base["restricao"])
    linhas = []
    for e in etapas:
        if e.nome == alvo.nome or e.grupo != alvo.grupo or e.servidores < n:
            continue
        novo = realocar(etapas, e.nome, alvo.nome, n)
        _, r = analise_fluxo(novo, horas_por_servidor, demanda)
        linhas.append({"origem": e.nome, "destino": alvo.nome, "vazao": r["vazao"],
                       "delta_vazao": r["vazao"] - base["vazao"],
                       "nova_restricao": "demanda (externa)" if r["restricao_externa"] else r["restricao"]})
    return pd.DataFrame(linhas, columns=["origem", "destino", "vazao", "delta_vazao", "nova_restricao"])


def servidores_necessarios(carga_horas: float, horas_por_servidor: float, utilizacao_alvo: float = 1.0) -> float:
    """Q = carga / (H × ρ_alvo) — a fórmula do quadro do curso (Q = Σ D·E / H) com utilização-alvo opcional.
    Ex. canônico (nove passos): 7.770 h/ano ÷ 928,2 h/ano = 8,37 servidores."""
    if horas_por_servidor <= 0 or not 0 < utilizacao_alvo <= 1:
        raise ValueError("horas_por_servidor > 0 e 0 < utilizacao_alvo ≤ 1.")
    return carga_horas / (horas_por_servidor * utilizacao_alvo)


def dimensionar_pela_restricao(etapas: Sequence[Etapa], horas_por_servidor: float, vazao_alvo: float,
                               restricao: str | None = None, alvo_restricao: float = 0.95,
                               alvo_demais: float = 0.85) -> pd.DataFrame:
    """Dimensiona cada etapa para a vazão-alvo do sistema, com a lógica da TOC.

    A restrição (a atual, ou a escolhida em `restricao` — "restrição estratégica", em geral o recurso mais caro ou
    escasso) é dimensionada para operar com utilização alta (`alvo_restricao`); as não-restrições recebem
    capacidade protetiva (`alvo_demais` < `alvo_restricao`), para absorver a variabilidade e não deixar o tambor
    parar. Q_frac = vazao_alvo × fracao × tempo_h / (H × ρ_alvo); Q = teto. Diferença = Q − servidores atuais.
    """
    if restricao is None:
        restricao = analise_fluxo(etapas, horas_por_servidor)[1]["restricao"]
    linhas = []
    for e in etapas:
        H = _h(e, horas_por_servidor)
        rho = alvo_restricao if e.nome == restricao else alvo_demais
        carga = vazao_alvo * e.fracao * e.tempo_h
        q = servidores_necessarios(carga, H, rho)
        qi = math.ceil(q - 1e-9)
        linhas.append({"etapa": e.nome, "papel": "restrição" if e.nome == restricao else "não-restrição",
                       "utilizacao_alvo": rho, "carga_h": carga, "Q_fracionario": q, "Q": qi,
                       "atual": e.servidores, "diferenca": qi - e.servidores,
                       "utilizacao_resultante": carga / (qi * H) if qi else float("nan")})
    return pd.DataFrame(linhas).set_index("etapa")


def pulmao_tambor(etapas: Sequence[Etapa], restricao: str, vazao_semanal: float, fator: float = 0.5,
                  z: float = 1.645, dias_semana: float = 5.0) -> dict[str, float]:
    """Tamanho do pulmão de tempo antes da restrição (tambor-pulmão-corda).

    Lead time a montante (LT) = Σ atravessamento_dias das etapas anteriores à restrição.
      - Regra prática: pulmão = fator × LT. O `fator` é parâmetro de partida DECLARADO pelo usuário (0,5 por
        padrão — decisão de método deste módulo, não norma); a literatura de DBR manda calibrá-lo depois pelo
        gerenciamento do pulmão — zonas verde/amarela/vermelha em terços (Cox e Schleier, 2010).
      - Estatístico: pulmão = z × σ_LT, com σ_LT = √Σ (cv_i × LT_i)² (etapas independentes) — protege a restrição
        de ficar sem trabalho com nível de serviço Φ(z).
    Conversão tempo → casos pela lei de Little (L = λ·W; Little, 1961): casos = vazão diária × pulmão (dias).
    Retorna lead_time_montante, pulmao_regra_dias/casos, pulmao_estatistico_dias/casos (NaN se faltar cv) e os
    limites das zonas (terços do pulmão pela regra prática, em casos).
    """
    nomes = [e.nome for e in etapas]
    if restricao not in nomes:
        raise ValueError("Restrição inexistente.")
    montante = etapas[:nomes.index(restricao)]
    if any(e.atravessamento_dias is None for e in montante):
        raise ValueError("Informe atravessamento_dias para todas as etapas a montante da restrição.")
    lt = float(sum(e.atravessamento_dias for e in montante))
    por_dia = vazao_semanal / dias_semana
    regra = fator * lt
    if montante and all(e.cv is not None for e in montante):
        sigma = math.sqrt(sum((e.cv * e.atravessamento_dias) ** 2 for e in montante))
        est = z * sigma
    else:
        sigma, est = float("nan"), float("nan")
    casos = regra * por_dia
    return {"lead_time_montante": lt, "vazao_dia": por_dia, "pulmao_regra_dias": regra, "pulmao_regra_casos": casos,
            "sigma_lt": sigma, "pulmao_estatistico_dias": est, "pulmao_estatistico_casos": est * por_dia,
            "zona_verde_ate": casos, "zona_amarela_abaixo_de": 2 * casos / 3, "zona_vermelha_abaixo_de": casos / 3}


def efeito_subordinar(demanda: float, vazao: float, semanas: int, pulmao_casos: float) -> pd.DataFrame:
    """Passo 3 — subordinar (a corda). Compara, semana a semana, duas políticas de liberação de trabalho:

    sem corda — tudo o que chega entra no fluxo: o estoque em processo (WIP) diante da restrição cresce
    (demanda − vazão) por semana e o tempo de atravessamento interno (Little: W = WIP / vazão) explode;
    com corda — a entrada é liberada ao ritmo do tambor (vazão da restrição) e o WIP interno fica no pulmão; o
    excedente fica numa fila de entrada visível, triada e priorizada. A corda NÃO elimina o déficit de capacidade
    (o acúmulo total é o mesmo): torna-o visível e mantém previsível o prazo do que está em andamento. Só o passo 4
    (elevar) fecha a diferença.
    """
    linhas = []
    for s in range(1, semanas + 1):
        acum = max(0.0, demanda - vazao) * s
        wip_sem = pulmao_casos + acum
        wip_com = pulmao_casos
        linhas.append({"semana": s, "wip_sem_corda": wip_sem, "atravessamento_sem_corda_sem": wip_sem / vazao,
                       "wip_com_corda": wip_com, "fila_entrada_com_corda": acum,
                       "atravessamento_com_corda_sem": wip_com / vazao})
    return pd.DataFrame(linhas).set_index("semana")


# =====================================================================================================================
# Gráficos
# =====================================================================================================================
def grafico_perfis(perfis: Mapping[str, pd.DataFrame], capacidades: Mapping[str, float] | None, caminho: str) -> str:
    """Histogramas de recursos antes/depois (colunas = cronogramas; linhas = recursos), com linha de capacidade."""
    plt = estilo_graficos()
    capacidades = capacidades or {}
    nomes = list(perfis)
    recursos = list(next(iter(perfis.values())).columns)
    T = max(len(p) for p in perfis.values())
    ymax = max(float(p.to_numpy().max()) for p in perfis.values())
    for r in recursos:
        if math.isfinite(capacidades.get(r, math.inf)):
            ymax = max(ymax, capacidades[r])
    cores = [PALETA["cinza"], PALETA["azul"], PALETA["verde_escuro"], PALETA["marinho"]]
    fig, axs = plt.subplots(len(recursos), len(nomes), figsize=(4.2 * len(nomes), 2.6 * len(recursos)),
                            squeeze=False, sharex=True, sharey=True)
    for j, nome in enumerate(nomes):
        p = perfis[nome].reindex(range(1, T + 1), fill_value=0.0)
        for i, r in enumerate(recursos):
            ax = axs[i][j]
            ax.bar(p.index, p[r], width=0.85, color=cores[j % len(cores)])
            cap = capacidades.get(r, math.inf)
            if math.isfinite(cap):
                ax.axhline(cap, color=PALETA["marinho"], ls="--", lw=1.2)
                ax.text(T + 0.4, cap, f"capac. {br(cap, 0)}", va="bottom", ha="right", fontsize=8,
                        color=PALETA["marinho"])
            ax.set_title(f"{nome} — {r}\npico {br(p[r].max(), 1)} · Σr² {br(soma_quadrados(p[r]), 0)}", fontsize=9)
            ax.set_ylim(0, ymax * 1.15)
            if j == 0:
                ax.set_ylabel("servidores")
            if i == len(recursos) - 1:
                ax.set_xlabel("semana")
    fig.suptitle("Histograma de recursos: cronograma mais cedo × suavizado × nivelado", fontsize=12)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_toc(cenarios: Mapping[str, Sequence[Etapa]], horas_por_servidor: float, demanda: float | None,
                caminho: str) -> str:
    """Capacidade de cada etapa (casos/semana) por cenário, com a linha da demanda e a restrição destacada."""
    plt = estilo_graficos()
    nomes = list(cenarios)
    etapas_nomes = [e.nome for e in next(iter(cenarios.values()))]
    x = np.arange(len(etapas_nomes))
    w = 0.8 / len(nomes)
    cores = [PALETA["cinza"], PALETA["azul"], PALETA["verde_escuro"], PALETA["marinho"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k, nome in enumerate(nomes):
        tab, res = analise_fluxo(cenarios[nome], horas_por_servidor, demanda)
        caps = tab["capacidade"].reindex(etapas_nomes).to_numpy()
        barras = ax.bar(x + (k - (len(nomes) - 1) / 2) * w, caps, width=w, color=cores[k % len(cores)],
                        label=f"{nome} (vazão {br(res['vazao'], 1)})")
        i_r = etapas_nomes.index(res["restricao"])
        barras[i_r].set_edgecolor(PALETA["marinho"])
        barras[i_r].set_linewidth(2.2)
        barras[i_r].set_hatch("//")
    if demanda is not None:
        ax.axhline(demanda, color=PALETA["marinho"], ls="--", lw=1.4)
        ax.text(-0.45, demanda, f"demanda {br(demanda, 0)}/semana", ha="left", va="bottom",
                color=PALETA["marinho"], fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in etapas_nomes], fontsize=9)
    ax.set_ylabel("capacidade (casos/semana)")
    ax.set_title("TOC — capacidade por etapa (hachura = restrição de cada cenário)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# =====================================================================================================================
# Relatórios e CLI
# =====================================================================================================================
def _fmt_tab(df: pd.DataFrame, casas: int = 2) -> str:
    def f(v):
        if isinstance(v, (bool, np.bool_)):
            return "sim" if v else ""
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(v):
                return "—" if math.isnan(v) else "∞"
            return br(v, casas)
        return str(v)
    aplicar = df.map if hasattr(df, "map") else df.applymap   # pandas < 2.1
    return aplicar(f).to_string()


def _parse_capacidades(itens: Sequence[str] | None, arquivo: str | None) -> dict[str, float]:
    caps: dict[str, float] = {}
    if arquivo:
        d = ler_csv(arquivo)
        d.columns = [c.lower().strip() for c in d.columns]
        caps.update({str(r["recurso"]).strip(): float(r["capacidade"]) for _, r in d.iterrows()})
    for it in itens or []:
        nome, _, val = it.partition("=")
        caps[nome.strip()] = float(val.replace(",", "."))
    return caps


def relatorio_recursos(ativs: Sequence[Atividade], capacidades: Mapping[str, float], prazo: int | None = None,
                       regra: str = "folga", grafico: str | None = None) -> str:
    """Relatório completo da Parte A (texto em português)."""
    t = cpm(ativs, prazo)
    fim = int(t["EF"].max()) if prazo is None else int(prazo)
    cedo = {i: int(t.loc[i, "ES"]) for i in t.index}
    suav = suavizar(ativs, prazo)
    crons = {"cedo": cedo, "suavizado": suav}
    perfis = {"Mais cedo (CPM)": perfil_recursos(ativs, cedo, fim), "Suavizado": perfil_recursos(ativs, suav, fim)}
    if capacidades:
        niv = nivelar(ativs, capacidades, regra)
        crons["nivelado"] = niv
        perfis["Nivelado"] = perfil_recursos(ativs, niv)
    L = ["=" * 100, "NIVELAMENTO DE RECURSOS — suavização (prazo fixo) × nivelamento (capacidade fixa)", "=" * 100,
         f"Atividades: {len(ativs)} · recursos: {', '.join(sorted({a.recurso for a in ativs}))} · "
         f"prazo pelo CPM: {fim} semanas" + ("" if prazo is None else " (prazo informado)"), ""]
    L += ["1) Caminho crítico (CPM) — semanas 1-based; folga em semanas", ""]
    tc = t.copy()
    tc["inicio_cedo"] = tc["ES"] + 1
    tc["inicio_tarde"] = tc["LS"] + 1
    L.append(_fmt_tab(tc[["inicio_cedo", "EF", "inicio_tarde", "LF", "folga_total", "folga_livre", "critica"]]
                      .rename(columns={"EF": "fim_cedo", "LF": "fim_tarde"}), 0))
    cc = caminho_critico(ativs, t)
    L += ["", f"Caminho crítico: {' → '.join(cc)}", ""]
    L += ["2) Cronogramas (início–término, em semanas)", ""]
    tab = tabela_cronograma(ativs, **crons)
    L.append(_fmt_tab(tab.drop(columns=["entrega"]), 1))
    L += ["", "3) Indicadores do histograma por recurso", ""]
    for nome, p in perfis.items():
        ind = indicadores_perfil(p, capacidades)
        L.append(f"   [{nome}] término: semana {len(p)} · Σr² total = {br(soma_quadrados(p), 1)}")
        L.append(_fmt_tab(ind[["area", "pico", "media", "pico_sobre_media", "soma_quadrados", "capacidade",
                               "semanas_acima"]], 2))
        L.append("")
    ind0 = indicadores_perfil(perfis["Mais cedo (CPM)"], capacidades)
    ind1 = indicadores_perfil(perfis["Suavizado"], capacidades)
    L += ["4) Leitura para o DFT", ""]
    for r in ind0.index:
        L.append(f"   {r}: dimensionar pelo pico exigiria {math.ceil(ind0.loc[r, 'pico'] - 1e-9)} servidor(es) no "
                 f"cronograma mais cedo e {math.ceil(ind1.loc[r, 'pico'] - 1e-9)} no suavizado; a carga média é "
                 f"{br(ind0.loc[r, 'media'], 2)} (área {br(ind0.loc[r, 'area'], 1)} servidor-semanas em {fim} semanas).")
    if capacidades:
        pn = perfis["Nivelado"]
        atraso = len(pn) - fim
        L.append(f"   Nivelado com capacidade {', '.join(f'{k}={br(v, 0)}' for k, v in capacidades.items())}: "
                 f"término na semana {len(pn)} ({'+' if atraso >= 0 else ''}{atraso} semana(s) em relação ao CPM).")
        col = [c for c in tab.columns if c.startswith("atraso_nivelado")]
        if col and tab[col[0]].sum() > 0:
            viol = tab[tab[col[0]] > 0]
            L.append("   Atenção — entregas que estouram a semana-limite após o nivelamento: " +
                     ", ".join(f"{i} (+{int(v)} sem.)" for i, v in viol[col[0]].items()))
    L.append("   Regra: Q = Σ(D·E)/H usa a MÉDIA; ela só é exequível se a carga puder ser suavizada/nivelada dentro "
             "das janelas. O pico mostra quanto do quadro é efeito do calendário, não da carga.")
    seg = [next(a for a in ativs if a.id == i).duracao for i in cc]
    pc, pr = pulmao_projeto(seg, metodo="corte"), pulmao_projeto(seg, metodo="rss")
    L += ["", "5) Corrente crítica (ilustração sobre o caminho crítico; durações tratadas como estimativas seguras)",
          f"   Corrente segura: {br(pc['corrente_segura'], 1)} sem. · agressiva (50 %): "
          f"{br(pc['corrente_agressiva'], 1)} sem.",
          f"   Pulmão por corte de 50 %: {br(pc['pulmao'], 1)} sem. → prazo {br(pc['prazo_com_pulmao'], 1)} sem.",
          f"   Pulmão por raiz da soma dos quadrados: {br(pr['pulmao'], 1)} sem. → prazo "
          f"{br(pr['prazo_com_pulmao'], 1)} sem.", ""]
    if grafico:
        grafico_perfis(perfis, capacidades, grafico)
        L.append(f"Gráfico salvo em: {grafico}")
    return "\n".join(L)


def relatorio_toc(etapas: Sequence[Etapa], horas_por_servidor: float, demanda: float | None, reducao: float = 0.15,
                  destino: str | None = None, incremento: float = 1.0, custo_servidor: float | None = None,
                  ganho_por_caso: float | None = None, alvo_restricao: float = 0.95, alvo_demais: float = 0.85,
                  fator_pulmao: float = 0.5, z: float = 1.645, dias_semana: float = 5.0, semanas: int = 12,
                  grafico: str | None = None) -> str:
    """Relatório completo da Parte B — os 5 passos de focalização aplicados ao fluxo (texto em português)."""
    tab, res = analise_fluxo(etapas, horas_por_servidor, demanda)
    restr = res["restricao"]
    L = ["=" * 100, "TEORIA DAS RESTRIÇÕES — os 5 passos de focalização", "=" * 100,
         f"Etapas: {len(etapas)} · demanda: {'não informada' if demanda is None else br(demanda, 1) + ' casos/semana'}"
         f" · horas efetivas por servidor (padrão): {br(horas_por_servidor, 1)} h/semana", ""]
    L += ["PASSO 1 — IDENTIFICAR a restrição", ""]
    cols = ["servidores", "tempo_h", "fracao", "horas_servidor", "capacidade"] + (
        ["carga_h", "utilizacao"] if demanda is not None else []) + ["restricao"]
    L.append(_fmt_tab(tab[cols], 2))
    L += ["", f"   Restrição interna: {restr} (capacidade {br(res['capacidade_restricao'], 2)} casos/semana).",
          f"   Vazão do sistema: {br(res['vazao'], 2)} casos/semana."]
    if demanda is not None:
        if res["restricao_externa"]:
            L.append("   A demanda é menor que todas as capacidades: a restrição é EXTERNA (a própria demanda).")
        else:
            L.append(f"   Déficit: acumulam-se {br(res['acumulo_semanal'], 2)} casos/semana "
                     f"({br(res['acumulo_semanal'] * semanas, 1)} em {semanas} semanas). Uma hora perdida na "
                     f"restrição é uma hora perdida no sistema inteiro.")
    L += ["", f"PASSO 2 — EXPLORAR a restrição (tirar dela {br(reducao * 100, 0)} % do tempo por caso"
              + (f", transferindo-o para '{destino}')" if destino else ")"), ""]
    expl = explorar(etapas, restr, reducao, destino)
    tab2, res2 = analise_fluxo(expl, horas_por_servidor, demanda)
    L.append(f"   Vazão: {br(res['vazao'], 2)} → {br(res2['vazao'], 2)} casos/semana "
             f"(+{br(res2['vazao'] - res['vazao'], 2)}, sem nenhum servidor a mais).")
    L.append(f"   Nova restrição: {'demanda (externa)' if res2['restricao_externa'] else res2['restricao']}"
             f" · utilização de '{restr}': {br(tab.loc[restr, 'utilizacao'] if demanda else float('nan'), 2)} → "
             f"{br(tab2.loc[restr, 'utilizacao'] if demanda else float('nan'), 2)}")
    L += ["", "PASSO 3 — SUBORDINAR tudo à restrição (tambor-pulmão-corda)", ""]
    vaz_tambor = res2["vazao"]
    try:
        pul = pulmao_tambor(expl, restr, vaz_tambor, fator_pulmao, z, dias_semana)
        L += [f"   Tambor: '{restr}' dita o ritmo — {br(vaz_tambor, 2)} casos/semana "
              f"({br(pul['vazao_dia'], 2)}/dia).",
              f"   Lead time a montante: {br(pul['lead_time_montante'], 1)} dias úteis.",
              f"   Pulmão (regra prática {br(fator_pulmao, 2)} × LT): {br(pul['pulmao_regra_dias'], 1)} dias ≈ "
              f"{br(pul['pulmao_regra_casos'], 1)} casos na fila da restrição.",
              f"   Pulmão estatístico (z = {br(z, 3)}; σ_LT = {br(pul['sigma_lt'], 2)} d): "
              f"{br(pul['pulmao_estatistico_dias'], 1)} dias ≈ {br(pul['pulmao_estatistico_casos'], 1)} casos.",
              f"   Gerenciamento do pulmão (casos na fila): verde ≥ {br(pul['zona_amarela_abaixo_de'], 1)}; "
              f"amarelo entre {br(pul['zona_vermelha_abaixo_de'], 1)} e {br(pul['zona_amarela_abaixo_de'], 1)}; "
              f"vermelho < {br(pul['zona_vermelha_abaixo_de'], 1)} (acionar ação de recuperação).",
              "   Corda: liberar novos casos ao fluxo no ritmo do tambor; não-restrições não maximizam produção "
              "local (ociosidade nelas é capacidade protetiva, não desperdício)."]
        if demanda is not None and demanda > res["vazao"]:
            for rot, vz in (("situação atual", res["vazao"]), ("após explorar", vaz_tambor)):
                if demanda <= vz:
                    L.append(f"   {rot}: vazão ≥ demanda — não há acúmulo a controlar.")
                    continue
                sub = efeito_subordinar(demanda, vz, semanas, pul["pulmao_regra_casos"])
                u = sub.iloc[-1]
                L.append(f"   Em {semanas} semanas ({rot}, vazão {br(vz, 2)}): sem corda, WIP interno "
                         f"{br(u['wip_sem_corda'], 1)} casos e atravessamento {br(u['atravessamento_sem_corda_sem'], 1)}"
                         f" sem.; com corda, WIP {br(u['wip_com_corda'], 1)} e "
                         f"{br(u['atravessamento_com_corda_sem'], 1)} sem., com {br(u['fila_entrada_com_corda'], 1)} "
                         f"casos visíveis na fila de entrada (o déficit não some).")
    except ValueError as exc:
        L.append(f"   (pulmão não calculado: {exc})")
    L += ["", f"PASSO 4 — ELEVAR a restrição (+{br(incremento, 0)} servidor em cada etapa, uma de cada vez)", "",
          "   (a) Comparação ingênua — elevar SEM ter explorado:"]
    elev0 = cenarios_elevar(etapas, horas_por_servidor, demanda, incremento, custo_servidor, ganho_por_caso)
    L.append(_fmt_tab(elev0, 2))
    L += ["", "   (b) Sequência da TOC — elevar DEPOIS de explorar:"]
    elev = cenarios_elevar(expl, horas_por_servidor, demanda, incremento, custo_servidor, ganho_por_caso)
    L.append(_fmt_tab(elev, 2))
    L.append("   Só o servidor a mais na restrição aumenta a vazão; nas não-restrições, ΔT = 0 e a OE sobe.")
    if demanda is not None and not res2["restricao_externa"]:
        e_r = next(e for e in expl if e.nome == restr)
        H_r = _h(e_r, horas_por_servidor)
        falta = demanda * e_r.fracao * e_r.tempo_h / H_r - e_r.servidores
        L.append(f"   Elevação mínima para atender a demanda após explorar: +{br(falta, 2)} servidor na restrição "
                 f"(≈ {br(falta * H_r, 1)} h/semana a 100 % de utilização) — pode vir de realocação parcial, apoio "
                 f"em tarefas não técnicas ou horas adicionais, antes de se cogitar provimento.")
    realoc = cenarios_realocar(expl, horas_por_servidor, demanda)
    if len(realoc):
        L += ["", "   Alternativa sem custo — realocar 1 servidor polivalente do mesmo grupo para a restrição:",
              _fmt_tab(realoc.set_index("origem"), 2)]
    if demanda is not None:
        dim = dimensionar_pela_restricao(expl, horas_por_servidor, demanda, restr, alvo_restricao, alvo_demais)
        L += ["", f"   Dimensionamento pela restrição (vazão-alvo = demanda {br(demanda, 1)}; ρ restrição "
                  f"{br(alvo_restricao, 2)}, demais {br(alvo_demais, 2)} — capacidade protetiva):",
              _fmt_tab(dim[["papel", "utilizacao_alvo", "carga_h", "Q_fracionario", "Q", "atual", "diferenca",
                            "utilizacao_resultante"]], 2),
              f"   Total: {int(dim['Q'].sum())} servidores necessários × {br(dim['atual'].sum(), 0)} atuais."]
    melhor = elev.drop(index="atual")["delta_vazao"].idxmax()
    novo = [replace(e, servidores=e.servidores + incremento) if f"+{br(incremento, 0)} em {e.nome}" == melhor else e
            for e in expl]
    _, res4 = analise_fluxo(novo, horas_por_servidor, demanda)
    L += ["", "PASSO 5 — NÃO DEIXAR A INÉRCIA virar restrição: voltar ao passo 1", "",
          f"   Após '{melhor}': vazão {br(res4['vazao'], 2)}; restrição agora: "
          f"{'demanda (externa) — ' + res4['restricao'] + ' é a etapa interna mais carregada' if res4['restricao_externa'] else res4['restricao']}.",
          "   Reavaliar políticas (metas de produção por etapa, regras de distribuição) que faziam sentido para a "
          "restrição antiga.", ""]
    if grafico:
        cen = {"atual": list(etapas), "explorar": expl, "explorar + elevar": novo}
        grafico_toc(cen, horas_por_servidor, demanda, grafico)
        L.append(f"Gráfico salvo em: {grafico}")
    return "\n".join(L)


def cli(argv: list[str]) -> int:
    """Linha de comando: `dft nivelamento ARQUIVO.csv [opções]`.

    O modo é detectado pelas colunas: com `duracao` e `recurso` → nivelamento de recursos; com `servidores` e
    `tempo_h` → Teoria das Restrições. Use `--modo` para forçar.
    """
    ap = argparse.ArgumentParser(prog="dft nivelamento", description="Nivelamento de recursos e Teoria das Restrições.")
    ap.add_argument("arquivo", help="CSV de atividades (recursos) ou de etapas do fluxo (toc)")
    ap.add_argument("--modo", choices=["auto", "recursos", "toc"], default="auto")
    g = ap.add_argument_group("recursos")
    g.add_argument("--capacidade", action="append", metavar="RECURSO=N", help="capacidade (repetível)")
    g.add_argument("--capacidades", metavar="CSV", help="CSV com colunas recurso;capacidade")
    g.add_argument("--prazo", type=int, default=None, help="prazo do projeto em semanas (padrão: término do CPM)")
    g.add_argument("--regra", choices=sorted(_REGRAS), default="folga", help="regra de prioridade do nivelamento")
    t = ap.add_argument_group("toc")
    t.add_argument("--demanda", type=float, default=None, help="casos/semana (ou coluna 'demanda' no CSV)")
    t.add_argument("--horas-servidor", type=float, default=28.0, help="horas efetivas por servidor por semana")
    t.add_argument("--explorar", type=float, default=0.15, help="redução do tempo por caso na restrição (fração)")
    t.add_argument("--destino-explorar", default=None, help="etapa que absorve o trabalho tirado da restrição")
    t.add_argument("--incremento", type=float, default=1.0)
    t.add_argument("--custo-servidor", type=float, default=None, help="R$ por servidor por semana (OE)")
    t.add_argument("--ganho-por-caso", type=float, default=None, help="R$ de ganho (T) por caso concluído")
    t.add_argument("--alvo-restricao", type=float, default=0.95)
    t.add_argument("--alvo-demais", type=float, default=0.85)
    t.add_argument("--fator-pulmao", type=float, default=0.5)
    t.add_argument("--z", type=float, default=1.645)
    t.add_argument("--semanas", type=int, default=12)
    ap.add_argument("--grafico", metavar="saida.png", default=None)
    a = ap.parse_args(argv)
    try:                                   # console do Windows: garante acentos e setas na saída
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    df = ler_csv(a.arquivo)
    cols = {c.lower().strip() for c in df.columns}
    modo = a.modo
    if modo == "auto":
        modo = "recursos" if {"duracao", "recurso"} <= cols else "toc" if {"servidores", "tempo_h"} <= cols else ""
        if not modo:
            print("Não reconheci o tipo do CSV: use --modo recursos|toc.")
            return 2
    try:
        if modo == "recursos":
            print(relatorio_recursos(atividades_de_dataframe(df), _parse_capacidades(a.capacidade, a.capacidades),
                                     a.prazo, a.regra, a.grafico))
        else:
            demanda = a.demanda
            if demanda is None and "demanda" in cols:
                col = next(c for c in df.columns if c.lower().strip() == "demanda")
                v = df[col].dropna()
                demanda = float(v.iloc[0]) if len(v) else None
            print(relatorio_toc(etapas_de_dataframe(df), a.horas_servidor, demanda, a.explorar, a.destino_explorar,
                                a.incremento, a.custo_servidor, a.ganho_por_caso, a.alvo_restricao, a.alvo_demais,
                                a.fator_pulmao, a.z, semanas=a.semanas, grafico=a.grafico))
    except ValueError as exc:
        print(f"Erro: {exc}")
        return 1
    return 0
