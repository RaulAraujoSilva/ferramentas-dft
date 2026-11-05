"""2. Identificação de recursos por atividade — matriz entrega × atividade × perfil, carga e custo por perfil.

Etapa do DFT: **Fase 5 — Identificação do tipo de recurso** (Rodrigues et al., 2021), alimentando a Fase 8
(consolidação: visões "quadro por cargo", "h/h por entrega", "competência" e "custo por entrega").

O que o módulo faz (funções puras, sem I/O escondido):

1. ``matriz_recursos``: cruza o catálogo de entregas (com demanda) com as atividades de cada entrega (perfil,
   tempo e frequência) e devolve a matriz longa entrega × atividade × perfil, em horas por unidade e horas/ano.
2. ``horas_por_perfil`` / ``horas_por_entrega`` / ``participacao_perfis``: agregações da matriz.
3. ``disponibilidade_perfil`` e ``taxa_custo_capacidade``: denominador por perfil (dias efetivos × jornada × ITP)
   e custo-hora pela lógica do TDABC (custo da capacidade fornecida ÷ capacidade prática).
4. ``custo_por_entrega``: custo unitário e anual de cada entrega (TDABC: Σ tempo × taxa do perfil).
5. ``carga_por_perfil``: Q_r = carga_r / H_r com arredondamento declarado, déficit em coluna separada, razão WISN
   e capacidade não utilizada (TDABC) — a tabela que vai para a fase de consolidação.
6. ``alertas``: atividades sem perfil, perfis sem atividade, perfis não cadastrados, entregas sem atividade,
   regras RACI e candidatas à realocação de nível ("cargo de menor complexidade possível").
7. ``fatores_wisn`` / ``requerido_wisn``: fatores de ajuste do método WISN da OMS (CAF e IAF), para quem precisa
   reproduzir o método — com o alerta da decisão D2 do curso (não embutir atividade indireta duas vezes).

Fórmulas (notação do dossiê L1, Parte 3.2, variante (a), e L2, §4.3):

    h(e, a, r)  = tempo(e, a, r) × frequência(e, a, r)                  [h por unidade da entrega]
    E(e, r)     = Σ_a h(e, a, r)                                         [esforço da entrega e no perfil r]
    carga_r     = Σ_e D_e × E(e, r)                                      [h/ano do perfil r]
    H_r         = dias_efetivos_r × jornada_r × ITP_r                    [h/ano por pessoa do perfil r]
    Q_r         = carga_r / H_r ;   Q = Σ_r Q_r   (nunca carga_total / H único)
    taxa_r      = custo_anual_r / H_r                                    [R$/h — TDABC, capacidade prática]
    custo(e)    = Σ_r E(e, r) × taxa_r ;  custo_anual(e) = D_e × custo(e)
    ociosa_r    = lotação_r × H_r − carga_r                              [h/ano; < 0 = sobrecarga]
    razão WISN  = lotação_r / Q_r                                        [< 1 = pressão de carga]

Referências: Rodrigues et al. (2021); Portaria SEDGG/ME nº 7.888/2022; Serrano et al. (2018); Mendes, Serrano e
Meneses (2022); Kaplan e Anderson (2004; 2007); WHO (2023) WISN; PMI (2017) PMBOK, matriz RACI; Peterson et al.
(2001) O*NET; Morgeson e Campion (1997). Ver ``docs/metodos/2_recursos.md``.
Números canônicos (L1, Parte 3.6, Passo 8 e Parte 4, Fase 5, §5.6): técnico 2.517,6 h → 2,71 → 3 e agente
5.252,4 h → 5,66 → 6 com H = 928,2 h; com jornadas distintas, técnico 3.417,6 h / 928,2 = 3,68 → 4 e agente
4.352,4 h / 1.237,6 = 3,52 → 4 (total 8, contra 8,37 → 9 do cálculo agregado).
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._comum import PALETA, SERIES, br, estilo_graficos, formatar_eixos_br, ler_csv

REGRAS_ARREDONDAMENTO = ("teto", "comum", "piso_com_residuo")
SEM_PERFIL = "(sem perfil)"


class ErroRecursos(ValueError):
    """Erro de validação dos dados de entrada do módulo de recursos."""


# --------------------------------------------------------------------------------------------------------------
# Normalização das tabelas de entrada
# --------------------------------------------------------------------------------------------------------------

def _texto(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _numero(s: pd.Series) -> pd.Series:
    """Converte para float aceitando vírgula decimal residual ("1,5")."""
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        t = s.astype(str).str.strip()
        com_virgula = t.str.contains(",", regex=False)
        t = t.where(~com_virgula, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        s = t.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _exigir(df: pd.DataFrame, colunas: list[str], tabela: str) -> None:
    faltam = [c for c in colunas if c not in df.columns]
    if faltam:
        raise ErroRecursos(f"Tabela de {tabela}: faltam as colunas {faltam}. Colunas lidas: {list(df.columns)}.")


def normalizar_entregas(entregas: pd.DataFrame) -> pd.DataFrame:
    """Valida o catálogo de entregas.

    Entrada: colunas ``entrega`` (identificador), ``demanda`` (ocorrências no período, já com vetores aplicados —
    Fase 4) e, opcionalmente, ``nome`` e ``unidade_medida``.
    Saída: DataFrame com ``entrega``, ``nome``, ``demanda`` (float ≥ 0) e ``unidade_medida``.
    """
    df = entregas.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    _exigir(df, ["entrega", "demanda"], "entregas")
    df["entrega"] = _texto(df["entrega"])
    df["nome"] = _texto(df["nome"]) if "nome" in df else df["entrega"]
    df.loc[df["nome"] == "", "nome"] = df["entrega"]
    df["unidade_medida"] = _texto(df["unidade_medida"]) if "unidade_medida" in df else ""
    df["demanda"] = _numero(df["demanda"])
    if df["demanda"].isna().any() or (df["demanda"] < 0).any():
        ruins = df.loc[df["demanda"].isna() | (df["demanda"] < 0), "entrega"].tolist()
        raise ErroRecursos(f"Demanda ausente ou negativa nas entregas {ruins}.")
    if df["entrega"].duplicated().any():
        raise ErroRecursos(f"Entregas duplicadas: {df.loc[df['entrega'].duplicated(), 'entrega'].tolist()}.")
    return df[["entrega", "nome", "demanda", "unidade_medida"]].reset_index(drop=True)


def normalizar_atividades(atividades: pd.DataFrame) -> pd.DataFrame:
    """Valida a tabela de atividades por entrega.

    Entrada: uma linha por (entrega, atividade, perfil) com ``entrega``, ``atividade``, ``perfil`` (pode vir vazio
    — vira alerta), ``tempo_h`` (horas por ocorrência da atividade, idealmente o P80 da Fase 6) e, opcionalmente,
    ``frequencia`` (ocorrências da atividade por unidade da entrega; padrão 1 — implementa o termo condicional da
    "equação de tempo" do TDABC), ``raci`` (R, A, C, I ou a combinação A/R, normalizada para "AR") e
    ``privativa`` (sim/não).
    Atividade compartilhada entre perfis = uma linha por perfil, cada uma com o seu tempo.
    """
    df = atividades.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    _exigir(df, ["entrega", "atividade", "perfil", "tempo_h"], "atividades")
    for c in ("entrega", "atividade", "perfil"):
        df[c] = _texto(df[c])
    df["tempo_h"] = _numero(df["tempo_h"])
    df["frequencia"] = _numero(df["frequencia"]).fillna(1.0) if "frequencia" in df else 1.0
    df["raci"] = (_texto(df["raci"]).str.upper().str.replace(r"[\s/+,-]", "", regex=True)
                  .map(lambda x: "".join(sorted(set(x)))) if "raci" in df else "")
    if "privativa" in df:
        df["privativa"] = _texto(df["privativa"]).str.lower().isin(["sim", "s", "1", "true", "verdadeiro", "x"])
    else:
        df["privativa"] = False
    if df["tempo_h"].isna().any() or (df["tempo_h"] < 0).any():
        ruins = df.loc[df["tempo_h"].isna() | (df["tempo_h"] < 0), ["entrega", "atividade"]].values.tolist()
        raise ErroRecursos(f"Tempo ausente ou negativo nas atividades {ruins}.")
    if (df["frequencia"] < 0).any():
        raise ErroRecursos("Frequência negativa em alguma atividade.")
    invalidos = sorted(x for x in set(df["raci"]) if set(x) - set("RACI"))
    if invalidos:
        raise ErroRecursos(f"Valores RACI inválidos: {invalidos}. Use R, A, C, I ou A/R (quem executa e responde).")
    return df[["entrega", "atividade", "perfil", "tempo_h", "frequencia", "raci", "privativa"]].reset_index(drop=True)


def disponibilidade_perfil(dias_efetivos: float, jornada_h: float, itp: float = 0.70) -> float:
    """Disponibilidade efetiva H de uma pessoa do perfil, em h/período (Fase 7; L1, Parte 3.5; L2, §4.2).

    ``H = dias_efetivos × jornada_h × ITP``. Ex.: 221 × 6 × 0,70 = 928,2 h/ano (exemplo de nove passos).
    """
    if dias_efetivos <= 0 or jornada_h <= 0:
        raise ErroRecursos("Dias efetivos e jornada precisam ser positivos.")
    if not 0 < itp <= 1:
        raise ErroRecursos("O ITP precisa estar em (0, 1].")
    return float(dias_efetivos) * float(jornada_h) * float(itp)


def taxa_custo_capacidade(custo_anual: float, capacidade_pratica_h: float) -> float:
    """Taxa de custo da capacidade (TDABC; Kaplan e Anderson, 2004), em R$/h.

    ``taxa = custo da capacidade fornecida ÷ capacidade prática``. No DFT a capacidade prática é a disponibilidade
    efetiva H (já descontadas ausências e ITP). Dividir pela jornada bruta subestima o custo-hora.
    """
    if capacidade_pratica_h <= 0:
        raise ErroRecursos("A capacidade prática precisa ser positiva.")
    if custo_anual < 0:
        raise ErroRecursos("O custo anual não pode ser negativo.")
    return float(custo_anual) / float(capacidade_pratica_h)


def normalizar_perfis(perfis: pd.DataFrame) -> pd.DataFrame:
    """Valida a tabela de perfis (tipos de recurso) e calcula H e custo-hora.

    Entrada: ``perfil`` e, para o denominador, ``disponibilidade_h`` **ou** ``jornada_h`` + ``dias_efetivos``
    (+ ``itp``, padrão 0,70). Para custo: ``custo_hora`` **ou** ``custo_anual`` (remuneração + encargos +
    benefícios por pessoa/ano; o custo-hora sai de custo_anual ÷ H). Opcionais: ``cargo``, ``lotacao_atual``
    (pessoas em exercício, não lotadas — Portaria 7.888/2022, art. 4º, IV) e ``complexidade`` (1 = menor).
    Saída: colunas padronizadas, com ``H`` e ``custo_hora`` (NaN quando não houver dado de custo).
    """
    df = perfis.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    _exigir(df, ["perfil"], "perfis")
    df["perfil"] = _texto(df["perfil"])
    if df["perfil"].duplicated().any():
        raise ErroRecursos(f"Perfis duplicados: {df.loc[df['perfil'].duplicated(), 'perfil'].tolist()}.")
    df["cargo"] = _texto(df["cargo"]) if "cargo" in df else df["perfil"]
    for c in ("disponibilidade_h", "jornada_h", "dias_efetivos", "itp", "custo_hora", "custo_anual",
              "lotacao_atual", "complexidade"):
        df[c] = _numero(df[c]) if c in df else np.nan
    df["itp"] = df["itp"].fillna(0.70)
    hs = []
    for _, l in df.iterrows():
        if pd.notna(l["disponibilidade_h"]):
            if l["disponibilidade_h"] <= 0:
                raise ErroRecursos(f"Perfil {l['perfil']}: disponibilidade_h precisa ser positiva.")
            hs.append(float(l["disponibilidade_h"]))
        elif pd.notna(l["jornada_h"]) and pd.notna(l["dias_efetivos"]):
            hs.append(disponibilidade_perfil(l["dias_efetivos"], l["jornada_h"], l["itp"]))
        else:
            raise ErroRecursos(f"Perfil {l['perfil']}: informe disponibilidade_h ou jornada_h + dias_efetivos.")
    df["H"] = hs
    taxa = df["custo_hora"].astype(float).copy()
    sem_taxa = taxa.isna() & df["custo_anual"].notna()
    taxa[sem_taxa] = [taxa_custo_capacidade(c, h) for c, h in zip(df.loc[sem_taxa, "custo_anual"],
                                                                df.loc[sem_taxa, "H"])]
    df["custo_hora"] = taxa
    return df[["perfil", "cargo", "H", "custo_hora", "custo_anual", "lotacao_atual", "complexidade",
               "jornada_h", "dias_efetivos", "itp"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------
# Cálculo
# --------------------------------------------------------------------------------------------------------------

def matriz_recursos(entregas: pd.DataFrame, atividades: pd.DataFrame,
                    perfis: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matriz longa entrega × atividade × perfil (Fase 5; L1, Fase 5, §5.7, primeiro produto).

    Entradas já normalizadas (ou brutas — são normalizadas aqui). Saída: uma linha por (entrega, atividade, perfil)
    com ``h_unit`` = tempo × frequência (h por unidade da entrega), ``demanda``, ``h_ano`` = demanda × h_unit e,
    se houver perfis com custo, ``custo_hora`` e ``custo_ano``. Atividade sem perfil recebe o rótulo
    ``(sem perfil)``; atividade de entrega fora do catálogo fica com demanda 0 (e vira alerta).
    """
    ent = normalizar_entregas(entregas)
    atv = normalizar_atividades(atividades)
    m = atv.merge(ent[["entrega", "nome", "demanda"]], on="entrega", how="left")
    m["demanda"] = m["demanda"].fillna(0.0)
    m["nome"] = m["nome"].fillna(m["entrega"])
    m["perfil"] = m["perfil"].replace("", SEM_PERFIL)
    m["h_unit"] = m["tempo_h"] * m["frequencia"]
    m["h_ano"] = m["demanda"] * m["h_unit"]
    if perfis is not None:
        pf = normalizar_perfis(perfis)
        m = m.merge(pf[["perfil", "custo_hora"]], on="perfil", how="left")
        m["custo_ano"] = m["h_ano"] * m["custo_hora"]
    return m


def horas_por_perfil(matriz: pd.DataFrame) -> pd.Series:
    """Carga de trabalho por perfil, em h/ano (numerador de Q_r). Inclui ``(sem perfil)`` se houver."""
    return matriz.groupby("perfil", sort=False)["h_ano"].sum()


def horas_por_entrega(matriz: pd.DataFrame) -> pd.DataFrame:
    """Esforço unitário e carga anual por entrega, com % do volume e % da carga (visão "h/h por entrega").

    Uma entrega com % do volume muito maior que % da carga é de alto volume e baixa complexidade — candidata
    natural à automação (L1, Parte 3.6, Passo 9: E2 = 3.600 de 6.480 ocorrências, 56% do volume, e 20% da carga).
    """
    g = matriz.groupby(["entrega", "nome"], sort=False).agg(demanda=("demanda", "first"),
                                                            h_unit=("h_unit", "sum"), h_ano=("h_ano", "sum"))
    g = g.reset_index()
    tot_d, tot_h = g["demanda"].sum(), g["h_ano"].sum()
    g["pct_volume"] = g["demanda"] / tot_d if tot_d else 0.0
    g["pct_carga"] = g["h_ano"] / tot_h if tot_h else 0.0
    return g


def matriz_entrega_perfil(matriz: pd.DataFrame, valor: str = "h_ano") -> pd.DataFrame:
    """Tabela cruzada entrega × perfil (h/ano por padrão; ``valor='h_unit'`` dá o esforço unitário por perfil)."""
    t = matriz.pivot_table(index="entrega", columns="perfil", values=valor, aggfunc="sum", fill_value=0.0,
                           sort=False)
    ordem = list(dict.fromkeys(matriz["entrega"]))
    return t.reindex(ordem)


def participacao_perfis(matriz: pd.DataFrame) -> pd.DataFrame:
    """Participação de cada perfil no esforço de cada entrega (linhas somam 1) — o "percentual de esforço
    atribuído a cada tipo" do produto da Fase 5. Última linha ``TOTAL``: participação na carga da unidade."""
    t = matriz_entrega_perfil(matriz, "h_ano")
    tot = t.sum(axis=1).replace(0, np.nan)
    p = t.div(tot, axis=0).fillna(0.0)
    geral = t.sum(axis=0) / t.values.sum() if t.values.sum() else t.sum(axis=0) * 0
    p.loc["TOTAL"] = geral
    return p


def custo_por_entrega(matriz: pd.DataFrame) -> pd.DataFrame:
    """Custo por entrega pelo TDABC: custo unitário = Σ h_unit × taxa do perfil; custo anual = demanda × unitário.

    Equivale à fórmula do volume de 2022 ``CE = %CT(E)/Q(E)`` (Mendes; Serrano; Meneses, 2022) quando o custo total
    de pessoal é repartido pelas horas efetivamente consumidas por cada entrega. Considera só custo direto de
    pessoal; horas sem perfil ou de perfil sem custo ficam fora e são contadas em ``h_sem_custo``.
    """
    if "custo_hora" not in matriz:
        raise ErroRecursos("A matriz não tem custo: passe a tabela de perfis com custo_anual ou custo_hora.")
    m = matriz.copy()
    m["c_unit"] = m["h_unit"] * m["custo_hora"]
    m["h_sem_custo"] = np.where(m["custo_hora"].isna(), m["h_ano"], 0.0)
    g = m.groupby(["entrega", "nome"], sort=False).agg(demanda=("demanda", "first"),
                                                       custo_unitario=("c_unit", lambda s: s.sum(min_count=0)),
                                                       custo_anual=("custo_ano", lambda s: s.sum(min_count=0)),
                                                       h_sem_custo=("h_sem_custo", "sum")).reset_index()
    tot = g["custo_anual"].sum()
    g["pct_custo"] = g["custo_anual"] / tot if tot else 0.0
    return g


def arredondar(q: float, regra: str = "teto") -> tuple[int, float]:
    """Arredondamento declarado na Fase 1 (decisão D6 do curso): ``teto`` (padrão), ``comum`` (≥ 0,5 sobe) ou
    ``piso_com_residuo`` (trunca e devolve a fração como resíduo). Retorna (inteiro, resíduo)."""
    if regra == "teto":
        return math.ceil(q - 1e-12), 0.0
    if regra == "comum":
        return int(math.floor(q + 0.5)), 0.0
    if regra == "piso_com_residuo":
        p = math.floor(q + 1e-12)
        return p, q - p
    raise ErroRecursos(f"Regra de arredondamento desconhecida: {regra}. Use {REGRAS_ARREDONDAMENTO}.")


def carga_por_perfil(matriz: pd.DataFrame, perfis: pd.DataFrame, regra: str = "teto") -> pd.DataFrame:
    """Carga por perfil pronta para a consolidação (Fase 8): Q_r = carga_r / H_r, por perfil, nunca agregado.

    Saída, uma linha por perfil cadastrado (inclusive os sem atividade) e, se houver, ``(sem perfil)``:
    ``carga_h``, ``H``, ``q_fracionario``, ``q`` (regra declarada), ``residuo``, ``lotacao_atual``, ``deficit``
    (q − lotação; coluna separada: necessidade ≠ déficit), ``razao_wisn`` (lotação / q_fracionário; < 1 indica
    pressão de carga), ``capacidade_h`` (lotação × H), ``ociosa_h`` (capacidade − carga; negativo = sobrecarga),
    ``custo_hora`` e ``custo_ocioso`` (TDABC: custo da capacidade não utilizada).
    """
    pf = normalizar_perfis(perfis)
    carga = horas_por_perfil(matriz)
    linhas = []
    for _, p in pf.iterrows():
        c = float(carga.get(p["perfil"], 0.0))
        qf = c / p["H"]
        q, res = arredondar(qf, regra)
        lot = p["lotacao_atual"]
        cap = lot * p["H"] if pd.notna(lot) else np.nan
        ociosa = cap - c if pd.notna(cap) else np.nan
        linhas.append({"perfil": p["perfil"], "cargo": p["cargo"], "carga_h": c, "H": p["H"],
                       "q_fracionario": qf, "q": q, "residuo": res, "lotacao_atual": lot,
                       "deficit": (q - lot) if pd.notna(lot) else np.nan,
                       "razao_wisn": (lot / qf) if (pd.notna(lot) and qf > 0) else np.nan,
                       "capacidade_h": cap, "ociosa_h": ociosa, "custo_hora": p["custo_hora"],
                       "custo_ocioso": ociosa * p["custo_hora"] if pd.notna(ociosa) and pd.notna(p["custo_hora"])
                       else np.nan})
    extra = set(carga.index) - set(pf["perfil"])
    for nome in sorted(extra):
        linhas.append({"perfil": nome, "cargo": "", "carga_h": float(carga[nome]), "H": np.nan,
                       "q_fracionario": np.nan, "q": np.nan, "residuo": np.nan, "lotacao_atual": np.nan,
                       "deficit": np.nan, "razao_wisn": np.nan, "capacidade_h": np.nan, "ociosa_h": np.nan,
                       "custo_hora": np.nan, "custo_ocioso": np.nan})
    return pd.DataFrame(linhas)


def fatores_wisn(cas_percentuais: list[float] | None = None, ias_horas: list[float] | None = None,
                 tempo_disponivel_h: float = 1.0) -> tuple[float, float]:
    """Fatores de ajuste do método WISN (WHO, 2023): CAF e IAF.

    - CAF (fator de ajuste da categoria) = 1 / (1 − ΣCAS/100), em que CAS é o percentual do tempo de trabalho
      consumido por atividades de apoio feitas por **todos** do perfil (registros, reuniões).
    - IAF (fator de ajuste individual) = ΣIAS / tempo disponível, em que IAS são as horas/ano de atividades
      adicionais feitas por **alguns** (chefia, comissões), em pessoas.
    Atenção (decisão D2 do curso): se a atividade indireta já está no catálogo como entrega própria, ela **não**
    entra no CAS/IAF; e o ITP já cobre parte desse tempo — não recontar.
    """
    cas = sum(cas_percentuais or [])
    if not 0 <= cas < 100:
        raise ErroRecursos("A soma dos CAS precisa estar em [0, 100).")
    if tempo_disponivel_h <= 0:
        raise ErroRecursos("O tempo disponível precisa ser positivo.")
    caf = 1.0 / (1.0 - cas / 100.0)
    iaf = sum(ias_horas or []) / tempo_disponivel_h
    return caf, iaf


def requerido_wisn(carga_servico_h: float, tempo_disponivel_h: float, caf: float = 1.0, iaf: float = 0.0) -> float:
    """Pessoal requerido pelo WISN: (carga de serviço / tempo disponível) × CAF + IAF.

    A carga de serviço ÷ tempo disponível é o mesmo que Σ (volume / carga-padrão), com carga-padrão = tempo
    disponível / padrão de atividade.
    """
    if tempo_disponivel_h <= 0:
        raise ErroRecursos("O tempo disponível precisa ser positivo.")
    return carga_servico_h / tempo_disponivel_h * caf + iaf


@dataclass
class Alerta:
    tipo: str
    mensagem: str


def alertas(entregas: pd.DataFrame, atividades: pd.DataFrame, perfis: pd.DataFrame | None = None) -> list[Alerta]:
    """Verificações de consistência da matriz (checklist da Fase 5 do L1, §5.7, mais RACI e menor cargo possível).

    Tipos: ``atividade_sem_perfil``, ``perfil_sem_atividade``, ``perfil_nao_cadastrado``, ``entrega_sem_atividade``,
    ``entrega_fora_do_catalogo``, ``atividade_tempo_zero``, ``raci_sem_A``, ``raci_varios_A``, ``raci_sem_R``,
    ``raci_informado_com_tempo``, ``realocacao_de_nivel`` (atividade não privativa em perfil de complexidade acima
    da menor disponível — verificar; não é conclusão).
    """
    ent = normalizar_entregas(entregas)
    atv = normalizar_atividades(atividades)
    out: list[Alerta] = []
    sem = atv[atv["perfil"] == ""]
    for _, l in sem.iterrows():
        out.append(Alerta("atividade_sem_perfil",
                          f"{l['entrega']} · “{l['atividade']}” não tem perfil: {br(l['tempo_h'] * l['frequencia'])} h"
                          " por unidade ficam fora de todos os Q_r (quadro subestimado)."))
    for e in sorted(set(atv["entrega"]) - set(ent["entrega"])):
        out.append(Alerta("entrega_fora_do_catalogo",
                          f"Entrega {e} aparece nas atividades mas não está no catálogo (demanda tratada como 0)."))
    for e in ent.loc[~ent["entrega"].isin(atv["entrega"]), "entrega"]:
        out.append(Alerta("entrega_sem_atividade",
                          f"Entrega {e} não tem atividade associada: sua carga não é atribuída a nenhum perfil."))
    executa = (atv["raci"] == "") | atv["raci"].str.contains("R") | atv["raci"].str.contains("A")
    for _, l in atv[(atv["tempo_h"] * atv["frequencia"] == 0) & executa].iterrows():
        out.append(Alerta("atividade_tempo_zero",
                          f"{l['entrega']} · “{l['atividade']}” ({l['perfil'] or SEM_PERFIL}) tem tempo zero."))
    if perfis is not None:
        pf = normalizar_perfis(perfis)
        usados = set(atv.loc[atv["perfil"] != "", "perfil"])
        for p in pf.loc[~pf["perfil"].isin(usados), "perfil"]:
            out.append(Alerta("perfil_sem_atividade",
                              f"Perfil “{p}” está cadastrado mas não executa nenhuma atividade: verificar se falta "
                              "vínculo no catálogo ou se é capacidade sem carga (realocação)."))
        for p in sorted(usados - set(pf["perfil"])):
            out.append(Alerta("perfil_nao_cadastrado",
                              f"Perfil “{p}” aparece nas atividades mas não está na tabela de perfis (sem H nem custo)."))
        if pf["complexidade"].notna().any():
            comp = pf.set_index("perfil")["complexidade"]
            com_carga = [p for p in usados if p in comp.index and pd.notna(comp[p])]
            if com_carga:
                minimo = min(comp[p] for p in com_carga)
                cand = atv[(~atv["privativa"]) & atv["perfil"].isin(com_carga)]
                for _, l in cand.iterrows():
                    if comp[l["perfil"]] > minimo:
                        out.append(Alerta("realocacao_de_nivel",
                                          f"{l['entrega']} · “{l['atividade']}” não é privativa e está em "
                                          f"“{l['perfil']}” (complexidade {comp[l['perfil']]:g} > {minimo:g}): "
                                          "verificar o cargo de menor complexidade possível."))
    if (atv["raci"] != "").any():
        for (e, a), g in atv.groupby(["entrega", "atividade"], sort=False):
            n_a = int(g["raci"].str.contains("A").sum())
            if n_a == 0:
                out.append(Alerta("raci_sem_A", f"{e} · “{a}”: nenhum perfil “A” (aprovador único)."))
            elif n_a > 1:
                out.append(Alerta("raci_varios_A", f"{e} · “{a}”: {n_a} perfis “A” — a regra RACI pede um só."))
            if not g["raci"].str.contains("R").any():
                out.append(Alerta("raci_sem_R", f"{e} · “{a}”: ninguém executa (sem R)."))
            for _, l in g[(g["raci"] == "I") & (g["tempo_h"] * g["frequencia"] > 0)].iterrows():
                out.append(Alerta("raci_informado_com_tempo",
                                  f"{e} · “{a}”: “{l['perfil']}” é I (informado) mas tem tempo lançado — "
                                  "confirmar se é C."))
    return out


def analisar(entregas: pd.DataFrame, atividades: pd.DataFrame, perfis: pd.DataFrame,
             regra: str = "teto") -> dict:
    """Roda o fluxo completo e devolve um dicionário com todas as tabelas (sem imprimir nada)."""
    m = matriz_recursos(entregas, atividades, perfis)
    res = {"matriz": m, "por_entrega": horas_por_entrega(m), "entrega_perfil": matriz_entrega_perfil(m),
           "participacao": participacao_perfis(m), "carga": carga_por_perfil(m, perfis, regra),
           "alertas": alertas(entregas, atividades, perfis)}
    res["custo"] = custo_por_entrega(m) if m["custo_hora"].notna().any() else None
    return res


# --------------------------------------------------------------------------------------------------------------
# Relatório, gráfico e linha de comando
# --------------------------------------------------------------------------------------------------------------

def _rs(x: float) -> str:
    return "—" if pd.isna(x) else f"R$ {br(x)}"


def _n(x: float, casas: int = 2) -> str:
    return "—" if pd.isna(x) else br(x, casas)


def relatorio(res: dict, regra: str = "teto") -> str:
    """Relatório legível em português a partir do resultado de ``analisar``."""
    m, pe, carga = res["matriz"], res["por_entrega"], res["carga"]
    L: list[str] = []
    L.append("IDENTIFICAÇÃO DE RECURSOS POR ATIVIDADE — Fase 5 do DFT (dados fictícios)")
    L.append("=" * 100)
    L.append("\n1. Matriz entrega × atividade × perfil (h por unidade = tempo × frequência; h/ano = demanda × h)")
    L.append(f"{'Entrega':<8}{'Atividade':<40}{'Perfil':<26}{'h/unid.':>9}{'h/ano':>12}")
    for _, l in m.iterrows():
        L.append(f"{l['entrega']:<8}{l['atividade'][:39]:<40}{l['perfil'][:25]:<26}{br(l['h_unit'], 3):>9}"
                 f"{br(l['h_ano'], 1):>12}")
    L.append("\n2. Horas-homem por entrega (visão \"h/h por produto\" — Tabela 3 do artigo-base)")
    L.append(f"{'Entrega':<8}{'Nome':<40}{'Demanda':>9}{'h/unid.':>9}{'h/ano':>11}{'% vol.':>8}{'% carga':>9}")
    for _, l in pe.iterrows():
        L.append(f"{l['entrega']:<8}{l['nome'][:39]:<40}{br(l['demanda'], 0):>9}{br(l['h_unit'], 3):>9}"
                 f"{br(l['h_ano'], 1):>11}{br(100 * l['pct_volume'], 0):>7}%{br(100 * l['pct_carga'], 0):>8}%")
    L.append(f"{'':<8}{'Total':<40}{br(pe['demanda'].sum(), 0):>9}{'':>9}{br(pe['h_ano'].sum(), 1):>11}")
    cand = pe[(pe["pct_volume"] >= 2 * pe["pct_carga"]) & (pe["pct_volume"] >= 0.2)]
    for _, l in cand.iterrows():
        L.append(f"   → {l['entrega']}: {br(100 * l['pct_volume'], 0)}% do volume e {br(100 * l['pct_carga'], 0)}% da "
                 "carga — alto volume e baixa complexidade: candidata a automação (estudo próprio).")
    L.append("\n3. Participação de cada perfil no esforço de cada entrega")
    p = res["participacao"]
    L.append(f"{'Entrega':<8}" + "".join(f"{c[:22]:>24}" for c in p.columns))
    for idx, l in p.iterrows():
        L.append(f"{idx:<8}" + "".join(f"{br(100 * v, 1) + '%':>24}" for v in l.values))
    if res.get("custo") is not None:
        c = res["custo"]
        L.append("\n4. Custo por entrega — TDABC (taxa do perfil = custo anual ÷ disponibilidade efetiva H)")
        for _, l in carga.dropna(subset=["custo_hora"]).iterrows():
            L.append(f"   taxa de custo da capacidade — {l['perfil']}: {_rs(l['custo_hora'])}/h")
        L.append(f"{'Entrega':<8}{'Nome':<40}{'Custo unit.':>16}{'Custo anual':>20}{'% custo':>9}")
        for _, l in c.iterrows():
            L.append(f"{l['entrega']:<8}{l['nome'][:39]:<40}{_rs(l['custo_unitario']):>16}"
                     f"{_rs(l['custo_anual']):>20}{br(100 * l['pct_custo'], 0):>8}%")
        L.append(f"{'':<8}{'Total (custo direto de pessoal)':<40}{'':>16}{_rs(c['custo_anual'].sum()):>20}")
        if c["h_sem_custo"].sum() > 0:
            L.append(f"   ! {br(c['h_sem_custo'].sum(), 1)} h/ano sem custo (sem perfil ou perfil sem custo) — fora do total.")
    L.append(f"\n5. Carga por perfil para a consolidação (Q_r = carga_r / H_r; arredondamento declarado: {regra})")
    L.append(f"{'Perfil':<28}{'Carga h/ano':>12}{'H h/ano':>10}{'Q_r':>8}{'Q':>4}{'Lotação':>9}{'Déficit':>9}"
             f"{'Razão WISN':>11}{'Ociosa h':>11}")
    for _, l in carga.iterrows():
        q_txt = "—" if pd.isna(l["q"]) else str(int(l["q"]))
        d_txt = "—" if pd.isna(l["deficit"]) else format(int(l["deficit"]), "+d")
        L.append(f"{l['perfil'][:27]:<28}{_n(l['carga_h'], 1):>12}{_n(l['H'], 1):>10}{_n(l['q_fracionario']):>8}"
                 f"{q_txt:>4}{_n(l['lotacao_atual'], 0):>9}{d_txt:>9}{_n(l['razao_wisn']):>11}"
                 f"{_n(l['ociosa_h'], 1):>11}")
    ok = carga.dropna(subset=["q"])
    L.append(f"{'Total (soma dos Q_r)':<28}{br(carga['carga_h'].sum(), 1):>12}{'':>10}"
             f"{br(ok['q_fracionario'].sum()):>8}{int(ok['q'].sum()):>4}")
    fora = carga.loc[carga["q"].isna(), "carga_h"].sum()
    if fora > 0:
        L.append(f"   ! {br(fora, 1)} h/ano estão em atividade sem perfil ou em perfil não cadastrado e não entram em "
                 "nenhum Q_r — o quadro está subestimado até a correção.")
    hs = ok.loc[ok["carga_h"] > 0, "H"]
    if len(hs) and hs.nunique() > 1:
        h_ref = hs.min()
        qa = carga["carga_h"].sum() / h_ref
        L.append(f"   Comparação: carga total ÷ H único ({br(h_ref, 1)} h) = {br(qa)} → {arredondar(qa, regra)[0]} — "
                 "agregar jornadas distintas é incorreto; vale a soma dos Q_r.")
    if ok["custo_ocioso"].notna().any():
        L.append("   Ociosa < 0 = sobrecarga (carga maior que a capacidade em exercício). Custo da capacidade não "
                 f"utilizada (TDABC): {_rs(ok.loc[ok['custo_ocioso'] > 0, 'custo_ocioso'].sum())}/ano.")
    al = res["alertas"]
    L.append(f"\n6. Alertas ({len(al)})")
    if not al:
        L.append("   Nenhum: toda atividade tem perfil e todo perfil tem atividade.")
    for a in al:
        L.append(f"   [{a.tipo}] {a.mensagem}")
    L.append("\nLembrete: o DFT dimensiona unidades, não avalia servidores (Mendes; Serrano; Meneses, 2022). "
             "Privatividade de atribuição é matéria jurídica.")
    return "\n".join(L)


def grafico(res: dict, caminho: str) -> str:
    """Salva PNG com (a) h/ano por entrega empilhadas por perfil e (b) carga × capacidade em exercício por perfil."""
    plt = estilo_graficos()
    t = res["entrega_perfil"]
    carga = res["carga"].dropna(subset=["H"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    cores = (SERIES + [PALETA["verde"], PALETA["azul_escuro"], PALETA["linha"]]) * 3
    base = np.zeros(len(t))
    for i, col in enumerate(t.columns):
        a1.barh(t.index, t[col].values, left=base, color=cores[i], label=col)
        base += t[col].values
    a1.invert_yaxis()
    a1.set_xlabel("horas por ano")
    a1.set_title("Carga por entrega, por perfil")
    a1.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    y = np.arange(len(carga))
    a2.barh(y - 0.2, carga["carga_h"], height=0.4, color=PALETA["azul"], label="carga (h/ano)")
    if carga["capacidade_h"].notna().any():
        a2.barh(y + 0.2, carga["capacidade_h"].fillna(0), height=0.4, color=PALETA["cinza"],
                label="capacidade em exercício (lotação × H)")
    a2.set_yticks(y, carga["perfil"])
    a2.invert_yaxis()
    a2.set_xlabel("horas por ano")
    a2.set_title("Carga × capacidade por perfil")
    a2.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    formatar_eixos_br(fig)
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def _irmao(caminho: str, papel: str) -> str:
    """Infere o CSV irmão (entregas/perfis) a partir do nome do CSV de atividades."""
    pasta, nome = os.path.split(caminho)
    cand = os.path.join(pasta, nome.replace("atividades", papel).replace("_rascunho", ""))
    if os.path.exists(cand) and cand != caminho:
        return cand
    return os.path.join(pasta, f"recursos_{papel}.csv")


def cli(argv: list[str]) -> int:
    """``dft recursos dados/recursos_atividades.csv [--entregas ...] [--perfis ...] [--arredondamento teto]
    [--saida-carga carga.csv] [--grafico saida.png]``."""
    ap = argparse.ArgumentParser(prog="dft recursos",
                                 description="Matriz entrega × atividade × perfil, horas e custo por perfil e por "
                                             "entrega (TDABC) e carga por perfil para a consolidação.")
    ap.add_argument("atividades", help="CSV de atividades (entrega;atividade;perfil;tempo_h[;frequencia;raci;privativa])")
    ap.add_argument("--entregas", help="CSV de entregas (entrega;nome;demanda). Padrão: arquivo irmão *_entregas.csv")
    ap.add_argument("--perfis", help="CSV de perfis (perfil;jornada_h;dias_efetivos;itp;custo_anual;lotacao_atual). "
                                     "Padrão: arquivo irmão *_perfis.csv")
    ap.add_argument("--arredondamento", choices=REGRAS_ARREDONDAMENTO, default="teto",
                    help="regra declarada na Fase 1 (padrão: teto)")
    ap.add_argument("--saida-carga", help="grava a carga por perfil (CSV ; e vírgula decimal) para a consolidação")
    ap.add_argument("--grafico", help="salva figura PNG")
    a = ap.parse_args(argv)
    try:
        ent = ler_csv(a.entregas or _irmao(a.atividades, "entregas"))
        atv = ler_csv(a.atividades)
        pf = ler_csv(a.perfis or _irmao(a.atividades, "perfis"))
        res = analisar(ent, atv, pf, a.arredondamento)
    except (ErroRecursos, FileNotFoundError) as e:
        print(f"Erro: {e}")
        return 2
    print(relatorio(res, a.arredondamento))
    if a.saida_carga:
        res["carga"].to_csv(a.saida_carga, sep=";", decimal=",", index=False, encoding="utf-8-sig")
        print(f"\nCarga por perfil gravada em {a.saida_carga}")
    if a.grafico:
        print(f"Figura salva em {grafico(res, a.grafico)}")
    return 0
