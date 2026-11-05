"""Testes do módulo 3 — nivelamento de recursos e Teoria das Restrições."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from dft_ferramentas._comum import ler_csv
from dft_ferramentas.nivelamento import (
    Atividade, Etapa, analise_fluxo, atividades_de_dataframe, caminho_critico, cenarios_elevar, cenarios_realocar,
    cli, cpm, dimensionar_pela_restricao, efeito_subordinar, etapas_de_dataframe, explorar, indicadores_perfil,
    nivelar, ordem_topologica, perfil_recursos, pulmao_projeto, pulmao_tambor, servidores_necessarios,
    soma_quadrados, suavizar, vazao_sistema,
)

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
CAPS = {"Fiscal": 4.0, "Administrativo": 2.0}
H = 28.0
DEMANDA = 28.0


@pytest.fixture(scope="module")
def ativs():
    return atividades_de_dataframe(ler_csv(DADOS / "nivelamento_atividades.csv"))


@pytest.fixture(scope="module")
def etapas():
    return etapas_de_dataframe(ler_csv(DADOS / "nivelamento_fluxo.csv"))


# ------------------------------------------------------------------ Parte A
def test_ciclo_e_predecessora_desconhecida():
    a = [Atividade("A", "a", 1, "R", 1, ("B",)), Atividade("B", "b", 1, "R", 1, ("A",))]
    with pytest.raises(ValueError, match="ciclo"):
        ordem_topologica(a)
    with pytest.raises(ValueError, match="desconhecida"):
        ordem_topologica([Atividade("A", "a", 1, "R", 1, ("Z",))])


def test_cpm_do_exemplo(ativs):
    t = cpm(ativs)
    assert int(t["EF"].max()) == 13
    assert caminho_critico(ativs, t) == ["A01", "A02", "A06", "A08", "A10"]
    assert t.loc["A03", "folga_total"] == 2 and t.loc["A05", "folga_total"] == 4
    assert t.loc["A09", "ES"] == 4 and t.loc["A09", "LF"] == 8          # janela: semanas 5 a 8
    assert set(t.index[t["critica"]]) == {"A01", "A02", "A06", "A08", "A10"}


def test_prazo_inviavel_gera_folga_negativa(ativs):
    t = cpm(ativs, prazo=12)
    assert t["folga_total"].min() == -1
    with pytest.raises(ValueError, match="inviáveis"):
        suavizar(ativs, prazo=12)


def test_burgess_caso_de_mao():
    """A e B (2 semanas, 2 servidores cada) cabem em sequência dentro da folga dada por X (4 semanas)."""
    a = [Atividade("X", "x", 4, "Outro", 1), Atividade("A", "a", 2, "R", 2), Atividade("B", "b", 2, "R", 2)]
    cedo = {i: int(v) for i, v in cpm(a)["ES"].items()}
    s = suavizar(a)
    p0, p1 = perfil_recursos(a, cedo)["R"], perfil_recursos(a, s)["R"]
    assert list(p0) == [4, 4, 0, 0] and soma_quadrados(p0) == 32
    assert list(p1) == [2, 2, 2, 2] and soma_quadrados(p1) == 16
    assert s == {"X": 0, "A": 0, "B": 2}


def test_suavizacao_mantem_prazo_e_reduz_pico(ativs):
    t = cpm(ativs)
    cedo = {i: int(v) for i, v in t["ES"].items()}
    s = suavizar(ativs)
    p0, p1 = perfil_recursos(ativs, cedo), perfil_recursos(ativs, s)
    assert len(p1) == 13                                                   # prazo intocado
    assert soma_quadrados(p0) == 184 and soma_quadrados(p1) == 134
    assert p0["Fiscal"].max() == 8 and p1["Fiscal"].max() == 5
    assert math.isclose(p0["Fiscal"].sum(), p1["Fiscal"].sum())            # área (carga) preservada
    m = {a.id: a for a in ativs}
    for a in ativs:                                                         # precedências e janelas
        assert s[a.id] >= a.liberacao
        for p in a.predecessoras:
            assert s[a.id] >= s[p] + m[p].duracao
        if a.fim_max is not None:
            assert s[a.id] + a.duracao <= a.fim_max


def test_nivelamento_respeita_capacidade_e_precedencia(ativs):
    n = nivelar(ativs, CAPS)
    p = perfil_recursos(ativs, n)
    assert len(p) == 14                                                    # +1 semana em relação ao CPM
    ind = indicadores_perfil(p, CAPS)
    assert (ind["semanas_acima"] == 0).all()
    assert p["Fiscal"].max() == 4
    m = {a.id: a for a in ativs}
    for a in ativs:
        for q in a.predecessoras:
            assert n[a.id] >= n[q] + m[q].duracao
    assert n["A04"] == 6 and n["A10"] == 12                                # A04 espera A03; A10 atrasa 1 semana


def test_nivelamento_sem_conflito_igual_ao_cpm(ativs):
    cedo = {i: int(v) for i, v in cpm(ativs)["ES"].items()}
    assert nivelar(ativs, {"Fiscal": 100, "Administrativo": 100}) == cedo


def test_nivelamento_carga_acima_da_capacidade(ativs):
    with pytest.raises(ValueError, match="acima da capacidade"):
        nivelar(ativs, {"Fiscal": 2.0})
    with pytest.raises(ValueError):
        nivelar(ativs, CAPS, regra="inexistente")


@pytest.mark.parametrize("regra", ["folga", "lft", "lst"])
def test_regras_de_prioridade_validas(ativs, regra):
    p = perfil_recursos(ativs, nivelar(ativs, CAPS, regra))
    assert p["Fiscal"].max() <= 4 and p["Administrativo"].max() <= 2


def test_pulmao_projeto_corrente_critica():
    seg = [2, 2, 4, 3, 2]
    c = pulmao_projeto(seg, metodo="corte")
    assert c["corrente_agressiva"] == 6.5 and c["pulmao"] == 3.25 and c["prazo_com_pulmao"] == 9.75
    r = pulmao_projeto(seg, metodo="rss")
    assert math.isclose(r["pulmao"], math.sqrt(9.25))
    assert r["pulmao"] < c["pulmao"]                                       # RSS < 50 % (Herroelen e Leus, 2001)


# ------------------------------------------------------------------ Parte B
def test_identificar_restricao(etapas):
    tab, r = analise_fluxo(etapas, H, DEMANDA)
    caps = tab["capacidade"].round(3).to_dict()
    assert caps == {"Protocolo e autuação": 93.333, "Análise documental": 42.0, "Análise técnica do projeto": 23.333,
                    "Vistoria in loco": 31.111, "Emissão do alvará": 35.0}
    assert r["restricao"] == "Análise técnica do projeto"
    assert math.isclose(r["vazao"], 70 / 3) and math.isclose(r["acumulo_semanal"], 28 - 70 / 3)
    assert math.isclose(tab.loc["Análise técnica do projeto", "utilizacao"], 1.2)
    assert not r["restricao_externa"]


def test_restricao_externa_quando_demanda_baixa(etapas):
    _, r = analise_fluxo(etapas, H, 20.0)
    assert r["restricao_externa"] and r["vazao"] == 20.0 and r["acumulo_semanal"] == 0


def test_explorar_transfere_trabalho(etapas):
    e = explorar(etapas, "Análise técnica do projeto", 0.15, "Análise documental")
    m = {x.nome: x for x in e}
    assert math.isclose(m["Análise técnica do projeto"].tempo_h, 3.06)
    assert math.isclose(m["Análise documental"].tempo_h, 2.54)
    assert math.isclose(vazao_sistema(e, H, DEMANDA), 84 / 3.06)            # 27,45 casos/semana
    with pytest.raises(ValueError):
        explorar(etapas, "Análise técnica do projeto", 1.2)


def test_elevar_so_na_restricao_aumenta_vazao(etapas):
    tab = cenarios_elevar(etapas, H, DEMANDA, custo_servidor=3200, ganho_por_caso=900)
    ganhos = tab["delta_vazao"].drop("atual")
    assert ganhos.idxmax() == "+1 em Análise técnica do projeto"
    assert math.isclose(ganhos.max(), 28 - 70 / 3)
    assert (ganhos.drop("+1 em Análise técnica do projeto") == 0).all()     # não-restrições: ΔX = 0
    assert math.isclose(tab.loc["+1 em Análise técnica do projeto", "delta_T_menos_OE"], 1000.0)
    assert math.isclose(tab.loc["+1 em Emissão do alvará", "delta_T_menos_OE"], -3200.0)


def test_realocacao_pode_criar_nova_restricao(etapas):
    r = cenarios_realocar(etapas, H, DEMANDA)
    assert list(r["origem"]) == ["Vistoria in loco"]
    assert math.isclose(r.iloc[0]["vazao"], 28 / 1.8) and r.iloc[0]["nova_restricao"] == "Vistoria in loco"


def test_numero_canonico_nove_passos():
    """CT do exemplo de nove passos: H = 221 × 6 × 0,70 = 928,2 h/ano; Q = 7.770 / 928,2 = 8,37 → 9 (teto)."""
    Hano = 221 * 6 * 0.70
    assert math.isclose(Hano, 928.2)
    q = servidores_necessarios(7770, Hano)
    assert round(q, 2) == 8.37 and math.ceil(q) == 9
    assert servidores_necessarios(7770, Hano, 0.85) > q


def test_dimensionar_pela_restricao(etapas):
    e = explorar(etapas, "Análise técnica do projeto", 0.15, "Análise documental")
    d = dimensionar_pela_restricao(e, H, DEMANDA, "Análise técnica do projeto", 0.95, 0.85)
    assert d["Q"].to_dict() == {"Protocolo e autuação": 1, "Análise documental": 3, "Análise técnica do projeto": 4,
                                "Vistoria in loco": 3, "Emissão do alvará": 1}
    assert int(d["Q"].sum()) == 12 and d["atual"].sum() == 11
    assert math.isclose(d.loc["Análise técnica do projeto", "Q_fracionario"], 28 * 3.06 / (28 * 0.95))


def test_pulmao_tambor_e_little(etapas):
    p = pulmao_tambor(etapas, "Análise técnica do projeto", 27.45, fator=0.5, z=1.645)
    assert p["lead_time_montante"] == 5.0 and p["pulmao_regra_dias"] == 2.5
    assert math.isclose(p["pulmao_regra_casos"], 2.5 * 27.45 / 5)           # Little: L = λ·W
    assert math.isclose(p["sigma_lt"], math.sqrt(0.5 ** 2 + 2.4 ** 2))
    assert math.isclose(p["pulmao_estatistico_dias"], 1.645 * math.sqrt(6.01))
    assert math.isclose(p["zona_vermelha_abaixo_de"], p["pulmao_regra_casos"] / 3)


def test_efeito_subordinar():
    s = efeito_subordinar(28, 70 / 3, 12, 10)
    assert math.isclose(s.loc[12, "fila_entrada_com_corda"], 56.0)
    assert s["wip_com_corda"].nunique() == 1 and s["wip_sem_corda"].is_monotonic_increasing
    assert math.isclose(s.loc[12, "atravessamento_sem_corda_sem"], 66 / (70 / 3))


# ------------------------------------------------------------------ CLI
def test_cli_recursos_e_toc(tmp_path, capsys):
    g1, g2 = tmp_path / "r.png", tmp_path / "t.png"
    assert cli([str(DADOS / "nivelamento_atividades.csv"), "--capacidades", str(DADOS / "nivelamento_capacidades.csv"),
                "--grafico", str(g1)]) == 0
    out = capsys.readouterr().out
    assert "Caminho crítico: A01 → A02 → A06 → A08 → A10" in out and g1.exists()
    assert cli([str(DADOS / "nivelamento_fluxo.csv"), "--destino-explorar", "Análise documental",
                "--custo-servidor", "3200", "--ganho-por-caso", "900", "--grafico", str(g2)]) == 0
    out = capsys.readouterr().out
    assert "Restrição interna: Análise técnica do projeto" in out and "PASSO 5" in out and g2.exists()
    assert cli([str(DADOS / "nivelamento_atividades.csv"), "--capacidade", "Fiscal=2"]) == 1
