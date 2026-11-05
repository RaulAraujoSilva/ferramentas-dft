"""Testes do módulo capacidade (mapeamento de processos e análise de capacidade)."""
import math
from pathlib import Path

import pandas as pd
import pytest

from dft_ferramentas import capacidade as cap

RAIZ = Path(__file__).resolve().parents[1]
ATIV = RAIZ / "dados" / "capacidade_atividades.csv"
FLUXO = RAIZ / "dados" / "capacidade_fluxo.csv"


# --- grandezas básicas -----------------------------------------------------------------------------------------

def test_capacidade_e_utilizacao_sao_coerentes():
    c = cap.capacidade_etapa(pessoas=2, horas_disponiveis=77.35, tempo_min=90)
    assert c == pytest.approx(2 * 77.35 * 60 / 90)
    u = cap.utilizacao(demanda=109, tempo_min=90, pessoas=2, horas_disponiveis=77.35)
    assert u == pytest.approx(109 / c)
    assert cap.capacidade_etapa(2, 154.7, 90, horas_totais=True) == pytest.approx(c)


def test_takt_e_tempo_de_ciclo_reproduzem_utilizacao():
    # ρ = tempo de ciclo / takt (com takt sobre o tempo disponível por pessoa)
    takt = cap.takt_time(77.35 * 60, 109)
    ciclo = cap.tempo_ciclo_etapa(90, 2)
    assert ciclo / takt == pytest.approx(cap.utilizacao(109, 90, 2, 77.35))


def test_tempo_invalido():
    with pytest.raises(ValueError):
        cap.capacidade_etapa(1, 10, 0)
    with pytest.raises(ValueError):
        cap.pessoas_necessarias(10, 10, utilizacao_alvo=0)


# --- números canônicos do curso (exemplo de nove passos, dossiê L1) ---------------------------------------------

def test_exemplo_nove_passos_carga_7770_e_Q_8_37():
    demanda = [1200, 3600, 900, 480, 300]
    esforco_h = [1.83, 0.43, 2.50, 1.20, 4.00]
    carga = sum(cap.carga_horas(d, e * 60) for d, e in zip(demanda, esforco_h))
    assert carga == pytest.approx(7770, abs=0.5)
    H = 221 * 6 * 0.70
    assert H == pytest.approx(928.2)
    assert round(cap.pessoas_necessarias(7770, H), 2) == 8.37
    # com o quadro atual de 7 servidores a unidade operaria a ~119,6% da capacidade efetiva
    assert 7770 / (7 * H) == pytest.approx(1.1958, abs=1e-4)


# --- tipos de capacidade ---------------------------------------------------------------------------------------

def test_capacidades_teorica_efetiva_demonstrada():
    c = cap.capacidades(pessoas=2, horas_brutas_pessoa=110.5, tempo_min=90, fator_disponibilidade=0.70)
    assert c["teorica"] == pytest.approx(2 * 110.5 * 60 / 90)
    assert c["efetiva"] == pytest.approx(c["teorica"] * 0.70)
    dem = cap.capacidade_demonstrada([98, 102, 100, 96])
    assert dem == pytest.approx(99)
    ind = cap.indicadores_capacidade(dem, c["teorica"], c["efetiva"])
    assert ind["eficiencia_sobre_efetiva"] > ind["utilizacao_sobre_teorica"]
    with pytest.raises(ValueError):
        cap.capacidade_demonstrada([])


# --- lei de Little ---------------------------------------------------------------------------------------------

def test_little_resolve_qualquer_grandeza():
    assert cap.little(vazao=60, lead_time=3) == 180
    assert cap.little(wip=180, lead_time=3) == 60
    assert cap.little(wip=180, vazao=60) == 3
    with pytest.raises(ValueError):
        cap.little(wip=1)
    assert cap.eficiencia_ciclo(4, 160) == pytest.approx(0.025)


# --- filas ----------------------------------------------------------------------------------------------------

def test_mm1_exato():
    lam, mu = 3.0, 4.0
    r = cap.mmc(lam, mu, 1)
    assert r["p_espera"] == pytest.approx(0.75)
    assert r["Wq"] == pytest.approx(0.75 / (mu - lam))
    assert r["L"] == pytest.approx(lam * r["W"])  # Little dentro do sistema


def test_erlang_c_confere_com_formula_direta():
    c, a = 5, 4.0
    soma = sum(a ** k / math.factorial(k) for k in range(c))
    termo = a ** c / math.factorial(c) * c / (c - a)
    assert cap.erlang_c(c, a) == pytest.approx(termo / (soma + termo), rel=1e-12)
    assert cap.erlang_c(3, 3.0) == 1.0


def test_kingman_exato_no_mm1_e_proximo_no_mmc():
    te = 1 / 4
    assert cap.kingman_espera(0.75, 1, 1, te) == pytest.approx(cap.mmc(3, 4, 1)["Wq"])
    exato = cap.mmc(lam=4.0, mu=1.0, c=5)["Wq"]
    aprox = cap.kingman_espera(0.8, 1, 1, 1.0, c=5)
    assert aprox == pytest.approx(exato, rel=0.10)
    assert cap.kingman_espera(1.0, 1, 1, 1) == math.inf
    # a espera cresce de forma não linear com a utilização
    assert cap.kingman_espera(0.95, 1, 1, 1) > 4 * cap.kingman_espera(0.8, 1, 1, 1)


def test_servidores_para_meta_de_nivel_de_servico():
    lam, mu, meta = 30 / 60, 1 / 8, 10.0  # 30 atendimentos/h, 8 min cada, meta 10 min (tempo em minutos)
    c = cap.servidores_para_meta(lam, mu, meta, nivel=0.8)
    assert cap.nivel_servico(lam, mu, c, meta) >= 0.8
    assert cap.nivel_servico(lam, mu, c - 1, meta) < 0.8
    assert c > lam / mu


# --- do fluxo à carga -------------------------------------------------------------------------------------------

def test_retrabalho_xor_and_pelas_visitas():
    v = cap.razao_visitas([("INICIO", "A", 1), ("A", "A", 0.2), ("A", "FIM", 0.8)])
    assert v["A"] == pytest.approx(1 / 0.8)
    assert cap.tempo_retrabalho(30, 0.2) == pytest.approx(30 * v["A"])
    v = cap.razao_visitas([("INICIO", "A", 1), ("A", "B", 0.3), ("A", "C", 0.7), ("B", "FIM", 1), ("C", "FIM", 1)])
    t = {"A": 10, "B": 50, "C": 20}
    assert cap.esforco_unitario(v, t) * 60 == pytest.approx(10 + cap.tempo_xor([50, 20], [0.3, 0.7]))
    v = cap.razao_visitas([("INICIO", "A", 1), ("A", "B", 1), ("A", "C", 1), ("B", "D", 1), ("C", "D", 0.5 * 2),
                           ("D", "FIM", 1)])
    assert v["B"] == v["C"] == 1
    assert cap.tempo_and([30, 40]) == {"duracao": 40, "trabalho": 70}
    with pytest.raises(ValueError):
        cap.tempo_xor([1, 2], [0.5, 0.6])


def test_fluxo_ficticio_conserva_casos():
    fluxo = cap.ler_fluxo(FLUXO)
    assert cap.validar_fluxo(fluxo) == []
    v = cap.razao_visitas(fluxo)
    assert v["A2"] == pytest.approx(1 / (1 - 0.8 * (0.15 + 0.85 * 0.15)))
    assert v["A6"] + 0.2 * v["A7"] == pytest.approx(1.0)  # todo caso que entra sai por um dos fins
    tab = cap.carga_por_atividade(v, {k: 1 for k in v}, 100)
    assert tab.set_index("atividade").loc["A1", "demanda"] == pytest.approx(100)


def test_laco_sem_saida_e_avisos():
    with pytest.raises(ValueError):
        cap.razao_visitas([("INICIO", "A", 1), ("A", "B", 1), ("B", "A", 1)])
    avisos = cap.validar_fluxo([("INICIO", "A", 1), ("A", "B", 0.5), ("X", "FIM", 1)])
    texto = " | ".join(avisos)
    assert "somam 0.500" in texto and "não tem saída" in texto and "'X' não é alcançável" in texto


# --- análise tabular e CLI --------------------------------------------------------------------------------------

def test_analise_com_csv_e_fluxo():
    ativ = cap.ler_atividades(ATIV)
    tab = cap.analisar_capacidade(ativ, fluxo=cap.ler_fluxo(FLUXO), demanda_processo=100)
    g = tab.loc[tab["gargalo"]].iloc[0]
    assert g["id"] == "A3"
    assert g["utilizacao"] == pytest.approx(1.0593, abs=1e-3)
    assert g["situacao"] == "acima da capacidade"
    res = cap.resumo_processo(tab, 100)
    assert res["capacidade_processo"] == pytest.approx(100 / g["utilizacao"])
    assert res["vazao"] == pytest.approx(94.4, abs=0.05)
    assert res["lead_time"] == pytest.approx(116 / res["vazao"])
    pp = cap.por_perfil(tab).set_index("perfil")
    assert pp.loc["Arquiteto/engenheiro", "utilizacao"] == pytest.approx(0.906, abs=1e-3)
    assert "GARGALO" in cap.relatorio(tab, res)


def test_analise_sem_fluxo_usa_coluna_demanda():
    tab = cap.analisar_capacidade(cap.ler_atividades(ATIV), utilizacao_alvo=0.85)
    a3 = tab.set_index("id").loc["A3"]
    assert a3["pessoas_necessarias"] == pytest.approx(109 * 90 / 60 / (77.35 * 0.85))
    with pytest.raises(ValueError):
        cap.analisar_capacidade(pd.DataFrame({"atividade": ["x"], "tempo_min": [1], "pessoas": [1],
                                              "horas_disponiveis": [1]}))


def test_cli(tmp_path, capsys):
    png = tmp_path / "cap.png"
    rc = cap.cli([str(ATIV), "--fluxo", str(FLUXO), "--demanda", "100", "--dias-periodo", "22", "--grafico", str(png)])
    saida = capsys.readouterr().out
    assert rc == 0 and png.exists()
    assert "Razão de visitas" in saida and "Lei de Little" in saida
    assert cap.cli([str(ATIV), "--fluxo", str(FLUXO)]) == 2
