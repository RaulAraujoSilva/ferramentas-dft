"""Testes do módulo simulacao — números canônicos do curso (CT-1, CT-2, CT-3) e verificação/validação do simulador."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from dft_ferramentas import simulacao as sim

RAIZ = Path(__file__).resolve().parents[1]
CT1 = sim.TresPontos(4, 6, 16, "beta_pert")
CT2 = sim.TresPontos(4, 6, 16, "lognormal_deslocada")


# ---------------------------------------------------------------- CT-1 / CT-2: quantis analíticos (L2, §6)

def test_ct1_parametros_vose_e_quantis_analiticos():
    al, be = sim._beta_pert_parametros(4, 6, 16)
    assert al == pytest.approx(1.666667, abs=1e-6)
    assert be == pytest.approx(4.333333, abs=1e-6)
    assert sim.quantil_analitico(CT1, 0.80) == pytest.approx(9.074998, abs=1e-6)
    assert sim.quantil_analitico(CT1, 0.50) == pytest.approx(7.023448, abs=1e-6)
    assert sim.quantil_analitico(CT1, 0.90) == pytest.approx(10.223219, abs=1e-6)
    assert CT1.media() == pytest.approx(7.333333, abs=1e-6)
    assert CT1.desvio() == pytest.approx(2.031498, abs=1e-6)


def test_ct2_lognormal_deslocada_p80():
    lam, mu, s = sim._lognormal_deslocada_parametros(4, 6, 16)
    assert (lam, mu, s) == pytest.approx((1.545912, 1.784366, 0.539021), abs=1e-6)
    assert sim.quantil_analitico(CT2, 0.80) == pytest.approx(10.920644, abs=1e-6)


def test_ct1_erro_padrao_teorico_do_p80():
    f = float(CT1.densidade(9.074998))
    assert f == pytest.approx(0.107836, abs=1e-6)
    assert sim.erro_padrao_quantil(0.8, 10_000, f) == pytest.approx(0.0371, abs=1e-4)


# ---------------------------------------------------------------- CT-1: P80 simulado converge dentro do erro de MC

def test_ct1_p80_simulado_dentro_do_erro_de_monte_carlo():
    e = sim.Entrega("Parecer técnico de licitação", sim.TresPontos(1, 1, 1), CT1)
    res = sim.simular_esforco([e], n=10_000, semente=42)
    se = sim.erro_padrao_quantil(0.8, 10_000, float(CT1.densidade(9.074998)))
    assert abs(res.p80_total - 9.074998) <= 0.10            # assertiva do L2, CT-1, passo 4
    assert abs(res.p80_total - 9.074998) <= 3 * se
    lo, hi = sim.ic_quantil_ordem(res.total, 0.8)
    assert lo <= 9.074998 <= hi
    em = sim.erro_mc_p80(res)
    assert em["erro_padrao"] == pytest.approx(se, rel=0.15)  # SE com densidade por núcleo ≈ SE teórico


def test_convergencia_erro_cai_com_raiz_de_n():
    f = float(CT1.densidade(9.074998))
    e = sim.Entrega("p", sim.TresPontos(1, 1, 1), CT1)
    res = sim.simular_esforco([e], n=100_000, semente=7)
    assert abs(res.p80_total - 9.074998) <= 3 * sim.erro_padrao_quantil(0.8, 100_000, f)
    assert sim.erro_padrao_quantil(0.8, 100_000, f) == pytest.approx(
        sim.erro_padrao_quantil(0.8, 10_000, f) / math.sqrt(10), rel=1e-12)
    assert sim.n_para_precisao_quantil(0.8, f, 0.0727) == pytest.approx(10_000, rel=0.01)


def test_cobertura_do_ic_por_estatisticas_de_ordem():
    g = np.random.default_rng(123)
    cobre = 0
    for _ in range(200):
        x = CT1.quantil(g.random(2000))
        lo, hi = sim.ic_quantil_ordem(x, 0.8, 0.95)
        cobre += lo <= 9.074998 <= hi
    assert 0.90 <= cobre / 200 <= 0.99


def test_reprodutibilidade_pela_semente():
    e = [sim.Entrega.simples("a", 100, 4, 6, 16), sim.Entrega.simples("b", 50, 1, 2, 5)]
    r1 = sim.simular_esforco(e, n=2000, semente=99, correlacao=0.5)
    r2 = sim.simular_esforco(e, n=2000, semente=99, correlacao=0.5)
    assert np.array_equal(r1.total, r2.total)


# ---------------------------------------------------------------- CT-3: quadro e servidores

def test_ct3_soma_dos_p80_e_servidores():
    entregas = sim.ler_entregas(RAIZ / "dados" / "simulacao_ct3.csv")
    carga = sim.soma_p80_analitica(entregas)
    assert carga == pytest.approx(28432.5106, abs=1e-3)
    q, qi = sim.servidores(carga, sim.DISPONIBILIDADE_CT3)
    assert q == pytest.approx(24.646767, abs=1e-6) and qi == 25
    res = sim.simular_esforco(entregas, n=10_000, semente=42)
    se = 2637.1918 * sim.erro_padrao_quantil(0.8, 10_000, float(CT1.densidade(9.074998)))
    assert abs(res.p80_total - carga) <= 3 * se


def test_exemplo_nove_passos_disponibilidade():
    q, qi = sim.servidores(7770, sim.DISPONIBILIDADE_NOVE_PASSOS)
    assert q == pytest.approx(8.37, abs=0.005) and qi == 9


# ---------------------------------------------------------------- carteira: soma dos P80 × P80 da soma

def _carteira():
    return [sim.Entrega.simples("a", 1000, 4, 6, 16), sim.Entrega.simples("b", 800, 2, 3, 9),
            sim.Entrega.simples("c", 500, 5, 8, 20, "lognormal_deslocada")]


def test_soma_dos_p80_maior_que_p80_da_soma_sem_correlacao():
    res = sim.simular_esforco(_carteira(), n=20_000, semente=1)
    assert res.soma_p80_individuais > res.p80_total * 1.03


def test_correlacao_perfeita_torna_quantis_aditivos():
    res = sim.simular_esforco(_carteira(), n=20_000, semente=1, correlacao=1.0)
    assert res.p80_total == pytest.approx(res.soma_p80_individuais, rel=0.01)
    r0 = sim.simular_esforco(_carteira(), n=20_000, semente=1, correlacao=0.0)
    r5 = sim.simular_esforco(_carteira(), n=20_000, semente=1, correlacao=0.5)
    assert r0.p80_total < r5.p80_total < res.p80_total


@pytest.mark.parametrize("metodo", ["copula", "iman_conover"])
def test_correlacao_de_postos_induzida(metodo):
    res = sim.simular_esforco(_carteira(), n=20_000, semente=5, correlacao=0.6, metodo_correlacao=metodo)
    rho = stats.spearmanr(res.horas[:, 0], res.horas[:, 1])[0]
    assert rho == pytest.approx(0.6, abs=0.03)


def test_iman_conover_preserva_marginais():
    g = np.random.default_rng(3)
    x = np.column_stack([g.normal(size=5000), g.exponential(size=5000)])
    y = sim.iman_conover(x, 0.7, g)
    assert np.allclose(np.sort(x, axis=0), np.sort(y, axis=0))


def test_agregacao_independente_reduz_dispersao():
    e = [sim.Entrega.simples("Parecer", 2637.1918, 4, 6, 16)]
    ind = sim.simular_esforco(e, n=10_000, semente=2, agregacao="independente")
    assert ind.percentis[50] == pytest.approx(2637.1918 * CT1.media(), rel=0.005)
    assert ind.p80_total < 2637.1918 * 9.074998 * 0.85


def test_tornado_ordena_pela_maior_influencia():
    res = sim.simular_esforco(_carteira(), n=10_000, semente=4)
    linhas = sim.tornado(res)
    maior = res.nomes[int(np.argmax(res.horas.std(axis=0)))]
    assert linhas[0]["entrega"] == maior
    assert linhas[0]["rho_postos"] == max(l["rho_postos"] for l in linhas)
    assert sum(l["contribuicao_variancia"] for l in linhas) == pytest.approx(1.0)


def test_tres_pontos_invalidos():
    with pytest.raises(ValueError):
        sim.TresPontos(5, 4, 10).quantil(0.5)
    with pytest.raises(ValueError):
        sim._lognormal_deslocada_parametros(4, 9.9, 16)   # moda alta demais: sem raiz com λ ≥ 0


# ---------------------------------------------------------------- simulação de eventos discretos: V&V (Sargent)

def _mm2():
    return [sim.Etapa("Atendimento", 2, sim.TresPontos(1, 1, 1, "exponencial"))]


def test_validacao_contra_erlang_c():
    an = sim.erlang_c(1.5, 1.0, 2)
    assert an["wq"] == pytest.approx(1.285714, abs=1e-6)     # M/M/2, ρ = 0,75
    r = sim.simular_processo(_mm2(), 1.5, horizonte=4000, aquecimento=200, replicacoes=10, semente=11).resumo()
    media, meia = r["espera_media"][0]
    assert abs(media - an["wq"]) <= max(3 * meia, 0.1 * an["wq"])
    assert r["utilizacao"][0][0] == pytest.approx(0.75, abs=0.02)


def test_lei_de_little():
    etapas = sim.ler_etapas(RAIZ / "dados" / "simulacao_processo.csv")
    r = sim.simular_processo(etapas, 1.3, horizonte=1500, aquecimento=150, replicacoes=5, semente=3).resumo()
    assert r["little_l"] == pytest.approx(r["little_lambda_w"], rel=0.05)


def test_mais_servidores_melhora_nivel_de_servico():
    etapas = sim.ler_etapas(RAIZ / "dados" / "simulacao_processo.csv")
    linhas = sim.varrer_servidores(etapas, 1.3, None, [2, 3, 4], horizonte=1000, aquecimento=100,
                                   replicacoes=4, semente=8, prazo=8)
    assert linhas[0]["etapa"] == "Análise técnica"
    w = [l["resumo"]["w_medio"][0] for l in linhas]
    u = [l["resumo"]["utilizacao"][1][0] for l in linhas]
    assert w[0] > w[1] > w[2] and u[0] > u[1] > u[2]
    assert linhas[1]["resumo"]["nivel_servico"][0] > linhas[0]["resumo"]["nivel_servico"][0]


def test_carga_teorica_com_retrabalho_bate_com_utilizacao_simulada():
    etapas = sim.ler_etapas(RAIZ / "dados" / "simulacao_processo.csv")
    etapas = [sim.Etapa(e.nome, 3, e.tempo, e.prob_retrabalho, e.volta_para) for e in etapas]
    carga = sim.carga_por_etapa(etapas, 1.3)
    r = sim.simular_processo(etapas, 1.3, horizonte=3000, aquecimento=200, replicacoes=5, semente=9).resumo()
    for i, c in enumerate(carga):
        assert r["utilizacao"][i][0] == pytest.approx(c / 3, rel=0.05)


@pytest.mark.skipif(importlib.util.find_spec("simpy") is None, reason="SimPy não instalado (opcional)")
def test_motor_proprio_e_simpy_dao_o_mesmo_resultado():
    etapas = sim.ler_etapas(RAIZ / "dados" / "simulacao_processo.csv")
    kw = dict(horizonte=500, aquecimento=50, replicacoes=2, semente=21)
    a = sim.simular_processo(etapas, 1.3, motor="proprio", **kw)
    b = sim.simular_processo(etapas, 1.3, motor="simpy", **kw)
    for ra, rb in zip(a.replicacoes, b.replicacoes):
        assert np.allclose(ra["atravessamento"], rb["atravessamento"], atol=1e-9)
        assert ra["fila_media"] == pytest.approx(rb["fila_media"], rel=1e-9)


# ---------------------------------------------------------------- linha de comando

def test_cli_esforco_e_fila(tmp_path, capsys):
    png = tmp_path / "e.png"
    assert sim.cli([str(RAIZ / "dados" / "simulacao_entregas.csv"), "--n", "2000", "--grafico", str(png)]) == 0
    assert png.exists() and "P80" in capsys.readouterr().out
    png2 = tmp_path / "f.png"
    assert sim.cli([str(RAIZ / "dados" / "simulacao_processo.csv"), "--servidores", "2-3", "--horizonte", "400",
                    "--aquecimento", "40", "--replicacoes", "3", "--prazo", "8", "--grafico", str(png2)]) == 0
    assert png2.exists() and "Little" in capsys.readouterr().out
