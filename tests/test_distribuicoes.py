"""Testes do módulo 2.1 — distribuições de tempo (inclui os números canônicos CT-1 e CT-2 do curso)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from dft_ferramentas import distribuicoes as dd

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"


# ---------------------------------------------------------------- CT-1 (L2 §6): beta-PERT ---------------------------
def test_ct1_beta_pert_vose():
    bp = dd.beta_pert(4, 6, 16)
    assert bp.media_pert == pytest.approx(7.333333, abs=1e-6)
    assert bp.dp_pert == pytest.approx(2.0, abs=1e-12)
    assert bp.alfa == pytest.approx(1.666667, abs=1e-6)
    assert bp.beta == pytest.approx(4.333333, abs=1e-6)
    assert bp.quantil(0.50) == pytest.approx(7.023448, abs=1e-6)
    assert bp.quantil(0.80) == pytest.approx(9.074998, abs=1e-6)
    assert bp.quantil(0.90) == pytest.approx(10.223219, abs=1e-6)
    assert bp.quantil(0.95) == pytest.approx(11.151745, abs=1e-6)
    assert bp.media() == pytest.approx(7.333333, abs=1e-6)
    assert bp.desvio() == pytest.approx(2.031498, abs=1e-6)
    assert bp.cdf(8.0) == pytest.approx(0.661989, abs=1e-6)


def test_ct1_beta_pert_classico():
    bc = dd.beta_pert(4, 6, 16, "classico")
    assert bc.alfa == pytest.approx(1.728395, abs=1e-6)
    assert bc.beta == pytest.approx(4.493827, abs=1e-6)
    assert bc.quantil(0.5) == pytest.approx(7.034867, abs=1e-6)
    assert bc.quantil(0.8) == pytest.approx(9.045662, abs=1e-6)


def test_beta_pert_simetrica_caso_especial():
    bp = dd.beta_pert(2, 5, 8)
    assert bp.alfa == pytest.approx(3.0) and bp.beta == pytest.approx(3.0)


def test_ct1_erro_padrao_monte_carlo():
    bp = dd.beta_pert(4, 6, 16)
    assert bp.pdf(bp.quantil(0.8)) == pytest.approx(0.107836, abs=1e-6)
    assert dd.erro_padrao_quantil(bp, 0.8, 10_000) == pytest.approx(0.0371, abs=1e-4)


def test_ct1_monte_carlo_dentro_da_tolerancia():
    bp = dd.beta_pert(4, 6, 16)
    amostra = bp.dist.rvs(size=10_000, random_state=np.random.default_rng(42))
    assert abs(np.quantile(amostra, 0.8) - 9.074998) <= 0.10


# ---------------------------------------------------------------- CT-2 (L2 §6): log-normal deslocada --------------
def test_ct2_lognormal_deslocada():
    ln = dd.lognormal_deslocada_tres_pontos(4, 6, 16, p=0.05)
    assert ln.lam == pytest.approx(1.545912, abs=1e-6)
    assert ln.mu == pytest.approx(1.784366, abs=1e-6)
    assert ln.sigma == pytest.approx(0.539021, abs=1e-6)
    z = stats.norm.ppf(0.95)
    assert ln.lam + math.exp(ln.mu - z * ln.sigma) == pytest.approx(4.0, abs=1e-9)
    assert ln.lam + math.exp(ln.mu + z * ln.sigma) == pytest.approx(16.0, abs=1e-9)
    assert ln.lam + math.exp(ln.mu - ln.sigma ** 2) == pytest.approx(6.0, abs=1e-9)
    assert ln.quantil(0.50) == pytest.approx(7.501716, abs=1e-6)
    assert ln.quantil(0.80) == pytest.approx(10.920644, abs=1e-6)
    assert ln.quantil(0.90) == pytest.approx(13.429399, abs=1e-6)
    assert ln.quantil(0.95) == pytest.approx(16.0, abs=1e-6)
    assert ln.media() == pytest.approx(8.432930, abs=1e-6)


def test_ct2_recusa_quando_nao_ha_raiz():
    with pytest.raises(dd.ErroAjuste):
        dd.lognormal_deslocada_tres_pontos(2, 5, 8)


def test_resumo_d3_delta_e_horas():
    r = dd.resumo_tres_pontos(4, 6, 16)
    assert r["delta_p80"] == pytest.approx(0.203378, abs=1e-5)
    horas = (r["lognormal"].quantil(0.8) - r["beta_pert"].quantil(0.8)) * 2637.1918
    assert horas == pytest.approx(4867.3, abs=0.5)
    assert r["pert"]["media"] == pytest.approx(44 / 6)


def test_validacao_tres_pontos():
    for a, m, b in [(0, 1, 2), (5, 4, 8), (3, 3, 3), (4, 9, 8)]:
        with pytest.raises(dd.ErroAjuste):
            dd.beta_pert(a, m, b)


def test_triangular_media():
    tr = dd.triangular(4, 6, 16)
    assert tr.media() == pytest.approx(26 / 3)
    assert tr.quantil(0.8) == pytest.approx(16 - math.sqrt(0.2 * 12 * 10))


def test_d1_moda_implicita():
    # o valor 2/3 da beta-PERT (4; 6; 16) é 8,0316; lido como quantil, a moda implícita volta a ser 6
    q = dd.beta_pert(4, 6, 16).quantil(2 / 3)
    r = dd.moda_implicita_quantil(4, q, 16)
    assert r["moda_implicita"] == pytest.approx(6.0, abs=1e-6)
    assert r["p80_lido_como_quantil"] == pytest.approx(9.074998, abs=1e-6)
    assert r["variacao_p80"] < 0          # ler quantil como moda superestima o P80
    with pytest.raises(dd.ErroAjuste):     # 6 h nem é atingível como quantil 2/3 com esses extremos
        dd.moda_implicita_quantil(4, 6, 16)


# ---------------------------------------------------------------- tamanho de amostra e work sampling -----------------
def test_tamanho_amostra_media():
    assert dd.tamanho_amostra_media(0.5, 0.05, 0.95) == 385      # (1,96·0,5/0,05)² = 384,15
    assert dd.tamanho_amostra_media(0.5, 0.05, 0.95, n_piloto=20) >= 385


def test_formula_oit_equivale_a_cv_populacional():
    x = np.array([6.1, 5.4, 7.9, 6.6, 5.8, 8.4, 6.2, 7.1, 5.5, 6.9])
    cv_pop = x.std(ddof=0) / x.mean()
    assert dd.ciclos_cronoanalise_oit(x, 0.05, 2.0) == math.ceil((40 * cv_pop) ** 2)


def test_tempo_padrao():
    r = dd.tempo_padrao([10, 12, 14], fator_ritmo=1.1, tolerancias=0.15)
    assert r["tempo_normal"] == pytest.approx(13.2)
    assert r["tempo_padrao"] == pytest.approx(15.18)


def test_amostragem_trabalho():
    assert dd.amostragem_trabalho_n(0.5, erro_absoluto=0.05, confianca=0.95) == 385
    # forma tabelada da OIT, n = 4p(100 − p)/L², com z = 2: p = 30%, L = 3 p.p. → 933,3
    n = dd.amostragem_trabalho_n(0.30, erro_absoluto=0.03, confianca=float(2 * stats.norm.cdf(2) - 1))
    assert n == math.ceil(4 * 30 * 70 / 9)
    props = dd.amostragem_trabalho_proporcoes(["a"] * 30 + ["b"] * 70)
    assert props[0]["categoria"] == "b" and props[0]["p"] == pytest.approx(0.7)
    assert props[0]["li"] < 0.7 < props[0]["ls"]
    assert dd.tempo_unitario_amostragem(0.5, 960, 96) == pytest.approx(5.0)


# ---------------------------------------------------------------- ajuste com dados ---------------------------------
def test_anderson_darling_bate_com_scipy():
    x = np.random.default_rng(1).normal(10, 2, 200)
    r = stats.anderson(x, "norm")
    a2 = dd.estatistica_anderson_darling(x, stats.norm(x.mean(), x.std(ddof=1)).cdf)
    assert a2 == pytest.approx(r.statistic, rel=1e-9)


def test_ajuste_recupera_lognormal_e_recomenda():
    x = np.random.default_rng(7).lognormal(1.6, 0.5, 400)
    ajustes = dd.ajustar_candidatas(x)
    ln = next(a for a in ajustes if a.nome == "lognormal")
    assert ln.parametros["mu"] == pytest.approx(np.log(x).mean())
    assert ln.parametros["sigma"] == pytest.approx(np.log(x).std())
    rec = dd.recomendar(ajustes)
    assert rec["escolhida"].nome in {"lognormal", "lognormal_deslocada"}
    assert ajustes[0].aic <= ajustes[-1].aic
    exp_ = next(a for a in ajustes if a.nome == "exponencial")
    assert exp_.aic - ajustes[0].aic > 10


def test_lognormal_deslocada_mle_recupera_deslocamento():
    g = np.random.default_rng(11)
    x = 2.0 + g.lognormal(1.0, 0.4, 2000)
    aj = dd.ajustar_candidata(x, "lognormal_deslocada")
    assert aj.parametros["lambda"] == pytest.approx(2.0, abs=0.35)
    assert aj.k == 3


def test_bootstrap_gera_p_valores():
    x = np.random.default_rng(3).gamma(3.0, 2.0, 60)
    ajustes = dd.ajustar_candidatas(x, ["gama", "exponencial"], bootstrap=49, semente=5)
    gama = next(a for a in ajustes if a.nome == "gama")
    assert 0 < gama.ad_p <= 1 and 0 < gama.ks_p <= 1
    assert next(a for a in ajustes if a.nome == "exponencial").ad_p < 0.05


def test_candidatas_limitadas_sob_pedido():
    x = np.random.default_rng(9).gamma(4, 1.5, 80)
    aj = dd.ajustar_candidata(x, "triangular")
    assert aj.parametros["minimo"] < x.min() and aj.parametros["maximo"] > x.max()
    rec = dd.recomendar(dd.ajustar_candidatas(x, ["gama", "triangular", "beta"]))
    assert rec["escolhida"].nome == "gama"


def test_rejeita_tempos_nao_positivos():
    with pytest.raises(dd.ErroAjuste):
        dd.ajustar_candidata([0, 1, 2, 3, 4, 5], "gama")


def test_censura_sem_censura_igual_mle_comum():
    x = np.random.default_rng(5).lognormal(2, 0.5, 80)
    aj = dd.ajustar_censurado(x, np.ones(80, bool), "lognormal")
    assert aj.parametros["mu"] == pytest.approx(np.log(x).mean(), abs=1e-5)
    assert aj.parametros["sigma"] == pytest.approx(np.log(x).std(), abs=1e-5)


def test_censura_corrige_vies():
    g = np.random.default_rng(8)
    real = g.lognormal(2.3, 0.6, 3000)
    corte = g.uniform(2, 30, 3000)
    obs = real <= corte
    t = np.where(obs, real, corte)
    aj = dd.ajustar_censurado(t, obs, "lognormal")
    verdadeiro = float(stats.lognorm(0.6, scale=math.exp(2.3)).ppf(0.8))
    ingenuo = dd.ajustar_candidata(t[obs], "lognormal").p80
    assert abs(aj.p80 - verdadeiro) / verdadeiro < 0.05
    assert ingenuo < 0.9 * verdadeiro


def test_intervalar_arredondado():
    g = np.random.default_rng(4)
    real = g.gamma(2.2, 0.35, 3000)
    x = np.maximum(np.round(real / 0.25) * 0.25, 0.25)
    lo, hi = dd.intervalos_de_arredondamento(x, 0.25, piso=0.25)
    aj = dd.ajustar_intervalar(lo, hi, "gama")
    assert aj.parametros["forma"] == pytest.approx(2.2, rel=0.08)
    assert aj.parametros["escala"] == pytest.approx(0.35, rel=0.08)


# ---------------------------------------------------------------- dados e CLI ---------------------------------------
def test_dados_ficticios_existem():
    for nome in ("tempos_parecer", "censurados", "arredondados", "amostragem_trabalho", "tres_pontos"):
        assert (DADOS / f"distribuicoes_{nome}.csv").exists()


def test_cli_tres_pontos(capsys):
    assert dd.cli(["--tres-pontos", "4", "6", "16"]) == 0
    saida = capsys.readouterr().out
    assert "9,074998" in saida and "10,920644" in saida and "20,3%" in saida


def test_cli_amostra_com_grafico(tmp_path, capsys):
    png = tmp_path / "ajuste.png"
    assert dd.cli([str(DADOS / "distribuicoes_tempos_parecer.csv"), "--grafico", str(png)]) == 0
    saida = capsys.readouterr().out
    assert "RECOMENDAÇÃO" in saida and "TAMANHO DE AMOSTRA" in saida
    assert png.exists() and png.stat().st_size > 10_000


def test_cli_outros_modos(tmp_path, capsys):
    assert dd.cli([str(DADOS / "distribuicoes_tres_pontos.csv"), "--grafico", str(tmp_path / "t.png")]) == 0
    assert dd.cli([str(DADOS / "distribuicoes_censurados.csv"), "--censura", "encerrado"]) == 0
    assert dd.cli([str(DADOS / "distribuicoes_arredondados.csv"), "--arredondamento", "0,25"]) == 0
    assert dd.cli([str(DADOS / "distribuicoes_amostragem_trabalho.csv"), "--amostragem-trabalho", "categoria"]) == 0
    saida = capsys.readouterr().out
    assert "não admitem log-normal deslocada" in saida
    assert "ainda abertos" in saida and "AMOSTRAGEM DO TRABALHO" in saida
    assert dd.cli([]) == 2
