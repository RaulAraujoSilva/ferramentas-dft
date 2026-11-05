"""Testes do módulo 4 — previsão de demanda.

Reproduzem o caso de teste canônico CT-3 do curso (dossiê L2, §6; motor TypeScript ``sistema/src/lib/motor``):
Holt vence SES (RMSE 6,20 < 7,41); Σ projeção de 12 meses = 2.481,05; vetores +12% (h ≥ 4) e −5% (h ≥ 7);
Σ ajustada = 2.637,19.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dft_ferramentas import previsao as P
from dft_ferramentas._comum import ler_csv

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
Y = [120, 132, 128, 145, 150, 148, 161, 158, 172, 169, 181, 178]
TREINO, TESTE = Y[:9], Y[9:]
JUST = ("A Lei Municipal nº 4.512/2026 (fictícia) amplia o rol de contratos administrativos sujeitos a parecer "
        "jurídico prévio, alcançando as dispensas acima de trinta mil reais; a estimativa de doze por cento decorre "
        "da contagem dos contratos do exercício anterior que passariam a exigir parecer.")
V1 = P.Vetor(id="v1", nome="Nova lei amplia o rol de contratos", fator=1.12, inicio=4, justificativa=JUST,
             fonte="Lei Municipal nº 4.512/2026 (fictícia)", responsavel="Coordenadoria", entregas=("parecer",),
             aprovacao_segundo="Procuradoria (fictícia)")
V2 = P.Vetor(id="v2", nome="Assinatura eletrônica", fator=0.95, inicio=7, justificativa=JUST,
             fonte="Contrato nº 27/2026 (fictício)", responsavel="TI", entregas=("parecer",))


def serie36():
    return ler_csv(DADOS / "previsao_serie.csv")["requerimentos"].to_numpy(float)


# ---------------------------------------------------------------- CT-3, passos 1 a 3
def test_ct3_ses_no_treino():
    s = P.ajustar_ses(TREINO)
    assert s.alpha == 0.90
    assert s.sse == pytest.approx(837.4104, abs=1e-4)
    assert s.lT == pytest.approx(170.617133, abs=1e-6)
    assert np.allclose(s.prever(3), 170.6171, atol=1e-4)
    assert s.alertas == []


def test_ct3_holt_no_treino_com_alerta_de_fronteira():
    h = P.ajustar_holt(TREINO)
    assert (h.alpha, h.beta, h.phi) == (0.35, 1.0, 1.0)
    assert (h.l0, h.b0) == (120, 12)
    assert h.sse == pytest.approx(369.1182, abs=1e-4)
    assert h.lT == pytest.approx(168.104103, abs=1e-6)
    assert h.bT == pytest.approx(6.379039, abs=1e-6)
    assert np.allclose(h.prever(3), [174.4831, 180.8622, 187.2412], atol=1e-4)
    texto = " | ".join(h.alertas)
    assert "fronteira da grade" in texto and "β*" in texto and "amortecido" in texto


def test_ct3_holt_vence_ses_rmse_6_20_menor_que_7_41():
    r = P.avaliar_treino_teste(Y, 3, ["ses", "holt"])
    assert r["ses"]["rmse"] == pytest.approx(7.4145, abs=1e-4)
    assert r["ses"]["mae"] == pytest.approx(6.4610, abs=1e-3)
    assert r["ses"]["mape"] == pytest.approx(3.6137, abs=1e-3)
    assert r["holt"]["rmse"] == pytest.approx(6.2044, abs=1e-4)
    assert r["holt"]["mae"] == pytest.approx(4.9541, abs=1e-3)
    assert r["holt"]["mape"] == pytest.approx(2.8374, abs=1e-3)
    assert r["holt"]["mase"] < r["ses"]["mase"]
    assert P.escolher_metodo(r) == "holt"


# ---------------------------------------------------------------- CT-3, passo 4
def test_ct3_reajuste_e_projecao_12_meses_soma_2481_05():
    h = P.ajustar_holt(Y)
    assert (h.alpha, h.beta) == (0.34, 0.98)
    assert h.sse == pytest.approx(453.3634, abs=1e-4)
    assert h.lT == pytest.approx(182.165145, abs=1e-6)
    assert h.bT == pytest.approx(3.782945, abs=1e-6)
    assert h.alertas == []
    proj = h.prever(12)
    esperado = [185.9481, 189.7310, 193.5140, 197.2969, 201.0799, 204.8628, 208.6458, 212.4287, 216.2117,
                219.9946, 223.7775, 227.5605]
    assert np.allclose(proj, esperado, atol=1e-3)
    assert proj.sum() == pytest.approx(2481.0515, abs=1e-3)


def test_ct3_desvio_residuos_e_intervalo_normal():
    h = P.ajustar_holt(Y)
    res = P.residuos_um_passo(h, Y)
    assert res.size == 11
    sd = P.desvio_residuos(res, 2)
    assert sd == pytest.approx(7.0974, abs=1e-4)
    lo, hi = P.intervalo_normal(h.prever(1)[0], sd, 1.28)
    assert (round(lo, 2), round(hi, 2)) == (176.86, 195.03)


# ---------------------------------------------------------------- CT-3, passo 5 (vetores, D4)
def test_ct3_vetores_fatores_e_soma_ajustada_2637_19():
    base = P.ajustar_holt(Y).prever(12)
    r = P.aplicar_vetores(base, [V1, V2], entrega="parecer")
    assert np.allclose(r["fatores"][:3], 1.0)
    assert np.allclose(r["fatores"][3:6], 1.12)
    assert np.allclose(r["fatores"][6:], 1.064)
    assert r["total_base"] == pytest.approx(2481.0515, abs=1e-3)
    assert r["total_ajustada"] == pytest.approx(2637.1918, abs=1e-3)
    assert r["delta_absoluto"] == pytest.approx(156.1404, abs=1e-3)
    assert r["delta_percentual"] == pytest.approx(6.2933, abs=1e-4)
    esperado = [185.9481, 189.7310, 193.5140, 220.9726, 225.2095, 229.4464, 221.9991, 226.0241, 230.0492,
                234.0743, 238.0993, 242.1244]
    assert np.allclose(r["ajustada"], esperado, atol=1e-3)
    assert np.array_equal(r["base"], base)  # a base sempre acompanha a ajustada
    assert r["alertas"] == []
    assert r["trilha"][0]["vigencia"] == "h ≥ 4, permanente"


def test_vetores_ordem_e_entrega():
    assert P.fator_composto([V1, V2], 9, "parecer") == P.fator_composto([V2, V1], 9, "parecer")
    assert P.fator_composto([V1, V2], 9, "certidao") == 1.0
    v3 = P.Vetor(nome="Mutirão temporário", fator=1.3, inicio=2, fim=3, justificativa=JUST, fonte="Plano (fictício)",
                 aprovacao_segundo="Gabinete")
    assert [P.fator_composto([v3], h) for h in (1, 2, 3, 4)] == [1.0, 1.3, 1.3, 1.0]


@pytest.mark.parametrize("mudanca, msg", [({"fator": 1.02}, "pequeno demais"), ({"fator": 0.98}, "pequeno demais"),
                                          ({"fator": 0}, "maior que zero"), ({"justificativa": "curta"}, "200"),
                                          ({"fonte": "  "}, "fonte"), ({"inicio": 0}, "início"),
                                          ({"inicio": 6, "fim": 3}, "fim")])
def test_vetores_travas_d4(mudanca, msg):
    import dataclasses

    with pytest.raises(P.ErroVetor, match=msg):
        P.validar_vetor(dataclasses.replace(V1, **mudanca))


def test_vetores_bordas_e_alertas():
    import dataclasses

    P.validar_vetor(dataclasses.replace(V1, fator=0.97))
    P.validar_vetor(dataclasses.replace(V1, fator=1.03))
    base = np.full(12, 100.0)
    r = P.aplicar_vetores(base, [dataclasses.replace(V1, aprovacao_segundo="")])
    assert any("anti-otimismo" in a for a in r["alertas"])
    r = P.aplicar_vetores(base, [dataclasses.replace(V1, fator=1.7)])
    assert any("faixa admitida" in a for a in r["alertas"])
    assert not any("anti-otimismo" in a for a in P.aplicar_vetores(base, [V2])["alertas"])


def test_ler_vetores_csv_reproduz_ct3():
    vs = P.ler_vetores_csv(str(DADOS / "previsao_vetores_ct3.csv"))
    assert [v.fator for v in vs] == [1.12, 0.95]
    assert [v.inicio for v in vs] == [4, 7] and all(v.fim is None for v in vs)
    r = P.aplicar_vetores(P.ajustar_holt(Y).prever(12), vs, entrega="parecer")
    assert r["total_ajustada"] == pytest.approx(2637.1918, abs=1e-3)


# ---------------------------------------------------------------- métodos de referência e medidas de erro
def test_metodos_de_referencia():
    r = P.metodos_referencia(TREINO, 3)
    assert r["media"][0] == pytest.approx(np.mean(TREINO))
    assert list(r["ingenuo"]) == [172, 172, 172]
    assert np.allclose(r["drift"], [178.5, 185.0, 191.5])
    assert r["sazonal_ingenuo"] is None
    assert list(P.prever_sazonal_ingenuo(Y, 14, 12)) == Y + Y[:2]


def test_medidas_erro_guardas():
    m = P.medidas_erro([0, 10, 20], [1, 11, 21], treino=[5, 6, 7])
    assert m["mape"] is None and any("MAPE" in a for a in m["alertas"])
    assert m["mae"] == pytest.approx(1.0) and m["rmse"] == pytest.approx(1.0)
    assert m["mase"] == pytest.approx(1.0)
    with pytest.raises(P.ErroPrevisao):
        P.medidas_erro([1, 2], [1])


def test_holt_amortecido_phi_na_faixa_e_previsao_converge():
    h = P.ajustar_holt_amortecido(Y)
    assert 0.80 <= h.phi <= 0.98 and h.tipo == "holt_amortecido"
    f = h.prever(240)
    assert abs(f[-1] - (h.lT + h.phi * h.bT / (1 - h.phi))) < 1e-6  # assíntota (FPP3 §8.2)


# ---------------------------------------------------------------- validação de origem móvel
def test_origem_movel_sem_embaralhar_e_escolha():
    v = P.validacao_origem_movel(Y, ["media", "ingenuo", "drift", "ses", "holt", "holt_amortecido"])
    for fim_treino, idx in v["origens"]:
        assert all(i >= fim_treino for i in idx)  # treino sempre antes do teste
    assert [o[0] for o in v["origens"]] == list(range(v["inicio"], len(Y)))
    assert P.escolher_metodo(v["metricas"]) == "holt"
    assert v["metricas"]["holt"]["rmse"] < v["metricas"]["ses"]["rmse"]


def test_origem_movel_nao_usa_o_futuro():
    """Alterar um valor futuro não pode mudar a previsão feita numa origem anterior."""
    capturado = []

    def espiao(y, h):
        capturado.append(len(y))
        return np.full(h, y[-1])

    met = P.Metodo("espiao", "Espião", 1, 2, espiao)
    v = P.validacao_origem_movel(Y, [met], inicio=6, h_validacao=2)
    assert capturado == list(range(6, 12))
    assert v["metricas"]["espiao"]["n"] == 2 * 5 + 1


def test_empate_vence_o_mais_simples():
    met = {"holt": {"rmse": 5.0, "complexidade": 3}, "ingenuo": {"rmse": 5.0, "complexidade": 1}}
    assert P.escolher_metodo(met) == "ingenuo"


# ---------------------------------------------------------------- projeção com intervalo
def test_projecao_bootstrap_reprodutivel_e_alarga():
    p1 = P.projetar(Y, "holt", 12)
    p2 = P.projetar(Y, "holt", 12)
    assert np.array_equal(p1["inferior"], p2["inferior"])
    assert p1["previsao"].sum() == pytest.approx(2481.0515, abs=1e-3)
    assert np.all(p1["inferior"] <= p1["superior"])
    larg = p1["superior"] - p1["inferior"]
    assert larg[-1] > larg[0]
    for chave in ("media", "ingenuo", "drift", "ses", "holt_amortecido"):
        p = P.projetar(Y, chave, 6)
        assert p["inferior"].shape == (6,) and np.all(p["inferior"] <= p["superior"])


# ---------------------------------------------------------------- árvore de decisão, STL, sazonais e intermitente
def test_arvore_de_decisao_por_tamanho():
    assert P.recomendar_metodos(Y[:8])["modelar"] is False
    r12 = P.recomendar_metodos(Y)
    assert r12["modelar"] and "hw_aditivo" not in r12["metodos"] and "holt_amortecido" in r12["metodos"]
    r36 = P.recomendar_metodos(serie36())
    assert r36["forca_sazonal"] >= 0.64
    assert {"sazonal_ingenuo", "hw_aditivo", "arima"} <= set(r36["metodos"])
    ri = P.recomendar_metodos(ler_csv(DADOS / "previsao_intermitente.csv").iloc[:, 1].to_numpy(float))
    assert ri["ramo"] == "intermitente" and "tsb" in ri["metodos"]


def test_stl_forcas():
    t = np.arange(48)
    saz = 10 * np.sin(2 * np.pi * t / 12)
    d = P.decompor_stl(100 + saz + 0.1 * np.random.default_rng(1).normal(size=48), 12)
    assert d["forca_sazonal"] > 0.9
    d2 = P.decompor_stl(100 + np.random.default_rng(2).normal(size=48), 12)
    assert d2["forca_sazonal"] < 0.64


def test_holt_winters_e_arima_na_serie_sazonal():
    y = serie36()
    p = P.projetar(y, "hw_aditivo", 12, 12)
    assert p["previsao"].shape == (12,) and np.all(p["inferior"] < p["superior"])
    assert np.argmax(p["previsao"]) == 0  # janeiro é o pico sazonal da série fictícia
    a = P.ajustar_arima_auto(y, 12)
    assert a.ordem_sazonal[1] == 1 and np.isfinite(a.aicc)
    assert a.prever(12).shape == (12,)


def test_croston_sba_tsb():
    y = [0, 2, 0, 0, 3, 0, 1, 0, 0, 4, 0, 0]
    c = P.classificar_demanda(y)
    assert c["classe"] == "intermitente" and c["adi"] == pytest.approx(3.0)
    fc = P.prever_croston(y, 3)[0]
    assert P.prever_croston(y, 3, variante="sba")[0] == pytest.approx(fc * 0.95)
    assert P.prever_tsb(y + [0] * 24, 1)[0] < P.prever_tsb(y, 1)[0]  # TSB decai sem demanda
    assert P.prever_croston(y + [0] * 24, 1)[0] == pytest.approx(fc)  # Croston não atualiza


# ---------------------------------------------------------------- linha de comando
def test_cli_ct3_completo(tmp_path, capsys):
    png = tmp_path / "ct3.png"
    csv = tmp_path / "proj.csv"
    rc = P.cli([str(DADOS / "previsao_ct3.csv"), "--validacao", "treino-teste", "--n-teste", "3", "--metodos",
                "ses,holt", "--vetores", str(DADOS / "previsao_vetores_ct3.csv"), "--entrega", "parecer",
                "--grafico", str(png), "--saida", str(csv)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Σ base = 2.481,05" in out and "Σ ajustada = 2.637,19" in out
    assert "6,20" in out and "7,41" in out and "Método escolhido: Holt" in out
    assert png.exists() and png.stat().st_size > 10_000 and csv.exists()


def test_cli_serie_curta_nao_modela(tmp_path, capsys):
    f = tmp_path / "curta.csv"
    f.write_text("mes;demanda\n" + "\n".join(f"2026-{i:02d};{10 + i}" for i in range(1, 9)), encoding="utf-8")
    assert P.cli([str(f)]) == 0
    assert "não modelar" in capsys.readouterr().out
