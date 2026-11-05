"""Testes do módulo recursos (Fase 5). Números canônicos: L1, Parte 3.6 (Passos 4, 6 e 8) e Fase 5, §5.6."""
from __future__ import annotations

import math
import os

import pandas as pd
import pytest

from dft_ferramentas import recursos as r
from dft_ferramentas._comum import ler_csv

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados")
TEC, AGE = "Técnico (arquiteto/engenheiro)", "Agente administrativo"


@pytest.fixture(scope="module")
def tabelas():
    return (ler_csv(os.path.join(DADOS, "recursos_entregas.csv")),
            ler_csv(os.path.join(DADOS, "recursos_atividades.csv")),
            ler_csv(os.path.join(DADOS, "recursos_perfis.csv")))


@pytest.fixture(scope="module")
def res(tabelas):
    return r.analisar(*tabelas, regra="teto")


def test_disponibilidade_canonica():
    assert r.disponibilidade_perfil(221, 6, 0.70) == pytest.approx(928.2)
    assert r.disponibilidade_perfil(221, 8, 0.70) == pytest.approx(1237.6)


def test_carga_total_e_esforco_por_entrega(res):
    pe = res["por_entrega"].set_index("entrega")
    assert pe["h_ano"].sum() == pytest.approx(7770.0)
    esperado = {"E1": 1.83, "E2": 0.43, "E3": 2.50, "E4": 1.20, "E5": 4.00}
    for e, v in esperado.items():
        assert pe.loc[e, "h_unit"] == pytest.approx(v)
    # Q agregado do exemplo de nove passos: 7.770 / 928,2 = 8,37
    assert pe["h_ano"].sum() / 928.2 == pytest.approx(8.3710407, rel=1e-6)


def test_canonico_jornadas_distintas(res):
    """L1, Fase 5, §5.6: técnico 3.418 h / 928,2 = 3,68 → 4; agente 4.352 h / 1.237,6 = 3,52 → 4; total 8."""
    c = res["carga"].set_index("perfil")
    assert c.loc[TEC, "carga_h"] == pytest.approx(3417.6)
    assert c.loc[AGE, "carga_h"] == pytest.approx(4352.4)
    assert c.loc[TEC, "q_fracionario"] == pytest.approx(3.6820, abs=1e-4)
    assert c.loc[AGE, "q_fracionario"] == pytest.approx(3.5168, abs=1e-4)
    assert int(c.loc[TEC, "q"]) == 4 and int(c.loc[AGE, "q"]) == 4
    assert int(c["q"].sum()) == 8


def test_canonico_passo_8(tabelas):
    """L1, Parte 3.6, Passo 8: E5 + 60% de E1 técnicos = 2.518 h → 2,71 → 3; agente 5.252 h → 5,66 → 6."""
    ent, atv, pf = tabelas
    pf = pf.copy()
    pf["jornada_h"] = 6
    atv = atv.copy()
    atv.loc[(atv["entrega"] == "E3") & (atv["perfil"] == TEC), "perfil"] = AGE
    c = r.carga_por_perfil(r.matriz_recursos(ent, atv, pf), pf).set_index("perfil")
    assert c.loc[TEC, "carga_h"] == pytest.approx(2517.6)
    assert c.loc[AGE, "carga_h"] == pytest.approx(5252.4)
    assert c.loc[TEC, "q_fracionario"] == pytest.approx(2.7123, abs=1e-4)
    assert c.loc[AGE, "q_fracionario"] == pytest.approx(5.6587, abs=1e-4)
    assert (int(c.loc[TEC, "q"]), int(c.loc[AGE, "q"])) == (3, 6)


def test_participacao_soma_um(res):
    p = res["participacao"]
    assert p.drop(index="TOTAL").sum(axis=1).round(9).eq(1.0).all()
    assert p.loc["E1", TEC] == pytest.approx(0.60)
    assert p.loc["E3", TEC] == pytest.approx(0.40)
    assert p.loc["TOTAL"].sum() == pytest.approx(1.0)


def test_volume_versus_carga_E2(res):
    pe = res["por_entrega"].set_index("entrega")
    assert pe.loc["E2", "pct_volume"] == pytest.approx(3600 / 6480)
    assert pe.loc["E2", "pct_carga"] == pytest.approx(1548 / 7770)


def test_tdabc_custo_por_entrega(res):
    taxa_tec = 168000 / 928.2
    taxa_age = 84000 / 1237.6
    c = res["custo"].set_index("entrega")
    assert c.loc["E5", "custo_unitario"] == pytest.approx(4.0 * taxa_tec)
    assert c.loc["E1", "custo_unitario"] == pytest.approx(1.098 * taxa_tec + 0.732 * taxa_age)
    total = 3417.6 * taxa_tec + 4352.4 * taxa_age
    assert c["custo_anual"].sum() == pytest.approx(total)
    assert c["pct_custo"].sum() == pytest.approx(1.0)


def test_capacidade_ociosa_e_razao_wisn(res):
    c = res["carga"].set_index("perfil")
    assert c.loc[TEC, "ociosa_h"] == pytest.approx(2 * 928.2 - 3417.6)  # sobrecarga
    assert c.loc[AGE, "ociosa_h"] == pytest.approx(5 * 1237.6 - 4352.4)
    assert c.loc[TEC, "razao_wisn"] == pytest.approx(2 / (3417.6 / 928.2))
    assert c.loc[TEC, "deficit"] == 2 and c.loc[AGE, "deficit"] == -1


def test_alertas_dados_validados(res):
    tipos = [a.tipo for a in res["alertas"]]
    assert tipos.count("perfil_sem_atividade") == 1  # Estagiário
    assert "atividade_sem_perfil" not in tipos
    assert "realocacao_de_nivel" in tipos  # E3 enquadramento não privativo no técnico
    assert not any(t.startswith("raci") for t in tipos)


def test_alertas_rascunho(tabelas):
    ent, _, pf = tabelas
    rasc = ler_csv(os.path.join(DADOS, "recursos_atividades_rascunho.csv"))
    tipos = [a.tipo for a in r.alertas(ent, rasc, pf)]
    assert "atividade_sem_perfil" in tipos
    assert tipos.count("perfil_nao_cadastrado") == 2
    assert "raci_varios_A" in tipos
    carga = r.carga_por_perfil(r.matriz_recursos(ent, rasc, pf), pf).set_index("perfil")
    assert carga.loc[r.SEM_PERFIL, "carga_h"] == pytest.approx(90.0)
    assert math.isnan(carga.loc["Fiscal de obras", "q"])


def test_entrega_sem_atividade_e_fora_do_catalogo():
    ent = pd.DataFrame({"entrega": ["A", "B"], "demanda": [10, 5]})
    atv = pd.DataFrame({"entrega": ["A", "Z"], "atividade": ["x", "y"], "perfil": ["P", "P"], "tempo_h": [1, 1]})
    tipos = [a.tipo for a in r.alertas(ent, atv)]
    assert "entrega_sem_atividade" in tipos and "entrega_fora_do_catalogo" in tipos


def test_frequencia_e_virgula_decimal():
    ent = pd.DataFrame({"entrega": ["A"], "demanda": ["100"]})
    atv = pd.DataFrame({"entrega": ["A", "A"], "atividade": ["base", "retrabalho"], "perfil": ["P", "P"],
                        "tempo_h": ["0,5", "2,0"], "frequencia": ["1", "0,25"]})
    m = r.matriz_recursos(ent, atv)
    assert m["h_ano"].sum() == pytest.approx(100 * (0.5 + 0.5))


def test_arredondamento_declarado():
    assert r.arredondar(8.37, "teto") == (9, 0.0)
    assert r.arredondar(8.37, "comum") == (8, 0.0)
    q, res_ = r.arredondar(8.37, "piso_com_residuo")
    assert q == 8 and res_ == pytest.approx(0.37)
    assert r.arredondar(4.0, "teto")[0] == 4
    with pytest.raises(r.ErroRecursos):
        r.arredondar(1.0, "banqueiro")


def test_wisn():
    caf, iaf = r.fatores_wisn([6, 4], [120], 1200)
    assert caf == pytest.approx(1 / 0.9)
    assert iaf == pytest.approx(0.1)
    assert r.requerido_wisn(2400, 1200, caf, iaf) == pytest.approx(2 / 0.9 + 0.1)
    with pytest.raises(r.ErroRecursos):
        r.fatores_wisn([100])


def test_validacoes():
    with pytest.raises(r.ErroRecursos):
        r.normalizar_entregas(pd.DataFrame({"entrega": ["A"], "demanda": [-1]}))
    with pytest.raises(r.ErroRecursos):
        r.normalizar_atividades(pd.DataFrame({"entrega": ["A"], "atividade": ["x"], "perfil": ["P"],
                                              "tempo_h": [1], "raci": ["Z"]}))
    with pytest.raises(r.ErroRecursos):
        r.normalizar_perfis(pd.DataFrame({"perfil": ["P"]}))
    with pytest.raises(r.ErroRecursos):
        r.taxa_custo_capacidade(1000, 0)


def test_custo_hora_informado_tem_precedencia():
    pf = r.normalizar_perfis(pd.DataFrame({"perfil": ["P"], "disponibilidade_h": [1000], "custo_hora": [50],
                                           "custo_anual": [99999]}))
    assert pf.loc[0, "custo_hora"] == 50


def test_cli_e_grafico(tmp_path, capsys):
    png = tmp_path / "g.png"
    csv = tmp_path / "carga.csv"
    cod = r.cli([os.path.join(DADOS, "recursos_atividades.csv"), "--grafico", str(png), "--saida-carga", str(csv)])
    saida = capsys.readouterr().out
    assert cod == 0 and png.exists() and csv.exists()
    assert "3,68" in saida and "3,52" in saida and "8,37" in saida
    lida = ler_csv(csv)
    assert lida["q"].sum() == 8
    assert r.cli([os.path.join(DADOS, "nao_existe_atividades.csv")]) == 2
