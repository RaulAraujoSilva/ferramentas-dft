"""Exemplo 2.1 — Identificação da curva de distribuição de tempo das atividades (Fase 6 do DFT).

Roda do terminal, na raiz do repositório:

    python exemplos/2_1_distribuicoes.py

Percorre os dois caminhos do método, com dados fictícios de `dados/`:
  (a) com registro de tempo: descritiva → tamanho de amostra → ajuste de candidatas (MLE) → AIC/BIC, KS, AD,
      qui-quadrado → recomendação; mais os casos de dados censurados, arredondados e de amostragem do trabalho;
  (b) sem registro: três pontos (4; 6; 16) → beta-PERT (Vose) e log-normal deslocada → CT-1 e CT-2 do curso.
Figuras em `exemplos/saida/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dft_ferramentas import distribuicoes as dd  # noqa: E402
from dft_ferramentas._comum import br, ler_csv  # noqa: E402

DADOS = RAIZ / "dados"
SAIDA = RAIZ / "exemplos" / "saida"


def titulo(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def main() -> int:
    SAIDA.mkdir(parents=True, exist_ok=True)

    titulo("(a) COM DADOS — 90 tempos de 'Parecer técnico de licitação' (horas, fictícios)")
    x = ler_csv(DADOS / "distribuicoes_tempos_parecer.csv")["tempo_h"].to_numpy(dtype=float)
    d = dd.estatisticas_descritivas(x)
    print(f"n = {d['n']}; média = {br(d['media'], 3)} h; mediana = {br(d['mediana'], 3)} h; CV = {br(d['cv'], 3)}; "
          f"assimetria = {br(d['assimetria'], 3)}; P80 empírico = {br(d['p80_empirico'], 3)} h")
    for e in (0.10, 0.05):
        n = dd.tamanho_amostra_media(d["cv"], e, 0.95, n_piloto=d["n"])
        print(f"Para estimar a média com ±{br(100 * e, 0)}% (95%): {n} registros "
              f"({'já temos' if d['n'] >= n else 'ainda faltam ' + str(n - d['n'])}).")

    ajustes = dd.ajustar_candidatas(x, bootstrap=199)
    print(f"\n{'candidata':<28}{'AIC':>9}{'ΔAIC':>8}{'A²':>8}{'p(AD)':>8}{'P80':>9}")
    for aj in ajustes:
        if aj.aic == aj.aic:  # finito
            print(f"{aj.rotulo:<28}{br(aj.aic, 1):>9}{br(aj.aic - ajustes[0].aic, 2):>8}{br(aj.ad, 3):>8}"
                  f"{br(aj.ad_p, 3):>8}{br(aj.p80, 3):>9}")
    rec = dd.recomendar(ajustes)
    print("\nRecomendação:", rec["justificativa"])
    print("Figura:", dd.grafico_ajuste(x, ajustes, str(SAIDA / "2_1_ajuste_parecer.png")))

    titulo("(a′) DADOS CENSURADOS — licenças com processo ainda aberto na data de corte")
    dfc = ler_csv(DADOS / "distribuicoes_censurados.csv")
    t, obs = dfc["tempo_h"].to_numpy(dtype=float), dfc["encerrado"].to_numpy() == 1
    cens = dd.ajustar_censurado(t, obs, "lognormal")
    so_conc = dd.ajustar_candidata(t[obs], "lognormal")
    print(f"{int((~obs).sum())} de {t.size} registros abertos. P80 log-normal com censura = {br(cens.p80, 2)} h; "
          f"descartando os abertos = {br(so_conc.p80, 2)} h ({br(100 * (so_conc.p80 / cens.p80 - 1), 1)}%).")

    titulo("(a″) DADOS ARREDONDADOS — ouvidoria anotada em múltiplos de 15 min")
    xr = ler_csv(DADOS / "distribuicoes_arredondados.csv")["tempo_h"].to_numpy(dtype=float)
    lo, hi = dd.intervalos_de_arredondamento(xr, 0.25, piso=0.25)
    for fam in ("gama", "lognormal", "weibull"):
        aj = dd.ajustar_intervalar(lo, hi, fam)
        print(f"{aj.rotulo:<28} AIC = {br(aj.aic, 1):>7}  P80 = {br(aj.p80, 3)} h")

    titulo("(a‴) AMOSTRAGEM DO TRABALHO — 400 observações instantâneas, 20 dias, 8 servidores")
    cats = ler_csv(DADOS / "distribuicoes_amostragem_trabalho.csv")["categoria"]
    props = dd.amostragem_trabalho_proporcoes(cats)
    for p in props:
        print(f"{p['categoria']:<26}{br(100 * p['p'], 1):>6}%  IC95% [{br(100 * p['li'], 1)}%; {br(100 * p['ls'], 1)}%]")
    p_ana = next(p["p"] for p in props if p["categoria"] == "analise_de_processos")
    horas = 8 * 20 * 6.0  # 8 servidores × 20 dias × 6 h de jornada observada (fictício)
    tu = dd.tempo_unitario_amostragem(p_ana, horas, quantidade_produzida=95, fator_ritmo=1.0, tolerancias=0.0)
    print(f"Tempo unitário de análise = {br(p_ana, 3)} × {br(horas, 0)} h / 95 processos = {br(tu, 2)} h/processo")
    print(f"Observações para ±5 p.p. (p = {br(p_ana, 2)}, 95%): "
          f"{dd.amostragem_trabalho_n(p_ana, erro_absoluto=0.05)}")

    titulo("(b) SEM DADOS — três pontos do 'Parecer técnico de licitação': a = 4 h, m = 6 h (moda), b = 16 h")
    r = dd.resumo_tres_pontos(4, 6, 16, incluir_triangular=True)
    for lin in r["linhas"]:
        print(f"{lin['rotulo']:<28} média = {br(lin['media'], 4):>7}  P50 = {br(lin['P50'], 4):>7}  "
              f"P80 = {br(lin['P80'], 6):>10}  P95 = {br(lin['P95'], 4):>8}")
    bp, ln = r["beta_pert"], r["lognormal"]
    print(f"\nCT-1 beta-PERT (Vose): α = {br(bp.alfa, 6)}, β = {br(bp.beta, 6)}, P80 = {br(bp.quantil(0.8), 6)} h, "
          f"F(8) = {br(bp.cdf(8), 6)}")
    print(f"CT-2 log-normal deslocada: λ = {br(ln.lam, 6)}, μ = {br(ln.mu, 6)}, σ = {br(ln.sigma, 6)}, "
          f"P80 = {br(ln.quantil(0.8), 6)} h")
    print(f"Δ P80 = {br(100 * r['delta_p80'], 1)}% → com 2.637,19 pareceres/ano: "
          f"{br((ln.quantil(0.8) - bp.quantil(0.8)) * 2637.1918, 1)} h/ano")
    print(f"Erro de Monte Carlo do P80 com N = 10.000: ±{br(dd.erro_padrao_quantil(bp), 4)} h")
    d1 = dd.moda_implicita_quantil(4, 8, 16)
    print(f"D1 — se '8 h' fosse o valor que cobre 2/3 (quantil), a moda seria {br(d1['moda_implicita'], 3)} h e o "
          f"P80 cairia de {br(d1['p80_lido_como_moda'], 3)} para {br(d1['p80_lido_como_quantil'], 3)} h "
          f"({br(100 * d1['variacao_p80'], 1)}%).")
    print("Figura:", dd.grafico_tres_pontos(4, 6, 16, str(SAIDA / "2_1_tres_pontos_ct1_ct2.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
