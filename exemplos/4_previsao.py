"""Exemplo 4 — Previsão de demanda (Fase 4 do DFT), com dados fictícios.

Roda do terminal, na raiz do repositório:

    python exemplos/4_previsao.py

Três partes:
  A. Caso canônico CT-3 (pareceres técnicos de licitação, 12 meses): SES × Holt no treino/teste, conferência por
     origem móvel, projeção de 12 meses com intervalo e aplicação dos dois vetores de demanda (D4).
  B. Série sazonal de 36 meses (requerimentos de protocolo): árvore de decisão, STL, origem móvel com todos os
     candidatos (inclui Holt-Winters e ARIMA automático) e projeção.
  C. Série intermitente de 24 meses (recursos de segunda instância): Croston, SBA e TSB.
Figuras em exemplos/saida/.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dft_ferramentas import previsao as P  # noqa: E402
from dft_ferramentas._comum import br, ler_csv  # noqa: E402

SAIDA = RAIZ / "exemplos" / "saida"
DADOS = RAIZ / "dados"


def titulo(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def parte_a() -> None:
    titulo("A. CT-3 — pareceres técnicos de licitação (12 meses)")
    y = ler_csv(DADOS / "previsao_ct3.csv")["pareceres"].to_numpy(float)
    print("Série:", ", ".join(br(v, 0) for v in y))
    print("\nPasso 1–3: treino = 9 primeiros meses; teste = 3 últimos (FPP3 §5.8).")
    tt = P.avaliar_treino_teste(y, 3, ["ses", "holt"])
    print(P.tabela_metricas(tt, P.escolher_metodo(tt)))
    ses, holt = P.ajustar_ses(y[:9]), P.ajustar_holt(y[:9])
    print(f"  SES: α* = {br(ses.alpha)}; SSE = {br(ses.sse, 4)}.  Holt: α* = {br(holt.alpha)}, β* = {br(holt.beta)};"
          f" SSE = {br(holt.sse, 4)}")
    for a in holt.alertas:
        print(f"  ! {a}")

    print("\nConferência por origem móvel (FPP3 §5.10), um passo à frente, sem embaralhar:")
    om = P.validacao_origem_movel(y, P.recomendar_metodos(y)["metodos"])
    print(P.tabela_metricas(om["metricas"], P.escolher_metodo(om["metricas"])))

    print("\nPasso 4: reajuste do Holt nos 12 meses e projeção de 12 meses.")
    proj = P.projetar(y, "holt", 12)
    mod = proj["detalhes"]["modelo"]
    print(f"  α* = {br(mod.alpha)}; β* = {br(mod.beta)}; ℓ_T = {br(mod.lT, 4)}; b_T = {br(mod.bT, 4)}; "
          f"Σ base = {br(proj['previsao'].sum(), 4)} pareceres/ano")
    lo, hi = P.intervalo_normal(proj["previsao"][0], proj["detalhes"]["desvio_residuos"])
    print(f"  Intervalo normal de 80% em h = 1: [{br(lo)}; {br(hi)}]  ·  bootstrap: "
          f"[{br(proj['inferior'][0])}; {br(proj['superior'][0])}]")

    print("\nPasso 5: vetores de demanda (D4).")
    vetores = P.ler_vetores_csv(str(DADOS / "previsao_vetores_ct3.csv"))
    aj = P.aplicar_vetores(proj["previsao"], vetores, "parecer", proj["inferior"], proj["superior"])
    print("  Fatores: " + "  ".join(br(f, 3) for f in aj["fatores"]))
    print(f"  Σ ajustada = {br(aj['total_ajustada'], 4)}  ·  Δ = {br(aj['delta_absoluto'], 4)} "
          f"({br(aj['delta_percentual'], 4)} %)")
    for a in aj["alertas"]:
        print(f"  ! {a}")
    P.grafico(y, proj, str(SAIDA / "4_previsao_ct3.png"), vetores, aj, "CT-3 — pareceres técnicos de licitação",
              ler_csv(DADOS / "previsao_ct3.csv")["mes"].tolist(), "pareceres/mês")
    print(f"  Figura: {SAIDA / '4_previsao_ct3.png'}")


def parte_b() -> None:
    titulo("B. Série sazonal de 36 meses — requerimentos de protocolo (fictícia)")
    df = ler_csv(DADOS / "previsao_serie.csv")
    y = df["requerimentos"].to_numpy(float)
    rec = P.recomendar_metodos(y, 12)
    print(f"Ramo da árvore: {rec['ramo']}; F_T = {br(rec['forca_tendencia'])}; F_S = {br(rec['forca_sazonal'])}")
    for j in rec["justificativa"]:
        print(f"  • {j}")
    print("Validação de origem móvel (pode levar ~20 s por causa do ARIMA automático)...")
    om = P.validacao_origem_movel(y, rec["metodos"], 12)
    escolhido = P.escolher_metodo(om["metricas"])
    print(P.tabela_metricas(om["metricas"], escolhido))
    proj = P.projetar(y, escolhido, 12, 12)
    if proj["detalhes"].get("descricao"):
        print(f"Modelo: {proj['detalhes']['descricao']}")
    print(f"Σ projeção de 12 meses = {br(proj['previsao'].sum(), 1)} requerimentos "
          f"(intervalo: {proj['detalhes']['tipo_intervalo']})")
    P.grafico(y, proj, str(SAIDA / "4_previsao_sazonal.png"), titulo="Requerimentos de protocolo (fictícios)",
              rotulos=df["mes"].tolist(), unidade="requerimentos/mês")
    print(f"Figura: {SAIDA / '4_previsao_sazonal.png'}")


def parte_c() -> None:
    titulo("C. Demanda intermitente — recursos de segunda instância (fictícia)")
    y = ler_csv(DADOS / "previsao_intermitente.csv").iloc[:, 1].to_numpy(float)
    c = P.classificar_demanda(y)
    print(f"ADI = {br(c['adi'])}; CV² = {br(c['cv2'])}; zeros = {br(100 * c['proporcao_zeros'], 0)} %; "
          f"classe: {c['classe']}")
    for k in ("croston", "sba", "tsb"):
        print(f"  {k.upper():<8} previsão mensal = {br(P.projetar(y, k, 1)['previsao'][0], 3)}")
    print("  Leitura: com demanda rara, dimensione pela taxa média (≈ 1 recurso/mês) e trate o pico como risco, "
          "não como tendência.")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    SAIDA.mkdir(parents=True, exist_ok=True)
    parte_a()
    parte_b()
    parte_c()
