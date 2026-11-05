"""Exemplo 2 — Identificação de recursos por atividade (Fase 5 do DFT). Dados fictícios.

Roda do terminal, na raiz do repositório:

    python exemplos/2_recursos.py

O que mostra:
  1. a unidade fictícia de licenciamento (5 entregas, 2 perfis com jornadas de 6 h e 8 h): matriz entrega ×
     atividade × perfil, horas por entrega e por perfil, custo por entrega (TDABC) e Q_r por perfil;
  2. o mesmo cálculo com H único (o erro de agregar jornadas): 8,37 → 9 contra 3,68 + 3,52 → 4 + 4 = 8;
  3. o Passo 8 do exemplo de nove passos (L1, Parte 3.6): 2.517,6 h → 3 técnicos e 5.252,4 h → 6 agentes;
  4. a versão "rascunho" (como sairia de uma sugestão de IA, antes da validação humana) e os alertas;
  5. a armadilha do custo-hora sobre a jornada bruta;
  6. um cálculo WISN (OMS) com CAF e IAF, para comparação.
Figura em exemplos/saida/2_recursos.png e carga por perfil em exemplos/saida/2_recursos_carga.csv.
"""
from __future__ import annotations

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from dft_ferramentas import recursos as r  # noqa: E402
from dft_ferramentas._comum import br, ler_csv  # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DADOS = os.path.join(RAIZ, "dados")
SAIDA = os.path.join(RAIZ, "exemplos", "saida")
os.makedirs(SAIDA, exist_ok=True)

ent = ler_csv(os.path.join(DADOS, "recursos_entregas.csv"))
atv = ler_csv(os.path.join(DADOS, "recursos_atividades.csv"))
pf = ler_csv(os.path.join(DADOS, "recursos_perfis.csv"))

# 1. Fluxo completo --------------------------------------------------------------------------------------------
res = r.analisar(ent, atv, pf, regra="teto")
print(r.relatorio(res, "teto"))
r.grafico(res, os.path.join(SAIDA, "2_recursos.png"))
res["carga"].to_csv(os.path.join(SAIDA, "2_recursos_carga.csv"), sep=";", decimal=",", index=False,
                    encoding="utf-8-sig")

# 2 e 3. H único e o Passo 8 do exemplo de nove passos --------------------------------------------------------
print("\n" + "-" * 100)
print("PASSO 8 DO EXEMPLO DE NOVE PASSOS (L1, Parte 3.6) — todos com H = 928,2 h; E3 inteiro no agente")
pf_unico = pf.copy()
pf_unico["jornada_h"] = 6
atv_p8 = atv.copy()
tecnico, agente = "Técnico (arquiteto/engenheiro)", "Agente administrativo"
e3_tec = (atv_p8["entrega"] == "E3") & (atv_p8["perfil"] == tecnico)
atv_p8.loc[e3_tec, "perfil"] = agente  # no Passo 8 só E5 e 60% de E1 são técnicos
m8 = r.matriz_recursos(ent, atv_p8, pf_unico)
c8 = r.carga_por_perfil(m8, pf_unico).set_index("perfil")
for p in (tecnico, agente):
    print(f"  {p:<32} carga {br(c8.loc[p, 'carga_h'], 1):>9} h  Q_r = {br(c8.loc[p, 'q_fracionario'])} → "
          f"{int(c8.loc[p, 'q'])}")
print(f"  Total: {int(c8.loc[[tecnico, agente], 'q'].sum())} servidores (a soma bate com o agregado só por coincidência)")

# 4. Rascunho com lacunas -------------------------------------------------------------------------------------
print("\n" + "-" * 100)
print("RASCUNHO (vinculação sugerida antes da validação humana) — apenas os alertas")
rasc = ler_csv(os.path.join(DADOS, "recursos_atividades_rascunho.csv"))
for a in r.alertas(ent, rasc, pf):
    print(f"  [{a.tipo}] {a.mensagem}")

# 5. Armadilha do custo-hora ----------------------------------------------------------------------------------
print("\n" + "-" * 100)
print("ARMADILHA — custo-hora calculado sobre a jornada bruta (2.088 h = 261 dias × 8 h)")
custo_ag = 84000.0
bruta = r.taxa_custo_capacidade(custo_ag, 261 * 8)
pratica = r.taxa_custo_capacidade(custo_ag, r.disponibilidade_perfil(221, 8, 0.70))
print(f"  Agente: R$ {br(bruta)}/h (bruta) × R$ {br(pratica)}/h (capacidade prática) → o custo por entrega fica "
      f"{br(100 * (1 - bruta / pratica), 0)}% menor do que o real.")

# 6. WISN ------------------------------------------------------------------------------------------------------
print("\n" + "-" * 100)
print("WISN (OMS) para o perfil técnico — CAS 10% (reuniões e registros) e IAS 120 h/ano (chefia)")
H_tec = r.disponibilidade_perfil(221, 6, 1.0)  # no WISN o tempo disponível não aplica ITP
caf, iaf = r.fatores_wisn([6, 4], [120], H_tec)
req = r.requerido_wisn(3417.6, H_tec, caf, iaf)
print(f"  tempo disponível {br(H_tec, 0)} h; CAF = {br(caf, 3)}; IAF = {br(iaf, 3)}; requerido = {br(req)}; "
      f"razão WISN (lotação 2) = {br(2 / req)}")
print("  Atenção (D2): no curso, apoio estruturado vira entrega própria — não somar CAF/IAF a ela nem ao ITP.")
print(f"\nFigura e carga por perfil em {SAIDA}")
