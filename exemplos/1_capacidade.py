"""Exemplo 1 — Mapeamento de processos e análise de capacidade (dados fictícios).

Processo: licenciamento de obra simples numa Secretaria de Urbanismo de um município fictício.
Rode da raiz do repositório:  python exemplos/1_capacidade.py
Figuras em exemplos/saida/.
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dft_ferramentas import capacidade as cap  # noqa: E402
from dft_ferramentas._comum import br  # noqa: E402

SAIDA = RAIZ / "exemplos" / "saida"
SAIDA.mkdir(parents=True, exist_ok=True)

H_MES = 221 * 6 * 0.70 / 12  # 77,35 h efetivas por pessoa/mês (928,2 h/ano do exemplo de nove passos ÷ 12)

print("PASSO 1 — Do fluxo às visitas: quantas vezes cada atividade roda por pedido que entra")
fluxo = cap.ler_fluxo(RAIZ / "dados" / "capacidade_fluxo.csv")
print("  Avisos de validação do fluxo:", cap.validar_fluxo(fluxo) or "nenhum")
visitas = cap.razao_visitas(fluxo)
for a, v in visitas.items():
    print(f"  {a}: {br(v, 3)} execuções por pedido")

print("\nPASSO 2 — Capacidade por etapa, utilização, gargalo, takt e Little (100 pedidos/mês)")
ativ = cap.ler_atividades(RAIZ / "dados" / "capacidade_atividades.csv")
tab = cap.analisar_capacidade(ativ, fluxo=fluxo, demanda_processo=100)
res = cap.resumo_processo(tab, 100)
print(cap.relatorio(tab, res, "mês", dias_periodo=22))
cap.grafico_capacidade(tab, SAIDA / "1_capacidade_situacao_atual.png", "Licenciamento de obra — situação atual")

print("\nPASSO 3 — Cenário: os 3 arquitetos atuam em pool (A3 e A5 compartilham a equipe)")
cargas = tab.set_index("id")["carga_h"]
pool = ativ.copy()
for etapa in ("A3", "A5"):  # cada etapa recebe fração do pool proporcional à sua carga
    pool.loc[pool["id"] == etapa, "pessoas"] = 3 * cargas[etapa] / (cargas["A3"] + cargas["A5"])
tab_pool = cap.analisar_capacidade(pool, fluxo=fluxo, demanda_processo=100)
res_pool = cap.resumo_processo(tab_pool, 100)
print(f"  Gargalo passa a ser: {res_pool['gargalo']} (utilização {br(100 * res_pool['utilizacao_gargalo'], 1)}%)")
print(f"  Capacidade do processo: {br(res['capacidade_processo'], 1)} → {br(res_pool['capacidade_processo'], 1)} "
      "pedidos/mês, sem contratar ninguém — mas a equipe de arquitetos fica a ~91%, na zona de fila.")
cap.grafico_capacidade(tab_pool, SAIDA / "1_capacidade_pool.png", "Licenciamento de obra — arquitetos em pool")

print("\nPASSO 4 — Esforço unitário que o fluxo entrega à Fase 6")
tempos = dict(zip(ativ["id"], ativ["tempo_min"]))
e = cap.esforco_unitario(visitas, tempos)
feliz = (tempos["A1"] + tempos["A2"] + tempos["A3"] + 0.4 * tempos["A4"] + tempos["A5"] + tempos["A6"]) / 60
print(f"  E = Σ visitas × tempo = {br(e, 3)} h por pedido; o 'caminho feliz' (sem exigências) daria {br(feliz, 3)} h —")
print(f"  ignorar o retrabalho subestimaria o esforço em {br(100 * (e / feliz - 1), 1)}%.")

print("\nPASSO 5 — Tipos de capacidade da análise técnica (A3)")
c = cap.capacidades(pessoas=2, horas_brutas_pessoa=221 * 6 / 12, tempo_min=90, fator_disponibilidade=0.70)
dem = cap.capacidade_demonstrada([101, 104, 99, 106, 103, 105])  # produção mensal realizada (com fila o tempo todo)
ind = cap.indicadores_capacidade(dem, c["teorica"], c["efetiva"])
print(f"  Teórica {br(c['teorica'], 1)} | efetiva {br(c['efetiva'], 1)} | demonstrada {br(dem, 1)} pedidos/mês")
print(f"  Utilização sobre a teórica {br(100 * ind['utilizacao_sobre_teorica'], 1)}%; eficiência sobre a efetiva "
      f"{br(100 * ind['eficiencia_sobre_efetiva'], 1)}%")

print("\nPASSO 6 — Fila no balcão de atendimento (M/M/c e Kingman): 30 chegadas/h, 8 min por atendimento")
lam, mu = 30 / 60, 1 / 8  # por minuto
for c_ in range(5, 8):
    m = cap.mmc(lam, mu, c_)
    print(f"  {c_} atendentes: utilização {br(100 * m['rho'], 1)}% | espera média {br(m['Wq'], 1)} min | "
          f"atendidos em até 10 min: {br(100 * cap.nivel_servico(lam, mu, c_, 10), 1)}%")
print(f"  Mínimo para 80% em até 10 min: {cap.servidores_para_meta(lam, mu, 10, 0.8)} atendentes "
      f"(a carga pura pediria {br(lam / mu, 1)})")
cap.grafico_curva_fila(SAIDA / "1_capacidade_curva_fila.png")

print("\nPASSO 7 — Ponte com o exemplo de nove passos do curso")
print(f"  Q = 7.770 / 928,2 = {br(cap.pessoas_necessarias(7770, 928.2), 2)} pessoas; com o quadro atual de 7, "
      f"utilização = {br(100 * 7770 / (7 * 928.2), 1)}% da capacidade efetiva.")
print(f"\nFiguras salvas em {SAIDA}")
