"""Exemplo 5 — Simulação de processos por Monte Carlo (dados fictícios).

Setor: Secretaria de Administração de um município fictício (licitações, certidões, recursos, vistorias, balcão).
Rode da raiz do repositório:  python exemplos/5_simulacao.py
Figuras em exemplos/saida/.
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dft_ferramentas import simulacao as sim  # noqa: E402
from dft_ferramentas._comum import br  # noqa: E402

SAIDA = RAIZ / "exemplos" / "saida"
SAIDA.mkdir(parents=True, exist_ok=True)
for s in (sys.stdout,):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("PASSO 1 — Uma entrega só (CT-1): o P80 simulado converge para o P80 analítico")
ct1 = sim.TresPontos(4, 6, 16, "beta_pert")
p80_exato = sim.quantil_analitico(ct1, 0.8)
se = sim.erro_padrao_quantil(0.8, 10_000, float(ct1.densidade(p80_exato)))
print(f"  P80 analítico (beta-PERT de Vose, 4; 6; 16) = {br(p80_exato, 6)} h; erro-padrão teórico com N = 10.000: "
      f"{br(se, 4)} h")
um = sim.simular_esforco([sim.Entrega("Parecer", sim.TresPontos(1, 1, 1), ct1)], n=10_000, semente=42)
lo, hi = sim.ic_quantil_ordem(um.total, 0.8)
print(f"  P80 simulado (semente 42) = {br(um.p80_total, 4)} h; IC 95% por estatísticas de ordem = "
      f"[{br(lo, 4)}; {br(hi, 4)}]")
for n, p in sim.convergencia_p80(um.total, 2000):
    print(f"    após {br(n, 0):>6} cenários: P80 = {br(p, 4)} h")
print(f"  N para conhecer o P80 com ±0,02 h (95%): {br(sim.n_para_precisao_quantil(0.8, float(ct1.densidade(p80_exato)), 0.02), 0)}")

print("\nPASSO 2 — CT-3: quadro pela regra oficial (Σ demanda × P80) e conferência por simulação")
ct3 = sim.ler_entregas(RAIZ / "dados" / "simulacao_ct3.csv")
carga = sim.soma_p80_analitica(ct3)
q, qi = sim.servidores(carga, sim.DISPONIBILIDADE_CT3)
print(f"  Σ demanda × P80 = {br(carga, 2)} h ÷ {br(sim.DISPONIBILIDADE_CT3, 2)} h = {br(q, 6)} → {qi} servidores")

print("\nPASSO 3 — O setor inteiro: seis entregas, volumes e tempos incertos, sem correlação")
entregas = sim.ler_entregas(RAIZ / "dados" / "simulacao_entregas.csv")
res = sim.simular_esforco(entregas, n=10_000)
print(sim.relatorio_esforco(res, entregas, [sim.DISPONIBILIDADE_CT3, sim.DISPONIBILIDADE_NOVE_PASSOS]))
sim.grafico_esforco(res, SAIDA / "5_simulacao_esforco.png", sim.DISPONIBILIDADE_CT3)

print("\nPASSO 4 — E se os tempos andarem juntos? (causa comum: sistema lento, equipe nova, lei nova)")
for rho in (0.0, 0.3, 0.6, 1.0):
    r = sim.simular_esforco(entregas, n=10_000, correlacao=rho)
    q, qi = sim.servidores(r.p80_total, sim.DISPONIBILIDADE_CT3)
    print(f"  ρ de postos = {br(rho, 1)}: P80 da carga = {br(r.p80_total, 0)} h → {br(q, 2)} servidores "
          f"(soma dos P80 = {br(r.soma_p80_individuais, 0)} h)")
ic = sim.simular_esforco(entregas, n=10_000, correlacao=0.6, metodo_correlacao="iman_conover")
print(f"  Conferência com Iman-Conover (ρ = 0,6): P80 = {br(ic.p80_total, 0)} h")

print("\nPASSO 5 — Fila de atendimento: quantos analistas mudam o nível de serviço (prazo de 8 h úteis)?")
etapas = sim.ler_etapas(RAIZ / "dados" / "simulacao_processo.csv")
linhas = sim.varrer_servidores(etapas, 1.3, None, [1, 2, 3, 4, 5], horizonte=2000, aquecimento=200,
                               replicacoes=10, prazo=8)
print(sim.relatorio_fila(linhas, etapas, 1.3, 8))
sim.grafico_servidores(linhas, SAIDA / "5_simulacao_fila.png", 8)

print("\nPASSO 6 — Validação do simulador contra a fórmula de Erlang C (M/M/2, ρ = 0,75)")
mm2 = [sim.Etapa("Atendimento", 2, sim.TresPontos(1, 1, 1, "exponencial"))]
an = sim.erlang_c(1.5, 1.0, 2)
r = sim.simular_processo(mm2, 1.5, horizonte=4000, aquecimento=200, replicacoes=10).resumo()
print(f"  espera média analítica = {br(an['wq'], 3)} h; simulada = {br(r['espera_media'][0][0], 3)} ± "
      f"{br(r['espera_media'][0][1], 3)} h")
print("\nFiguras: exemplos/saida/5_simulacao_esforco.png e exemplos/saida/5_simulacao_fila.png")
