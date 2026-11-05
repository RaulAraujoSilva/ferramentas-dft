"""Exemplo 3 — Nivelamento de recursos e Teoria das Restrições (dados fictícios de um serviço municipal).

Parte A: operação anual de renovação de alvarás sanitários de uma Vigilância Sanitária municipal fictícia
         (10 entregas, 2 perfis: Fiscal e Administrativo) — CPM, suavização (Burgess) e nivelamento (capacidade).
Parte B: fluxo de licenciamento de obras de uma Secretaria de Urbanismo fictícia (5 etapas, 28 processos/semana)
         — os 5 passos de focalização da TOC, tambor-pulmão-corda e dimensionamento pela restrição.

Rode da raiz do repositório:  python exemplos/3_nivelamento.py
Figuras em exemplos/saida/.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dft_ferramentas._comum import ler_csv  # noqa: E402
from dft_ferramentas.nivelamento import (  # noqa: E402
    atividades_de_dataframe, etapas_de_dataframe, relatorio_recursos, relatorio_toc,
)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    saida = RAIZ / "exemplos" / "saida"
    saida.mkdir(parents=True, exist_ok=True)

    ativs = atividades_de_dataframe(ler_csv(RAIZ / "dados" / "nivelamento_atividades.csv"))
    caps = {str(r["recurso"]): float(r["capacidade"])
            for _, r in ler_csv(RAIZ / "dados" / "nivelamento_capacidades.csv").iterrows()}
    print(relatorio_recursos(ativs, caps, grafico=str(saida / "3_nivelamento_histogramas.png")))
    print()

    etapas = etapas_de_dataframe(ler_csv(RAIZ / "dados" / "nivelamento_fluxo.csv"))
    print(relatorio_toc(etapas, horas_por_servidor=28.0, demanda=28.0, reducao=0.15,
                        destino="Análise documental", custo_servidor=3200.0, ganho_por_caso=900.0,
                        grafico=str(saida / "3_nivelamento_toc.png")))


if __name__ == "__main__":
    main()
