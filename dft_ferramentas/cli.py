"""Linha de comando: `dft <módulo> [opções]` — despacha para a função cli(argv) de cada módulo.

    dft capacidade dados/capacidade_atividades.csv
    dft distribuicoes --tres-pontos 4 6 16
    dft previsao dados/previsao_serie.csv --horizonte 12
    dft --lista
"""
from __future__ import annotations

import importlib
import sys

MODULOS = {
    "capacidade": "1. Mapeamento de processos e análise de capacidade",
    "recursos": "2. Identificação de recursos por atividade",
    "distribuicoes": "2.1 Curva de distribuição de tempo das atividades",
    "nivelamento": "3. Nivelamento de recursos e teoria das restrições",
    "previsao": "4. Previsão de demanda",
    "simulacao": "5. Simulação de processos por Monte Carlo",
}


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "--lista"):
        print("Ferramentas DFT — uso: dft <módulo> [opções]  (dft <módulo> --help)\n")
        for k, v in MODULOS.items():
            print(f"  {k:<14} {v}")
        return 0
    nome = argv[0]
    if nome not in MODULOS:
        print(f"módulo desconhecido: {nome}. Use dft --lista.")
        return 2
    mod = importlib.import_module(f"dft_ferramentas.{nome}")
    return int(mod.cli(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
