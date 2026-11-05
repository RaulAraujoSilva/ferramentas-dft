# Convenções do repositório (para quem contribui)

- Um módulo por ferramenta em `dft_ferramentas/` (`capacidade.py`, `recursos.py`, `distribuicoes.py`, `nivelamento.py`,
  `previsao.py`, `simulacao.py`). Funções puras, com docstring em português (o que faz, entradas, saídas, unidade,
  referência do método) e tipos. Sem I/O escondido; gráficos opcionais via `_comum.estilo_graficos()`.
- Cada módulo tem: `exemplos/<n>_<modulo>.py` (roda do terminal e imprime um relatório legível; salva figuras em
  `exemplos/saida/`), dados fictícios em `dados/<modulo>_*.csv` (separador `;`, vírgula decimal) e `tests/test_<modulo>.py`.
- Subcomando de linha de comando em `dft_ferramentas/cli.py` (`dft <modulo> <arquivo.csv> [opções]`).
- Documentação do método em `docs/metodos/<n>_<modulo>.md`: padrão usual (normas, manuais, artigos), passo a passo,
  como entra no DFT, armadilhas e referências ABNT.
- Números canônicos do curso (dossiê `metodologia-dft-2026`): CT-1 P80 beta-PERT = 9,074998 h; CT-2 P80 log-normal
  deslocada = 10,920644 h; CT-3 Holt vence SES (RMSE 6,20 < 7,41), 24 → 25 servidores; exemplo de nove passos
  928,2 h/ano e Q = 8,37. Testes devem reproduzi-los quando o módulo os cobrir.
- Dados sempre fictícios. Nunca nomear município real nem citar decisão do TCE-RJ.
- Dependências: numpy, scipy, pandas, matplotlib, statsmodels (simpy opcional). Python ≥ 3.10.
