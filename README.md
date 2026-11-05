# Ferramentas DFT — dimensionamento da força de trabalho com ferramentas abertas

Scripts em Python para cada etapa do **Dimensionamento da Força de Trabalho (DFT)** no setor público, usados no curso
*Dimensionamento da Força de Trabalho no Setor Público com Apoio de IA* da Escola de Contas e Gestão do TCE-RJ
(ECG/TCE-RJ). Cada ferramenta segue o **padrão mais usual da literatura e dos manuais** da área — documentado em
`docs/metodos/` com as referências — e vem com dados fictícios, exemplo pronto e testes que reproduzem os números do curso.

> Exemplos e dados são **fictícios**, construídos para fim didático. Ferramenta de apoio à decisão: o resultado é uma
> estimativa com premissas declaradas, sempre validada por quem conhece o serviço.

## As ferramentas

| Etapa | Módulo | O que faz | Padrões seguidos (detalhe em `docs/metodos/`) | Fase do método · encontro |
|---|---|---|---|---|
| 1. Mapeamento de processos e análise de capacidade | `capacidade` | capacidade, utilização e gargalo por etapa; lead time pela lei de Little; carga a partir do fluxo com desvios e retrabalho (razões de visita); fila (Kingman, Erlang C) | BPMN 2.0 (ISO/IEC 19510), ABPMP BPM CBOK, Guia de Gestão de Processos do MGI, Little, Dumas et al., Hopp & Spearman | Fases 3 e 6 · Encontros 2 e 3 |
| 2. Recursos por atividade | `recursos` | matriz entrega × atividade × perfil; horas-homem e custo por perfil e por entrega (TDABC); carga por perfil para a consolidação; alertas de lacuna | análise do trabalho (O*NET), RACI, TDABC (Kaplan & Anderson), WISN (OMS), modelo referencial federal | Fases 5 e 8 · Encontros 2 e 4 |
| 2.1 Curva de distribuição do tempo das atividades | `distribuicoes` | ajusta candidatas a tempos medidos (MLE, Anderson-Darling com bootstrap, AIC), dados censurados e arredondados; três pontos → beta-PERT e log-normal deslocada; tamanho de amostra | estudo de tempos (OIT), amostragem do trabalho, Law, NIST/SEMATECH, Malcolm et al., Vose | Fase 6 · Encontro 3 |
| 3. Nivelamento de recursos e teoria das restrições | `nivelamento` | suavização (prazo fixo, Burgess) e nivelamento (capacidade fixa, menor folga) com gráfico antes/depois; 5 passos da TOC, vazão do sistema, efeito de +1 servidor, pulmão | PMBOK, Burgess & Killebrew, Goldratt (A Meta, tambor-pulmão-corda), corrente crítica | Fases 7 e 8 · Encontro 4 |
| 4. Previsão de demanda | `previsao` | métodos de referência, SES, Holt, Holt amortecido, Holt-Winters, ARIMA automático, Croston/SBA/TSB; validação por origem móvel; intervalo por bootstrap; vetores de demanda | Hyndman & Athanasopoulos (fpp3), competições M, Gardner, Hyndman & Khandakar, Syntetos et al. | Fase 4 · Encontro 3 |
| 5. Simulação de processos por Monte Carlo | `simulacao` | carga anual do setor (N = 10.000, semente), P50/P80/P90 em horas e servidores, erro do percentil, correlação e tornado; fila de atendimento com 1, 2, 3… servidores | Metropolis & Ulam, Rubinstein & Kroese, Iman & Conover, Law, Banks et al., Sargent, PMI/AACE | Fases 6 e 8 · Encontros 3 e 4 |

## Instalação

```bash
pip install -e .            # ou: pip install -e ".[dev,simulacao]" para testes e SimPy
```

No Google Colab: `!pip install git+https://github.com/RaulAraujoSilva/ferramentas-dft` e depois `import dft_ferramentas`.

## Uso rápido

```bash
dft --lista
dft capacidade dados/capacidade_atividades.csv --grafico capacidade.png
dft recursos dados/recursos_atividades.csv --entregas dados/recursos_entregas.csv --perfis dados/recursos_perfis.csv
dft distribuicoes dados/distribuicoes_tempos_parecer.csv
dft distribuicoes dados/distribuicoes_tres_pontos.csv
dft nivelamento dados/nivelamento_atividades.csv --grafico nivelamento.png
dft previsao dados/previsao_ct3.csv --validacao treino-teste --n-teste 3 --metodos ses,holt
dft simulacao dados/simulacao_entregas.csv --n 10000 --disponibilidade 1153.60
```

Sem instalar: `python -m dft_ferramentas <módulo> ...`. Cada módulo tem um exemplo comentado em `exemplos/` que roda
do terminal e grava figuras em `exemplos/saida/`. Os CSV usam `;` e vírgula decimal (padrão brasileiro de planilha).

## Números do curso conferidos nos testes

| Caso | Resultado | Módulo |
|---|---|---|
| CT-1 (a = 4, m = 6, b = 16 h) | P80 beta-PERT = 9,074998 h | `distribuicoes`, `simulacao` |
| CT-2 (mesmos três pontos) | P80 log-normal deslocada = 10,920644 h | `distribuicoes`, `simulacao` |
| CT-3 (série de 12 meses) | Holt vence SES (RMSE 6,20 < 7,41); Σ = 2.481,05 → 2.637,19 com vetores; 24 → 25 servidores | `previsao`, `simulacao` |
| Exemplo de nove passos | 928,2 h/ano por servidor; Q = 7.770 / 928,2 = 8,37 | `capacidade`, `recursos`, `nivelamento`, `simulacao` |

```bash
pip install -e ".[dev]" && pytest          # 135 testes
```

## Estrutura

```
dft_ferramentas/   um módulo por ferramenta + cli.py (comando dft)
dados/             CSV fictícios
exemplos/          um script por ferramenta
tests/             pytest (números canônicos do curso)
docs/metodos/      dossiê de método de cada ferramenta + referências (JSON)
```

## Licença e créditos

MIT. Autor: Raul Araújo da Silva (CRIE/COPPE/UFRJ), docente do curso na ECG/TCE-RJ. Metodologia de base: Rodrigues,
Cavalcanti, Araújo da Silva e Macedo, *Proposta de metodologia de dimensionamento da força de trabalho para o TCE-RJ*,
Revista do TCE-RJ, 2021.

Esta ação educacional tem finalidade exclusivamente pedagógica; o uso das ferramentas não constitui validação, pelo
TCE-RJ ou pela ECG, de modelo de dimensionamento de qualquer órgão.
