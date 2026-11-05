# 1. Mapeamento de processos e análise de capacidade

**Módulo:** `dft_ferramentas/capacidade.py` · **Exemplo:** `exemplos/1_capacidade.py` · **Dados fictícios:**
`dados/capacidade_atividades.csv` e `dados/capacidade_fluxo.csv` · **Testes:** `tests/test_capacidade.py`

> Dados e exemplos são fictícios. Nenhum município real é nomeado e nenhuma decisão do TCE-RJ é citada.

## 1. Para que serve esta ferramenta no DFT

O DFT pergunta quantas pessoas são necessárias para entregar o que a unidade entrega. A conta central é carga
dividida por disponibilidade (Q = Σ Dᵢ·Eᵢ / H). O mapeamento de processos e a análise de capacidade respondem a três
perguntas que essa conta, sozinha, não responde:

1. **Quanto custa de fato uma entrega quando há desvios e retrabalho?** Um pedido que volta para "exigência" passa
   duas vezes pela triagem. O esforço unitário Eᵢ da Fase 6 precisa incluir essas voltas.
2. **Onde o processo trava?** Uma unidade pode ter quadro total suficiente e mesmo assim acumular fila numa etapa,
   porque as pessoas estão no lugar errado. Isso é gargalo, não déficit.
3. **A série histórica de produção mede demanda ou capacidade?** Onde há fila, a produção registrada é o teto da
   capacidade, não a necessidade do território (dossiê L1, Fase 4). A análise de capacidade mostra quando isso ocorre.

O artigo-base faz uma advertência que orienta todo o módulo. Para Rodrigues et al. (2021), o mapeamento de processo
"embora possa colaborar em muito com as estimativas, não deve ser considerado como pré-requisito". Por isso a
ferramenta trabalha com o **mínimo de notação** e com uma **tabela de atividades**, sem exigir um modelo BPMN
completo. Mapear vale a pena onde há fila, retrabalho alto ou entrega com várias fases executadas por cargos
diferentes. Nos demais casos, o catálogo de entregas basta.

## 2. Padrões usuais e fontes

| Padrão | O que estabelece | Uso aqui |
|---|---|---|
| **BPMN 2.0.2** (OMG, 2014), adotado como **ISO/IEC 19510:2013** (idêntica à BPMN 2.0.1, confirmada em 2022) | Notação e semântica de diagramas de processo, colaboração e coreografia. Define subclasses de conformidade de modelagem: **Descritiva**, **Analítica** e **Executável Comum** | A subclasse **Descritiva** é a notação mínima útil (ver §3.1) |
| **7PMG** (MENDLING; REIJERS; VAN DER AALST, 2010) | Sete diretrizes de qualidade de modelo: poucos elementos, um início e um fim, evitar OR, rótulos verbo + objeto, decompor acima de 50 elementos | Apresentada no Encontro 2 (Fase 3); aqui vira checagem automática (`validar_fluxo`) |
| **ABPMP BPM CBOK v3.0** (ABPMP, 2013) | Corpo de conhecimento do ciclo BPM. Apresenta o SIPOC (p. 94-95), a análise de capacidade (§4.4.6, p. 123) e define gargalo como "uma restrição de capacidade que cria uma fila" (§4.4.7, p. 124) | Definição de gargalo e perguntas da análise |
| **Guias de governo**: *Guia Prático de Gestão de Processos* (BRASIL, 2025), *Metodologia de Gestão de Processos de Negócio da CGU* (BRASIL, 2020) e o *Guia de Gestão de Processos de Governo* (GesPública/e-PING, 2011) | Ciclo BPM em cinco etapas (planejar, analisar, desenhar, implantar, monitorar), AS IS/TO BE, três níveis de modelagem e cadeia de valor | O guia do MGI cita a cadeia de valor como referencial para o **DFT**; o nível descritivo basta "se o objetivo é visualizar as etapas principais e identificar gargalos" |
| **SIPOC** (ABPMP, 2013; GEORGE, 2003) | Tabela fornecedor, entrada, processo, saída e cliente, usada para pactuar o escopo | Fronteira da entrega na Fase 3 |
| **Fluxograma e mapeamento do fluxo de valor (VSM)** (ROTHER; SHOOK, 1999; GEORGE, 2003) | VSM registra tempo de ciclo, estoque entre etapas, lead time e takt. George adapta a lógica a serviços e transações e põe a lei de Little no centro do diagnóstico | Takt, WIP e lead time por etapa |
| **Lean no setor público** (RADNOR; OSBORNE, 2013; PSOMAS; KERAMIDA; BOURANTA, 2022) | Lean falha quando vira caixa de ferramentas sem lógica de serviço público. A revisão de 56 artigos (2004-2020) agrupa os efeitos práticos em 14 temas, com destaque para fatores humanos e problemas de implantação | Limite de uso das ferramentas |
| **Lei de Little** (LITTLE, 1961; 2011) | L = λW: estoque médio = vazão média × tempo médio no sistema | Lead time a partir de WIP e vazão |
| **Kingman/VUT** (KINGMAN, 1961; HOPP; SPEARMAN, 2008) | Espera em fila ≈ Variabilidade × Utilização × Tempo | Por que não operar a 100% |
| **M/M/c e Erlang C** (SHORTLE et al., 2018; GANS; KOOLE; MANDELBAUM, 2003) | Probabilidade de espera e nível de serviço com c atendentes | Dimensionar atendimento ao público |
| **Análise quantitativa de fluxo** (DUMAS et al., 2018, cap. 7) | Tempo de ciclo por blocos: XOR (Σ pᵢTᵢ), AND (máximo), retrabalho (T/(1 − r)); lei de Little; filas | Base da conversão fluxo → carga |
| **Tipos de capacidade** (ANUPINDI et al., 2012; ARNOLD; CHAPMAN; CLIVE, 2017) | Capacidade teórica; capacidade nominal = tempo disponível × utilização × eficiência; capacidade demonstrada = média da produção realizada. Gargalo = recurso com a maior carga relativa à capacidade, contadas as visitas | Teórica × efetiva × demonstrada |

## 3. Conceitos e fórmulas

### 3.1 Notação mínima útil

Para análise de capacidade bastam os elementos da subclasse **Descritiva** da BPMN 2.0.2 (OMG, 2014, tab. 2.1):
piscina e raias (quem faz), tarefa, subprocesso, evento de início e de fim, fluxo de sequência, gateways exclusivo
(XOR) e paralelo (AND), objeto de dados e anotação. A subclasse **Analítica** acrescenta fluxo condicional e padrão,
gateways inclusivo e baseado em evento, tarefas em laço e de múltiplas instâncias. Use-a só se o laço precisar de
regra explícita.

Para o script, cada fluxo de sequência que sai de um gateway XOR recebe uma **probabilidade**. Cada ramo de um AND
recebe probabilidade 1. Com isso o diagrama vira a tabela `origem; destino; probabilidade`.

### 3.2 Capacidade, carga e utilização (por etapa *i*, no período)

- Horas da etapa: Hᵢ = pessoasᵢ × horas efetivas por pessoa (horas da Fase 7, já descontada a hora útil).
- **Capacidade efetiva**: Cᵢ = Hᵢ × 60 / tᵢ (unidades/período; tᵢ em minutos por execução).
- **Carga**: Lᵢ = dᵢ × tᵢ / 60 (horas), em que dᵢ é a demanda da etapa.
- **Utilização**: ρᵢ = Lᵢ / Hᵢ = dᵢ / Cᵢ. Se ρᵢ > 1, a etapa não dá conta e a fila cresce sem limite.
- **Pessoas necessárias**: Nᵢ = Lᵢ / (horas por pessoa × alvo). Com alvo = 1, é o Q do DFT aplicado à etapa.

### 3.3 Tipos de capacidade

| Tipo | Definição | Como se obtém |
|---|---|---|
| **Teórica** (de projeto) | Todas as horas brutas da jornada, ao tempo-padrão, sem perdas | `capacidades(...)["teorica"]` |
| **Efetiva** (nominal, *rated*) | Teórica × fator de disponibilidade (ausências, hora útil) × eficiência (ARNOLD; CHAPMAN; CLIVE, 2017) | `capacidades(...)["efetiva"]`. É a que o módulo usa |
| **Demonstrada** | Média da produção realizada em períodos anteriores | `capacidade_demonstrada(serie)`. Só mede capacidade se houve fila em todos os períodos |

Os manuais de operações chamam de **utilização** a razão produção real / capacidade teórica e de **eficiência** a
razão produção real / capacidade efetiva (`indicadores_capacidade`). Essa "utilização" não é a ρ das filas. No
relatório, "utilização" significa sempre ρ = carga / capacidade efetiva.

### 3.4 Gargalo e capacidade do processo

O gargalo é a etapa de **maior utilização** (ANUPINDI et al., 2012; ABPMP, 2013, §4.4.7). Não é a etapa mais lenta
nem a de menor capacidade absoluta, porque etapas com retrabalho recebem mais de uma visita por caso e etapas com mais
gente processam mais em paralelo. Para uma demanda de entrada D:

- capacidade do processo = D / ρ_max;
- vazão sustentável = min(D, D / ρ_max).

### 3.5 Takt time e tempo de ciclo

- **Takt** = tempo disponível / demanda (ROTHER; SHOOK, 1999). É o ritmo que a demanda impõe.
- **Tempo de ciclo da etapa** = tᵢ / pessoasᵢ. É o intervalo entre duas saídas com a equipe trabalhando em paralelo.
- Medindo o takt sobre o tempo disponível por pessoa, vale ρᵢ = ciclo / takt. Uma etapa "não acompanha o takt"
  exatamente quando ρᵢ > 1. Takt e utilização são a mesma informação dita de dois jeitos.

### 3.6 Lei de Little, lead time e eficiência do ciclo

- **WIP = vazão × lead time**, logo lead time = WIP / vazão (LITTLE, 1961). Exemplo: 180 processos em andamento e
  vazão de 60 por semana dão lead time de 3 semanas.
- **Tempo de atravessamento** (*lead time*, *throughput time*) é o tempo do caso no sistema, da entrada à saída,
  incluindo esperas. **Tempo de ciclo** pode significar duas coisas na literatura. Em VSM e manufatura, é o intervalo
  entre saídas da etapa (§3.5). Em BPM, Dumas et al. (2018) chamam de *cycle time* o tempo de atravessamento do caso.
  O módulo usa "tempo de ciclo" no primeiro sentido e "lead time" no segundo.
- **Eficiência do ciclo do processo** (PCE) = tempo de trabalho efetivo / lead time. Em serviços administrativos ela
  costuma ser baixa, porque o caso passa a maior parte do tempo parado em fila (GEORGE, 2003).
- Little vale para **médias de longo prazo em sistema estável**. Com demanda acima da capacidade o WIP cresce todo
  mês, e WIP / vazão é só uma fotografia do momento (LITTLE, 2011).

### 3.7 Utilização e filas

- **Kingman/VUT** (G/G/1), com a extensão para c servidores usada por Hopp e Spearman (2008):
  Wq ≈ [(cₐ² + cₑ²)/2] × [u^(√(2(c+1)) − 1) / (c(1 − u))] × tₑ.
  O termo u/(1 − u) vale 4 com u = 0,80, 9 com u = 0,90 e 19 com u = 0,95. A espera explode perto de 100%.
- **M/M/c (Erlang C):** P(espera) = C(c, a) com a = λ/μ. Wq = C/(cμ − λ). Nível de serviço
  P(Wq ≤ t) = 1 − C·e^(−(cμ−λ)t). É o padrão de dimensionamento de centrais de atendimento (GANS; KOOLE;
  MANDELBAUM, 2003).

### 3.8 Do fluxo à carga

O fluxo é tratado como uma cadeia de Markov absorvente. Seja P a matriz de probabilidades entre atividades e s o
vetor que sai do início. As **razões de visita** (execuções esperadas por caso) resolvem v = s + Pᵀv, isto é,
(I − Pᵀ)v = s. A formulação reproduz as fórmulas de bloco de Dumas et al. (2018): XOR dá Σ pᵢTᵢ e um laço de
retrabalho com probabilidade r dá 1/(1 − r) visitas. Ela também trata laços aninhados, que as fórmulas de bloco não
cobrem. Daí:

- demanda da atividade = D × vᵢ e carga = D × vᵢ × tᵢ / 60;
- **esforço unitário da entrega** E = Σ vᵢ tᵢ / 60 (horas por caso), o Eᵢ que vai para a Fase 6.

No AND, a duração do bloco é o maior ramo, mas o **trabalho** é a soma dos ramos. O DFT dimensiona trabalho, então
o paralelismo não reduz a carga.

## 4. Passo a passo

1. **Delimitar a entrega e o processo.** Use um SIPOC para pactuar fronteira, gatilho, saída e cliente. A fronteira
   é a mesma da Fase 6 ("começa quando…, termina quando…").
2. **Escolher o nível de detalhe.** Use o nível descritivo do guia do MGI (BRASIL, 2025), com 5 a 15 atividades.
   Detalhe só onde o perfil executor muda, onde há retorno (retrabalho) ou onde há espera relevante.
3. **Desenhar o fluxo mínimo.** Use a subclasse Descritiva: um início, um fim, tarefas com rótulo verbo + objeto,
   XOR para decisões e AND para trabalho em paralelo. Aplique as 7PMG.
4. **Atribuir probabilidades aos desvios.** Tire-as do sistema de protocolo (percentual de exigências, de
   indeferimentos, de vistorias). Na falta de registro, use estimativa da equipe, marcada como incerta.
5. **Coletar, por atividade:** perfil executor, tempo médio por execução (min), pessoas alocadas, horas efetivas
   disponíveis no período e, se houver, o WIP (casos parados na etapa na data de corte).
6. **Rodar o script.** Ele calcula razões de visita, demanda e carga por atividade, capacidade, utilização, gargalo,
   takt × tempo de ciclo, agregação por perfil, capacidade do processo, vazão e lead time por Little.
7. **Interpretar.** ρ > 1 é etapa estruturalmente sub-dimensionada. 0,85 ≤ ρ ≤ 1 é zona de fila. Utilização do perfil
   (agregada) abaixo de 1 com uma etapa acima de 1 indica problema de alocação, não de quadro.
8. **Registrar e devolver.** Leve o esforço unitário com retrabalho para a Fase 6, o perfil para a Fase 5, a nota
   "série histórica = teto de capacidade" para a Fase 4 e o diagnóstico de gargalo para o relatório da Fase 8.

## 5. Critérios de decisão

| Decisão | Critério |
|---|---|
| Mapear ou não? | Mapear se há (a) fila ou atraso recorrente, (b) retrabalho percebido acima de ~10%, (c) entrega com fases executadas por cargos diferentes ou (d) atendimento ao público com meta de espera. Se não, catálogo + esforço por entrega bastam (RODRIGUES et al., 2021) |
| Nível de notação | Descritiva por padrão; Analítica só para laços com regra explícita; Executável nunca é necessária para DFT |
| Qual capacidade usar | Efetiva (horas da Fase 7 / tempo). Teórica só para comparar. Demonstrada só para conferir e só em etapa que teve fila o tempo todo |
| Como achar o gargalo | Maior ρ = carga/capacidade, contadas as visitas. Nunca "a etapa mais demorada" |
| Utilização-alvo para pessoas necessárias | **1,0 por padrão** (coerente com Q = carga/H). Abaixo de 1 só com justificativa de fila e nível de serviço calculada (Erlang C ou VUT), nunca como desconto genérico |
| Modelo de fila | Atendimento presencial ou telefônico com chegadas aleatórias: M/M/c (Erlang C). Variabilidade conhecida (cₐ, cₑ ≠ 1) ou trabalho de retaguarda: VUT. Processos com prazo e lote: não use fila, use Little |
| Usar Little | Só com WIP e vazão medidos no mesmo recorte e com o sistema aproximadamente estável (entradas ≈ saídas). Com ρ_max > 1, reporte como fotografia |

## 6. Armadilhas

1. **Mapear tudo antes de catalogar.** O projeto atola no BPMN (dossiê L1). Mapeie só as entregas que passam nos
   critérios da §5.
2. **Usar o "caminho feliz".** No exemplo fictício, ignorar as exigências subestima o esforço por pedido em 11,5%
   (3,55 h contra 3,96 h).
3. **Confundir gargalo com déficit.** No exemplo, a análise técnica opera a 106%, mas os três arquitetos somados
   estão a 91%. Pôr as duas etapas em *pool* elimina o estouro sem contratação. É decisão de alocação, não de quadro.
4. **Somar durações em vez de trabalho no AND.** A duração do bloco paralelo é o maior ramo, mas o esforço é a soma.
5. **Dupla contagem de folga.** A hora útil da Fase 7 já desconta o tempo não produtivo. Aplicar por cima uma
   "utilização-alvo de 85%" sem fila que a justifique infla o quadro.
6. **Tratar produção histórica como demanda.** Onde há fila, a capacidade demonstrada coincide com a produção, e a
   demanda real está escondida no estoque. Compare entradas protocoladas com saídas.
7. **Confundir as duas "utilizações"**: produção/capacidade teórica (manuais de operações) e carga/capacidade efetiva
   (filas). Elas diferem pelo fator de disponibilidade.
8. **Aplicar Little com sistema instável ou recortes diferentes.** Por exemplo, WIP de um mês com vazão de outro, ou
   WIP que inclui casos suspensos aguardando o cidadão.
9. **Esquecer que tempo de ciclo tem dois sentidos** (intervalo entre saídas × tempo de atravessamento).
   Declare qual está usando.
10. **Lean como caixa de ferramentas.** Sem a lógica de serviço público, o redesenho corta etapas que geram valor
    para o cidadão (RADNOR; OSBORNE, 2013).

## 7. Como a etapa entra no DFT

| Fase (artigo-base) | Contribuição do módulo | Produto do aluno |
|---|---|---|
| 3 — Levantamento de produtos e serviços | SIPOC e fluxo mínimo definem fronteira e subdivisão em fases (os três critérios do artigo: unidade de medida diferente, executor diferente, execução longa) | Catálogo de entregas com fronteira |
| 4 — Estimativa de demanda | Diagnóstico de fila (ρ > 1, WIP crescente) mostra quando a série histórica é teto de capacidade | Nota de qualidade da série de demanda |
| 5 — Tipo de recurso | Raias e perfis por atividade; agregação de carga por perfil | Vinculação entrega ↔ cargo |
| 6 — Estimativa de esforço | Esforço unitário com desvios e retrabalho (E = Σ vᵢtᵢ) | Esforço por entrega |
| 7 — Disponibilidade | As horas efetivas por pessoa entram como `horas_disponiveis`. Erlang C dá o piso de cobertura para atendimento ao público (D5: Q_final = max(Q_carga, Q_cobertura)) | Disponibilidade e piso de cobertura |
| 8 — Consolidação | Gráfico capacidade × demanda e diagnóstico de gargalo como análise complementar ("outras análises além da definição do quadro", RODRIGUES et al., 2021) | Seção de diagnóstico do relatório |

Ponte com o exemplo de nove passos do curso: carga 7.770 h/ano e H = 928,2 h/ano dão Q = 8,37. Com o quadro atual de
7 servidores, a unidade opera a ρ = 7.770 / (7 × 928,2) = **119,6%** da capacidade efetiva. É a mesma conta vista
como utilização (reproduzida em `tests/test_capacidade.py`).

## 8. O que o script faz

| Função | Entrada | Saída |
|---|---|---|
| `capacidade_etapa`, `carga_horas`, `utilizacao`, `pessoas_necessarias` | pessoas, horas, tempo (min), demanda | capacidade (un./período), carga (h), ρ, pessoas |
| `takt_time`, `tempo_ciclo_etapa` | tempo disponível, demanda; tempo, pessoas | min/unidade |
| `capacidades`, `capacidade_demonstrada`, `indicadores_capacidade` | horas brutas, fator, eficiência; série de produção | teórica, efetiva, demonstrada; utilização e eficiência |
| `little`, `eficiencia_ciclo` | duas de WIP/vazão/lead time | a terceira; PCE |
| `kingman_espera`, `erlang_c`, `mmc`, `nivel_servico`, `servidores_para_meta` | u, cₐ², cₑ², tₑ, c; λ, μ, meta | espera em fila; medidas M/M/c; atendentes mínimos |
| `tempo_xor`, `tempo_and`, `tempo_retrabalho` | tempos e probabilidades | fórmulas de bloco (DUMAS et al., 2018) |
| `validar_fluxo`, `razao_visitas`, `carga_por_atividade`, `esforco_unitario` | transições origem; destino; probabilidade | avisos; visitas por caso; carga por atividade; E por caso |
| `analisar_capacidade`, `resumo_processo`, `por_perfil`, `relatorio`, `grafico_capacidade`, `grafico_curva_fila` | CSV de atividades (+ fluxo) | tabela por etapa, indicadores do processo, relatório em português, PNG |
| `cli(argv)` | `dft capacidade atividades.csv [--fluxo f.csv --demanda N] [--alvo 0,85] [--horas-totais] [--dias-periodo 22] [--grafico saida.png]` | relatório no terminal e gráfico |

Formato do CSV de atividades (`;` e vírgula decimal): `id; atividade; perfil; tempo_min; pessoas;
horas_disponiveis; demanda; wip`. `id` e `wip` são opcionais. Com `--fluxo`, a coluna `demanda` é recalculada.
`horas_disponiveis` são **por pessoa** no período. Use `--horas-totais` se já forem o total da etapa.

### Resultado do exemplo fictício (licenciamento de obra, 100 pedidos/mês, 77,35 h efetivas por pessoa/mês)

- Visitas por pedido: triagem 1,285; análise técnica 1,093; comunicação de exigência 0,357 (efeito do retrabalho).
- Gargalo: análise técnica, com ρ = 105,9%. Capacidade do processo de 94,4 pedidos/mês, e o estoque cresce cerca de
  5,6 pedidos/mês.
- Little: WIP 116 ÷ vazão 94,4 = lead time de 1,23 mês (≈ 27 dias úteis).
- Arquitetos em *pool*: capacidade do processo sobe para 110,3 pedidos/mês, com a equipe a 90,6% (zona de fila).
- Balcão (30 chegadas/h, 8 min): a carga pede 4 atendentes, mas o nível de serviço de 80% em até 10 min exige 5.

## Referências

ABPMP – ASSOCIATION OF BUSINESS PROCESS MANAGEMENT PROFESSIONALS. **BPM CBOK**: guia para o gerenciamento de processos
de negócio: corpo comum de conhecimento. Versão 3.0. 1. ed. [*S. l.*]: ABPMP Brasil, 2013. Disponível em:
https://edoc.ufam.edu.br/handle/123456789/8498. Acesso em: 5 nov. 2026.

ANUPINDI, Ravi; CHOPRA, Sunil; DESHMUKH, Sudhakar D.; VAN MIEGHEM, Jan A.; ZEMEL, Eitan. **Managing business process
flows**: principles of operations management. 3. ed. Upper Saddle River: Prentice Hall, 2012.

ARNOLD, J. R. Tony; CHAPMAN, Stephen N.; CLIVE, Lloyd M. **Introduction to materials management**. 8. ed. Boston:
Pearson, 2017.

BRASIL. Controladoria-Geral da União. **Metodologia de gestão de processos de negócio da CGU**. Versão 1.0. Brasília,
DF: CGU, 2020. Disponível em: https://repositorio.cgu.gov.br/bitstream/1/66337/3/Metodologia_de_Gestao_Processos_2020.pdf.
Acesso em: 5 nov. 2026.

BRASIL. Ministério da Gestão e da Inovação em Serviços Públicos. **Guia prático de gestão de processos**. 2. ed.
Brasília, DF: MGI, 2025. Disponível em:
https://www.gov.br/gestao/pt-br/acesso-a-informacao/estrategia-e-governanca/gestaodeprocessos/arquivos_pdf/guia-pratico-de-gestao-de-processos-1-1.pdf.
Acesso em: 5 nov. 2026.

BRASIL. Ministério do Planejamento, Orçamento e Gestão. Secretaria de Gestão. **Guia de gestão de processos de
governo**. Brasília, DF: MP, 2011. Notícia de lançamento disponível em:
https://serpro.gov.br/menu/noticias/noticias-antigas/guia-de-gestao-de-processos-de-governo-e-lancado-em-maio.
Acesso em: 5 nov. 2026.

DUMAS, Marlon; LA ROSA, Marcello; MENDLING, Jan; REIJERS, Hajo A. Quantitative process analysis. *In*: DUMAS, Marlon
*et al*. **Fundamentals of business process management**. 2. ed. Berlin: Springer, 2018. p. 255-296. DOI:
10.1007/978-3-662-56509-4_7. Disponível em: https://doi.org/10.1007/978-3-662-56509-4_7. Acesso em: 5 nov. 2026.

GANS, Noah; KOOLE, Ger; MANDELBAUM, Avishai. Telephone call centers: tutorial, review, and research prospects.
**Manufacturing & Service Operations Management**, Linthicum, v. 5, n. 2, p. 79-141, 2003. DOI:
10.1287/msom.5.2.79.16071. Disponível em: https://doi.org/10.1287/msom.5.2.79.16071. Acesso em: 5 nov. 2026.

GEORGE, Michael L. **Lean Six Sigma for service**: how to use lean speed and six sigma quality to improve services and
transactions. New York: McGraw-Hill, 2003.

HOPP, Wallace J.; SPEARMAN, Mark L. **Factory physics**. 3. ed. New York: McGraw-Hill/Irwin, 2008.

INTERNATIONAL ORGANIZATION FOR STANDARDIZATION; INTERNATIONAL ELECTROTECHNICAL COMMISSION. **ISO/IEC 19510:2013**:
information technology: Object Management Group Business Process Model and Notation. Geneva: ISO, 2013. Disponível
em: https://www.iso.org/standard/62652.html. Acesso em: 5 nov. 2026.

KINGMAN, J. F. C. The single server queue in heavy traffic. **Mathematical Proceedings of the Cambridge Philosophical
Society**, Cambridge, v. 57, n. 4, p. 902-904, 1961. DOI: 10.1017/S0305004100036094. Disponível em:
https://doi.org/10.1017/S0305004100036094. Acesso em: 5 nov. 2026.

LITTLE, John D. C. A proof for the queuing formula: L = λW. **Operations Research**, Baltimore, v. 9, n. 3,
p. 383-387, 1961. DOI: 10.1287/opre.9.3.383. Disponível em: https://doi.org/10.1287/opre.9.3.383. Acesso em: 5 nov. 2026.

LITTLE, John D. C. Little's law as viewed on its 50th anniversary. **Operations Research**, Linthicum, v. 59, n. 3,
p. 536-549, 2011. DOI: 10.1287/opre.1110.0940. Disponível em: https://doi.org/10.1287/opre.1110.0940. Acesso em:
5 nov. 2026.

MENDLING, Jan; REIJERS, Hajo A.; VAN DER AALST, Wil M. P. Seven process modeling guidelines (7PMG). **Information and
Software Technology**, Amsterdam, v. 52, n. 2, p. 127-136, 2010. DOI: 10.1016/j.infsof.2009.08.004. Disponível em:
https://doi.org/10.1016/j.infsof.2009.08.004. Acesso em: 5 nov. 2026.

OBJECT MANAGEMENT GROUP. **Business Process Model and Notation (BPMN)**: version 2.0.2. Needham: OMG, 2014.
(Documento formal/13-12-09). Disponível em: https://www.omg.org/spec/BPMN/2.0.2/PDF. Acesso em: 5 nov. 2026.

PSOMAS, Evangelos; KERAMIDA, Efthalia; BOURANTA, Nancy. Practical implications of Lean, Six Sigma and Lean Six Sigma in
the public administration sector: a systematic literature review. **International Journal of Lean Six Sigma**, Bingley,
v. 13, n. 6, p. 1277-1307, 2022. DOI: 10.1108/IJLSS-04-2021-0078. Disponível em:
https://doi.org/10.1108/IJLSS-04-2021-0078. Acesso em: 5 nov. 2026.

RADNOR, Zoe; OSBORNE, Stephen P. Lean: a failed theory for public services? **Public Management Review**, Abingdon,
v. 15, n. 2, p. 265-287, 2013. DOI: 10.1080/14719037.2012.748820. Disponível em:
https://doi.org/10.1080/14719037.2012.748820. Acesso em: 5 nov. 2026.

RODRIGUES, A. M.; CAVALCANTI, M. C. B.; SILVA, R. A. da; MACEDO, V. Proposta de metodologia de dimensionamento da
força de trabalho para o TCE-RJ. **Revista do TCE-RJ**, Rio de Janeiro, v. 2, n. 1, p. 72-83, 2021.

ROTHER, Mike; SHOOK, John. **Learning to see**: value stream mapping to add value and eliminate muda.
Brookline: Lean Enterprise Institute, 1999.

SHORTLE, John F.; THOMPSON, James M.; GROSS, Donald; HARRIS, Carl M. **Fundamentals of queueing theory**. 5. ed.
Hoboken: Wiley, 2018. DOI: 10.1002/9781119453765. Disponível em: https://doi.org/10.1002/9781119453765. Acesso em:
5 nov. 2026.
