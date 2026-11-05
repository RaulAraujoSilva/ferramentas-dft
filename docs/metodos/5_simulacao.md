# 5. Simulação de processos por Monte Carlo

> Módulo `dft_ferramentas/simulacao.py` · exemplo `exemplos/5_simulacao.py` · dados `dados/simulacao_*.csv` ·
> testes `tests/test_simulacao.py` · referências `docs/metodos/5_simulacao_referencias.json`.
> Dados sempre fictícios. Etapas do curso: **Fase 6 — Estimativa de esforço** (Aula 3, Produto 3) e
> **Fase 8 — Consolidação** (Produto 4), com a disponibilidade da **Fase 7** no denominador.

## 1. O problema que esta etapa resolve

O artigo-base manda fixar o esforço "no valor necessário a cobrir 80% dos menores tempos observados" e, "na ausência
de dados históricos de alocação", usar "a simulação de Monte Carlo para gerar uma base simulada de tempos"
(RODRIGUES et al., 2021, Fase 6). Em 2018 isso exigia um plugin pago de planilha. Hoje, bibliotecas abertas fazem o
mesmo (HARRIS et al., 2020).

O módulo responde a duas perguntas distintas, que o gestor costuma confundir:

1. **Quanto trabalho o setor tem por ano, com que grau de segurança?** É o Monte Carlo do esforço: soma, entrega por
   entrega, volume × tempo por execução, em 10.000 cenários, e lê P50, P80 e P90 da carga total em horas e em
   servidores.
2. **Com c servidores, quanto tempo o cidadão espera?** É a simulação de eventos discretos (SED) de um processo de
   atendimento. Duas equipes com a mesma carga anual podem ter filas muito diferentes, porque a fila depende da
   variabilidade e da utilização, não só da média.

A primeira pergunta dimensiona o quadro. A segunda mostra o efeito do quadro no nível de serviço. A partir de cerca
de 85% de utilização, um servidor a mais muda o prazo de dias para horas.

## 2. Padrões usuais e fontes

### 2.1 O método de Monte Carlo

Metropolis e Ulam (1949) formalizaram a ideia: quando a grandeza de interesse é difícil de calcular por fórmula,
sorteiam-se muitas realizações do modelo e usa-se a distribuição empírica dos resultados. Rubinstein e Kroese (2017)
descrevem o procedimento padrão:

1. gerar números pseudoaleatórios uniformes;
2. transformá-los nas variáveis de entrada, de preferência por **transformada inversa**, `X = F⁻¹(U)`;
3. avaliar o modelo determinístico;
4. repetir N vezes;
5. estimar médias, quantis e probabilidades com seus erros.

Ichsan, Isvara e Karim (2025) aplicam a mesma sequência a um cronograma de obra: três pontos por especialista,
distribuição PERT, 10.000 iterações e leitura em 80%, 90% e 95%. É o desenho do artigo-base, aplicado em outro setor.

### 2.2 Número de replicações e erro de Monte Carlo do percentil

Um percentil simulado é uma estimativa e tem erro. Há três formas usuais de medi-lo.

- **Fórmula assintótica** (DAVID; NAGARAJA, 2003): `SE(Q̂p) ≈ √(p(1−p)/N) / f(Qp)`, em que `f` é a densidade no
  quantil. No CT-1 (beta-PERT 4; 6; 16), `f(9,075) = 0,107836`. Com N = 10.000, `SE = 0,0371 h`, cerca de 2 minutos.
- **IC por estatísticas de ordem, sem supor distribuição.** O número de amostras abaixo do verdadeiro `Qp` segue uma
  `Binomial(N, p)`. Os postos `j = B⁻¹(α/2)` e `k = B⁻¹(1−α/2)+1` dão o intervalo `[x(j); x(k)]`. É o método do
  script para a carga total, cuja densidade não é conhecida.
- **Número de iterações pela precisão desejada** (LAW, 2015): `N = (z/h)²·p(1−p)/f²`. Para ±0,0727 h com 95%, N ≈
  10.000. Para ±0,02 h, N ≈ 132.000. Ichsan et al. (2025) usam `n = (3σ/ε)²`, regra válida para a **média**, não para
  percentis. O script oferece as duas regras e avisa a diferença.

O erro de Monte Carlo cai com `1/√N`. Multiplicar N por 100 reduz o erro só 10 vezes. Por isso o curso adota
N = 10.000: a incerteza que domina é a dos três pontos informados, não a do motor (L2, §3.7).

**Quantil amostral.** Há nove definições em uso (HYNDMAN; FAN, 1996). O script usa a do tipo 7 (interpolação linear),
padrão do NumPy, do R e do motor TypeScript do Sistema DFT. Com N = 10.000, a diferença entre os tipos é desprezível.

### 2.3 Geração de variáveis e semente

Usa-se um gerador testado (MATSUMOTO; NISHIMURA, 1998). O NumPy atual usa o PCG64 com `SeedSequence`. A **semente**
fica gravada no relatório: é um requisito de auditoria, para que a mesma entrada dê o mesmo quadro hoje e daqui a três
anos (L2, §3.7).

O Sistema DFT em TypeScript usa o gerador mulberry32. O script em Python usa o PCG64. Por isso os sorteios diferem,
mas os percentis coincidem dentro do erro de Monte Carlo, como o teste do CT-1 confere.

Na simulação de filas, cada finalidade tem o seu fluxo aleatório: chegadas, tempo de serviço de cada etapa e decisão de
retrabalho. Assim, configurações diferentes (1, 2, 3… servidores) consomem os mesmos números para os mesmos fins. É a
técnica de **números aleatórios comuns** (LAW, 2015; RUBINSTEIN; KROESE, 2017). Ela faz a diferença entre as
alternativas refletir a mudança de capacidade, e não o acaso.

### 2.4 Correlação entre entradas e a advertência da carteira

Se os tempos das entregas forem sorteados de forma independente, os desvios se compensam: um parecer demorado num
cenário convive com uma vistoria rápida. Daí a advertência do curso (L2, §3.8): **a soma dos P80 individuais é maior
que o P80 da soma**, salvo correlação perfeita. Quanto mais entregas distintas, maior a diferença. A regra oficial do
quadro (Σ demanda × P80) é, portanto, **conservadora por construção**. Uma ressalva técnica: o percentil não é uma
medida subaditiva em geral. Em carteiras de tempos assimétricos à direita, porém, a desigualdade vale na prática, e o
script mede o seu tamanho em cada caso.

Na vida real, os tempos têm causas comuns: sistema lento, equipe recém-chegada, lei nova. Há dois padrões para
representá-las.

- **Correlação de postos induzida.** Iman e Conover (1982) reordenam as amostras para atingir a correlação de
  Spearman desejada sem alterar as marginais. Esse é o padrão das ferramentas de risco em planilha e de estudos de
  confiabilidade (HELTON; DAVIS, 2003). A alternativa equivalente é a **cópula gaussiana** (NELSEN, 2006): sorteiam-se
  normais correlacionadas `Z·Lᵀ`, aplica-se `U = Φ(Z)` e usa-se a transformada inversa. A relação
  `ρ_Spearman = (6/π)·arcsen(r/2)` converte a correlação de postos pedida na correlação das normais.
- **Direcionadores de risco** (AACE 57R-09; HULETT; NOSBISCH, 2012). Em vez de pedir coeficientes ao especialista
  ("estimar correlação por opinião é difícil e problemático"), modela-se o evento comum que afeta várias atividades, e
  a correlação aparece sozinha. A AACE retitulou em 2021 a RP 41R-08 ("estimativa por faixas") e só admite o Monte
  Carlo item a item, com correlação explícita, para a incerteza inerente de projetos bem definidos (RP 118R-21).

O script implementa as duas formas de correlação de postos, com coeficiente uniforme ou matriz. Na prática do DFT,
use valores de 0,3 a 0,6 como **cenário de sensibilidade** e registre a justificativa.

### 2.5 Análise de sensibilidade: tornado e correlação de postos

O padrão de prática do PMI para riscos (PROJECT MANAGEMENT INSTITUTE, 2009) e os manuais de risco (VOSE, 2008)
apresentam a sensibilidade num **gráfico de tornado**. A literatura de sensibilidade distingue duas medidas (HAMBY,
1994; SALTELLI et al., 2008):

- **local, um fator de cada vez:** a entrega vai do seu P10 ao P90 e as demais ficam no P50. É a barra do tornado.
- **global, sobre a própria amostra de Monte Carlo:** a correlação de postos de Spearman entre a entrada e a saída. O
  quadrado normalizado aproxima a "contribuição à variância" de cada entrada. Vale para modelos monotônicos, como a
  soma de horas.

O script calcula as duas. O tornado responde à pergunta prática "onde vale a pena refinar a coleta?". Quase sempre a
resposta é uma ou duas entregas de alto volume e alta dispersão.

### 2.6 Simulação de risco de cronograma e esforço

A gestão de projetos consolidou a mesma prática. O PMI (2009) recomenda o Monte Carlo sobre estimativas de faixa para
obter a distribuição do prazo e do custo e fixar reservas por nível de confiança. A AACE 57R-09 simula prazo e custo
juntos e prioriza os riscos "no nível de confiança P-80" (HULETT; NOSBISCH, 2012). A analogia com o DFT é direta: o
"projeto" é o ano de trabalho do setor, as "atividades" são as entregas e a "reserva" é a folga de horas embutida no
P80.

### 2.7 Simulação de eventos discretos de processos e filas

Law (2015) e Banks et al. (2010) definem o padrão:

- **lista de eventos futuros** com avanço para o próximo evento (chegada, fim de serviço);
- **estatísticas por média no tempo**, como fila média e ocupação;
- **período de aquecimento** descartado, porque o sistema começa vazio e não representativo. O procedimento de Welch
  ajuda a escolhê-lo: média móvel, entre replicações, do atravessamento por ordem de chegada;
- **replicações independentes**, com IC pela t de Student entre as médias das replicações. Os casos de uma mesma
  replicação são autocorrelacionados, e por isso o IC "por caso" seria falso.

O SimPy é a biblioteca aberta de referência em Python. Modela cada caso como um processo e cada equipe como um
`Resource`. O script traz um **motor próprio** (lista de eventos com `heapq`, sem dependência) e o mesmo modelo
escrito em **SimPy**, usado para verificação cruzada. Com os mesmos fluxos aleatórios, os dois dão resultados
idênticos.

### 2.8 Verificação e validação (V&V)

Sargent (2013) separa três coisas:

- **validade do modelo conceitual:** as hipóteses representam o processo?
- **verificação:** o código faz o que o modelo diz?
- **validade operacional:** a saída se comporta como o sistema real?

O módulo usa as técnicas que Sargent lista:

- **comparação com modelo analítico:** a espera simulada numa fila M/M/2 bate com a fórmula de Erlang C (SHORTLE et
  al., 2018), e o P80 simulado do CT-1 bate com o quantil exato;
- **comparação entre implementações:** motor próprio contra SimPy;
- **lei de Little** (LITTLE, 1961), `L = λW`, como identidade que todo sistema estável cumpre;
- **utilização simulada** comparada à carga teórica das equações de tráfego de Jackson, com retrabalho;
- **casos extremos:** com utilização teórica de 100% ou mais, o sistema é marcado como instável;
- **validade de face com o gestor:** "uma fila de 8 h com dois analistas parece com o que vocês vivem?"

A validação operacional com dados históricos (comparar o atravessamento simulado com o do sistema de protocolo)
fica como etapa recomendada sempre que houver registros.

### 2.9 O percentil como apetite ao risco (P80, decisão D3)

Nenhuma fonte deriva o número 80. O artigo-base o adota. Ichsan et al. (2025) leem 80%, 90% e 95%. Flyvbjerg,
Garbuio e Lovallo (2009) mostram um governo que escolhe entre P50 e P80 como política declarada. O curso trata o
percentil como **decisão de governança**, tomada na Fase 1 e antes de ver o resultado. O script reporta P50, P80 e P90
lado a lado, para que o custo de cada nível de segurança fique visível. A decisão D3 manda usar a beta-PERT de Vose
como padrão e a log-normal deslocada como opção avançada, sempre com as duas comparadas. O script aceita as duas
distribuições, e a triangular só para sensibilidade.

## 3. Passo a passo

**(a) Monte Carlo do esforço anual do setor**

1. **Montar o catálogo** (Fase 3), com uma linha por entrega. O volume anual vem da Fase 4: um número fixo ou três
   pontos. O tempo por execução vem da Fase 6: mínimo, mais provável e máximo. O mais provável é a **moda** (D1).
2. **Escolher a distribuição** de cada tempo (D3):
    - beta-PERT por padrão;
    - log-normal deslocada se o gestor disser que "existe um caso, ainda que raro, que demoraria mais que o máximo";
    - triangular só para sensibilidade.
3. **Fixar N e a semente**: N = 10.000; a semente é registrada.
4. **Decidir a correlação.** Use 0 como base e 0,3 a 0,6 como cenário, com justificativa (causa comum).
5. **Rodar** e ler P50, P80 e P90 da carga total, o erro de Monte Carlo do P80 e a margem "soma dos P80 − P80 da
   soma".
6. **Converter em servidores**, dividindo pela disponibilidade efetiva da Fase 7: 1.153,60 h no CT-3 e 928,2 h no
   exemplo de nove passos. Arredonde para cima e mostre o valor fracionário ao lado.
7. **Ler o tornado** e voltar à coleta das entregas que dominam a variância.

**(b) Simulação de eventos discretos do atendimento**

1. **Desenhar o fluxo:** etapas em série, servidores por etapa, tempo por etapa (três pontos) e retrabalho (probabilidade
   e etapa de retorno).
2. **Estimar a taxa de chegadas** em casos por hora útil. O processo de Poisson é a hipótese padrão para chegadas
   independentes de muitos cidadãos.
3. **Conferir a carga teórica** por etapa, `λ·visitas·E[S]`. Se ela for maior ou igual ao número de servidores, a
   etapa é instável e nenhuma simulação a salva.
4. **Definir o horizonte, o aquecimento e as replicações.** Padrão: 2.000 h úteis, 200 h de aquecimento e
   10 replicações. Confira o aquecimento com `curva_welch`.
5. **Varrer 1, 2, 3… servidores** na etapa gargalo, com números aleatórios comuns. Leia o atravessamento médio e o
   P90, a fila, a espera, a utilização e o percentual de casos dentro do prazo.
6. **Verificar** pela lei de Little e, quando possível, validar com dados do protocolo.

## 4. Fórmulas

| Grandeza | Fórmula |
|---|---|
| Carga anual (modo "comum") | `H = Σᵢ Vᵢ · Tᵢ`, com um sorteio de `Tᵢ` por cenário |
| Carga anual (modo "independente") | `Hᵢ ≈ Normal(Vᵢ·μᵢ, Vᵢ·σᵢ²)` (teorema central do limite, `Vᵢ ≥ 30`) |
| Servidores | `Q = H_p / D`, inteiro = `⌈Q⌉` (`D` = disponibilidade efetiva) |
| Erro-padrão do quantil | `√(p(1−p)/N) / f(Qp)` |
| IC do quantil (sem distribuição) | `[x(j); x(k)]`, `j = B⁻¹(α/2; N, p)`, `k = B⁻¹(1−α/2; N, p) + 1` |
| N para precisão `h` | `N = (z/h)² · p(1−p) / f²` |
| Cópula gaussiana | `U = Φ(Z·Lᵀ)`, `L·Lᵀ = R`, `rᵢⱼ = 2·sen(π·ρᵢⱼ/6)` |
| Margem de conservadorismo | `Σᵢ P80(Hᵢ) − P80(Σᵢ Hᵢ)` |
| Carga por etapa (rede de Jackson) | `aₖ = λ · vₖ · E[Sₖ]`, `v = e₁ᵀ(I − P)⁻¹`; utilização `aₖ/cₖ` |
| Erlang C (validação M/M/c) | `Wq = C(c, a) / (cμ − λ)` |
| Lei de Little | `L = λ · W` |
| IC entre replicações | `x̄ ± t(0,975; R−1) · s/√R` |

**Modo "comum" × "independente": uma decisão de método.** A regra do curso, demanda × esforço por execução, supõe que
todas as execuções de uma entrega num ano compartilham o mesmo tempo sorteado. Isso equivale a tratar a incerteza dos
três pontos como incerteza sobre o **tempo típico do setor**: "6 h por parecer" pode, na verdade, ser 7 h. É o que o
script faz por padrão. Se, ao contrário, a dispersão for só a variação de um caso para outro, as 2.637 execuções se
compensam ao longo do ano e a carga quase não varia: no exemplo, o P80 cai de 35.055 h para cerca de 31.196 h. A
realidade fica entre os dois casos. O script mostra os dois números para que a margem seja vista, e não escondida.

## 5. Critérios de decisão

| Decisão | Critério usual | Escolha do curso |
|---|---|---|
| N | Precisão do quantil `N = (z/h)²p(1−p)/f²`; parar quando o P80 estabiliza | 10.000 (100.000 para relatório final) |
| Distribuição do tempo | Forma e cauda declaradas pelo gestor | beta-PERT (Vose); log-normal deslocada se houver cauda sem teto (D3) |
| Percentil | Apetite ao risco declarado antes do cálculo | P80, com P50 e P90 exibidos |
| Correlação | Causa comum identificada? Se sim, direcionador ou ρ de postos | 0 como base; 0,3 a 0,6 como sensibilidade justificada |
| Cópula × Iman-Conover | Iman-Conover preserva as marginais exatas; a cópula é mais simples e exata em postos | cópula (padrão), Iman-Conover para conferência |
| Agregação das execuções | Incerteza epistêmica (tempo típico) × aleatória (caso a caso) | "comum" (regra do quadro); "independente" como leitura complementar |
| Motor de fila | Sem dependência × biblioteca consagrada | próprio (padrão); SimPy para verificação |
| Aquecimento | Curva de Welch estabiliza | 10% do horizonte, conferido pela curva |
| Replicações | Meia-largura do IC aceitável para a decisão | 10 (aumentar se o IC cruzar a meta) |
| Servidores na fila | Menor c com utilização < 85% e nível de serviço na meta | lido na varredura, nunca só pela média |

## 6. Armadilhas

1. **Confundir precisão do motor com precisão do resultado.** Um P80 com erro de 2 minutos não corrige três pontos
   inventados. O Monte Carlo não melhora dado ruim (L1, Parte 5).
2. **Somar P80 e chamar o resultado de "P80 do setor".** A soma é mais conservadora. Reporte as duas leituras e a
   margem (L2, §3.8).
3. **Supor independência onde há causa comum** e subestimar a cauda do total. Faça o cenário com correlação.
4. **Pedir coeficientes de correlação de cabeça ao gestor.** Prefira identificar o evento comum (direcionador de risco).
5. **Escolher o percentil depois de ver o resultado.** O percentil é política (D3), declarada na Fase 1.
6. **Não registrar a semente.** O resultado fica irreprodutível e não resiste à auditoria.
7. **Usar `n = (3σ/ε)²` para percentis.** Essa regra controla o erro da média, não o da cauda.
8. **Simular filas com utilização ≥ 100%.** A fila cresce sem limite, e os números refletem só o horizonte escolhido.
9. **Não descartar o aquecimento**, ou calcular o IC com os casos de uma só replicação como se fossem independentes.
10. **Dimensionar a fila pela média.** Com 94% de utilização, o atravessamento médio pode até caber no prazo, mas o
    P90 estoura. Olhe o percentual no prazo.
11. **Esquecer o retrabalho.** Uma devolução de 15% aumenta a carga da etapa de análise em cerca de 18% (1/0,85).
12. **Tomar o modelo como validado porque "rodou".** Verifique (Little, Erlang C, dois motores) e valide com o gestor
    e com dados históricos (SARGENT, 2013).

## 7. Como a etapa entra no DFT

- **Fase 6 (esforço), Aula 3, Produto 3 (planilha de estimativas).** O aluno informa três pontos por entrega, roda o
  Monte Carlo de cada uma (P80 unitário), compara beta-PERT e log-normal (D3) e anexa o histograma com a acumulada e o
  P80 marcado. É a tradução visual da Fase 6 do artigo-base.
- **Fase 7 (disponibilidade).** A disponibilidade efetiva entra como denominador: 1.153,60 h no CT-3 e 928,2 h no
  exemplo de nove passos.
- **Fase 8 (consolidação), Produto 4 (quadro em pelo menos duas visões).** O quadro oficial é Σ demanda × P80 ÷
  disponibilidade (D6). O Monte Carlo conjunto acrescenta, como indicadores secundários: o P80 da carga total, a
  margem de conservadorismo em horas e em servidores, o tornado e, para processos com fila, a curva "servidores ×
  nível de serviço". Esta última é a visão que convence o gestor de que **número de servidores é decisão de prazo ao
  cidadão**.
- **Ligação com o módulo 1 (capacidade).** A análise de capacidade dá a utilização média. A SED mostra o que a média
  não mostra: espera, P90 e variabilidade.

## 8. Exemplo numérico (fictício, `dados/simulacao_entregas.csv`, semente 20261105)

Seis entregas de uma Secretaria de Administração fictícia: pareceres, certidões, recursos (log-normal), vistorias,
balcão e reunião semanal (atividade indireta, D2).

| Indicador | Valor |
|---|---|
| Carga total P50 / P80 / P90 | 29.544 h / 35.055 h / 38.425 h |
| Erro de Monte Carlo do P80 | ≈ 105 h (0,30%); IC 95% [34.848; 35.246] h |
| Servidores com 1.153,60 h (P50 / P80 / P90) | 25,61 → 26 / 30,39 → 31 / 33,31 → 34 |
| Servidores com 928,2 h (P80) | 37,77 → 38 |
| Soma dos P80 por entrega | 37.136 h, ou seja, margem de 2.080 h (5,6%) = 1,80 servidor |
| Tornado | Pareceres respondem por cerca de 92% da variância (ρ = 0,94) |
| P80 com ρ = 0 / 0,3 / 0,6 / 1,0 | 35.055 / 35.898 / 36.509 / 37.203 h |

Fila de atendimento (`dados/simulacao_processo.csv`, 1,3 caso/h, prazo de 8 h), variando os analistas da etapa
"Análise técnica" (carga teórica de 1,87 servidor, com 15% de retrabalho):

| c | ρ | W médio | W P90 | no prazo |
|---|---|---|---|---|
| 1 | 187% | instável | instável | 0% |
| 2 | 94% | 8,67 ± 2,79 h | 17,95 h | 64,1% |
| 3 | 62% | 2,29 ± 0,04 h | 3,84 h | 99,5% |
| 4 | 47% | 2,03 ± 0,02 h | 3,34 h | 99,9% |

A carga média pede 1,87 analista, mas com 2 a fila leva um terço dos casos para fora do prazo. O terceiro analista
"sobra" em horas e é exatamente o que compra o nível de serviço. Verificação: L = 11,31 ≈ λW = 11,32 (Little).

**Números canônicos reproduzidos pelos testes:**

| Caso | Valor esperado | Resultado do teste |
|---|---|---|
| CT-1, P80 analítico | 9,074998 h | igual, com tolerância de 1e-6 |
| CT-1, P80 simulado (N = 10.000, semente 42) | dentro de 0,10 h e de 3 erros-padrão (SE = 0,0371 h) | 9,0818 h; o IC 95% contém 9,074998 |
| CT-2, P80 | 10,920644 h | igual |
| CT-3, quadro | 28.432,51 h ÷ 1.153,60 = 24,646767 → 25 | igual |
| Exemplo de nove passos | 7.770 ÷ 928,2 = 8,37 | igual |

## 9. O que o script faz

| Função | O que faz |
|---|---|
| `TresPontos`, `quantil_analitico` | Beta-PERT (Vose), log-normal deslocada e triangular a partir de três pontos. Usa `dft_ferramentas.distribuicoes` se presente, com reserva local |
| `simular_esforco` | Monte Carlo da carga anual: volume × tempo, correlação (`copula` ou `iman_conover`), agregação `comum` ou `independente`, semente |
| `erro_mc_p80`, `erro_padrao_quantil`, `ic_quantil_ordem`, `n_para_precisao_quantil`, `n_iteracoes_ichsan`, `convergencia_p80` | Erro de Monte Carlo do percentil, IC sem distribuição, N necessário e diagnóstico de convergência |
| `soma_p80_analitica`, `servidores` | Regra oficial do quadro e conversão em servidores (arredondamento para cima) |
| `tornado` | Spearman, contribuição à variância e balanço P10–P90 por entrega |
| `uniformes_correlacionadas`, `iman_conover`, `matriz_correlacao` | Correlação de postos entre entradas |
| `Etapa`, `simular_processo`, `varrer_servidores` | Simulação de eventos discretos com chegadas Poisson, c servidores, tempos estocásticos, retrabalho, aquecimento, replicações, IC e nível de serviço |
| `carga_por_etapa`, `erlang_c`, `curva_welch` | Carga teórica (Jackson), validação M/M/c e escolha do aquecimento |
| `grafico_esforco`, `grafico_servidores` | Histograma com acumulada e P80 marcado, mais o tornado; curva de servidores × atravessamento e utilização |
| `cli` | `dft simulacao dados/simulacao_entregas.csv --disponibilidade 1153.6 928.2 --grafico saida.png`; `dft simulacao dados/simulacao_processo.csv --chegadas 1.3 --servidores 1-5 --prazo 8 [--motor simpy]` |

## Referências

AACE INTERNATIONAL. **Recommended Practice No. 41R-08**: risk analysis and contingency determination using range
estimating. Morgantown: AACE International, 2008. Retitulada em 2021 como *Understanding estimate ranging*; método
sucedido pela RP 118R-21.

AACE INTERNATIONAL. **Recommended Practice No. 57R-09**: integrated cost and schedule risk analysis using risk drivers
and Monte Carlo simulation of a CPM model. Morgantown: AACE International, 2011 (rev. 2022).

BANKS, Jerry; CARSON II, John S.; NELSON, Barry L.; NICOL, David M. **Discrete-event system simulation**. 5. ed.
Upper Saddle River: Pearson, 2010. ISBN 978-0-13-606212-7.

DAVID, H. A.; NAGARAJA, H. N. **Order statistics**. 3. ed. Hoboken: Wiley, 2003. (Wiley Series in Probability and
Statistics). DOI: 10.1002/0471722162. Disponível em: https://doi.org/10.1002/0471722162. Acesso em: 5 nov. 2026.

FLYVBJERG, Bent; GARBUIO, Massimo; LOVALLO, Dan. Delusion and deception in large infrastructure projects: two models
for explaining and preventing executive disaster. **California Management Review**, v. 51, n. 2, p. 170-194, 2009.
DOI: 10.2307/41166485. Disponível em: https://doi.org/10.2307/41166485. Acesso em: 5 nov. 2026.

HAMBY, D. M. A review of techniques for parameter sensitivity analysis of environmental models. **Environmental
Monitoring and Assessment**, v. 32, n. 2, p. 135-154, 1994. DOI: 10.1007/BF00547132. Disponível em:
https://doi.org/10.1007/BF00547132. Acesso em: 5 nov. 2026.

HARRIS, Charles R. et al. Array programming with NumPy. **Nature**, v. 585, n. 7825, p. 357-362, 2020. DOI:
10.1038/s41586-020-2649-2. Disponível em: https://doi.org/10.1038/s41586-020-2649-2. Acesso em: 5 nov. 2026.

HELTON, J. C.; DAVIS, F. J. Latin hypercube sampling and the propagation of uncertainty in analyses of complex
systems. **Reliability Engineering & System Safety**, v. 81, n. 1, p. 23-69, 2003. DOI:
10.1016/S0951-8320(03)00058-9. Disponível em: https://doi.org/10.1016/S0951-8320(03)00058-9. Acesso em: 5 nov. 2026.

HULETT, David T.; NOSBISCH, Michael R. Integrated cost and schedule using Monte Carlo simulation of a CPM model -
12419. *In*: WM2012 CONFERENCE, 2012, Phoenix. **Proceedings** [...]. Tempe: WM Symposia, 2012. Disponível em:
https://www.osti.gov/biblio/22293653. Acesso em: 5 nov. 2026.

HYNDMAN, Rob J.; FAN, Yanan. Sample quantiles in statistical packages. **The American Statistician**, v. 50, n. 4,
p. 361-365, 1996. DOI: 10.1080/00031305.1996.10473566. Disponível em:
https://doi.org/10.1080/00031305.1996.10473566. Acesso em: 5 nov. 2026.

ICHSAN, Mohammad; ISVARA, Wisnu; KARIM, Syaeful. Monte Carlo simulation for enhancing the schedule completion
forecast of Jakarta Central Railway Station construction project. **Applied Sciences**, v. 15, n. 13, 7464, 2025.
DOI: 10.3390/app15137464. Disponível em: https://doi.org/10.3390/app15137464. Acesso em: 5 nov. 2026.

IMAN, Ronald L.; CONOVER, W. J. A distribution-free approach to inducing rank correlation among input variables.
**Communications in Statistics - Simulation and Computation**, v. 11, n. 3, p. 311-334, 1982. DOI:
10.1080/03610918208812265. Disponível em: https://doi.org/10.1080/03610918208812265. Acesso em: 5 nov. 2026.

LAW, Averill M. **Simulation modeling and analysis**. 5. ed. New York: McGraw-Hill Education, 2015. ISBN
978-0-07-340132-4.

LITTLE, John D. C. A proof for the queuing formula: L = λW. **Operations Research**, v. 9, n. 3, p. 383-387, 1961.
DOI: 10.1287/opre.9.3.383. Disponível em: https://doi.org/10.1287/opre.9.3.383. Acesso em: 5 nov. 2026.

MALCOLM, D. G.; ROSEBOOM, J. H.; CLARK, C. E.; FAZAR, W. Application of a technique for research and development
program evaluation. **Operations Research**, v. 7, n. 5, p. 646-669, 1959. DOI: 10.1287/opre.7.5.646. Disponível em:
https://doi.org/10.1287/opre.7.5.646. Acesso em: 5 nov. 2026.

MATSUMOTO, Makoto; NISHIMURA, Takuji. Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random
number generator. **ACM Transactions on Modeling and Computer Simulation**, v. 8, n. 1, p. 3-30, 1998. DOI:
10.1145/272991.272995. Disponível em: https://doi.org/10.1145/272991.272995. Acesso em: 5 nov. 2026.

McKAY, M. D.; BECKMAN, R. J.; CONOVER, W. J. A comparison of three methods for selecting values of input variables in
the analysis of output from a computer code. **Technometrics**, v. 21, n. 2, p. 239-245, 1979. DOI:
10.1080/00401706.1979.10489755. Disponível em: https://doi.org/10.1080/00401706.1979.10489755. Acesso em: 5 nov. 2026.

METROPOLIS, Nicholas; ULAM, S. The Monte Carlo method. **Journal of the American Statistical Association**, v. 44,
n. 247, p. 335-341, 1949. DOI: 10.1080/01621459.1949.10483310. Disponível em:
https://doi.org/10.1080/01621459.1949.10483310. Acesso em: 5 nov. 2026.

NELSEN, Roger B. **An introduction to copulas**. 2. ed. New York: Springer, 2006. (Springer Series in Statistics).
DOI: 10.1007/0-387-28678-0. Disponível em: https://doi.org/10.1007/0-387-28678-0. Acesso em: 5 nov. 2026.

PROJECT MANAGEMENT INSTITUTE. **Practice standard for project risk management**. Newtown Square: PMI, 2009. ISBN
978-1-933890-38-8.

RODRIGUES, Adriana Maciel; CAVALCANTI, Marcos do Couto Bezerra; ARAÚJO DA SILVA, Raul; MACEDO, Valéria. Proposta de
metodologia de dimensionamento da força de trabalho para o TCE-RJ. **Revista do Tribunal de Contas do Estado do Rio de
Janeiro**, Rio de Janeiro, v. 2, n. 1, p. 72-83, 2021. DOI: 10.70690/xt46c612. Disponível em:
https://doi.org/10.70690/xt46c612. Acesso em: 5 nov. 2026.

RUBINSTEIN, Reuven Y.; KROESE, Dirk P. **Simulation and the Monte Carlo method**. 3. ed. Hoboken: Wiley, 2017.
(Wiley Series in Probability and Statistics). DOI: 10.1002/9781118631980. Disponível em:
https://doi.org/10.1002/9781118631980. Acesso em: 5 nov. 2026.

SALTELLI, Andrea; RATTO, Marco; ANDRES, Terry; CAMPOLONGO, Francesca; CARIBONI, Jessica; GATELLI, Debora; SAISANA,
Michaela; TARANTOLA, Stefano. **Global sensitivity analysis**: the primer. Chichester: Wiley, 2008. DOI:
10.1002/9780470725184. Disponível em: https://doi.org/10.1002/9780470725184. Acesso em: 5 nov. 2026.

SARGENT, R. G. Verification and validation of simulation models. **Journal of Simulation**, v. 7, n. 1, p. 12-24,
2013. DOI: 10.1057/jos.2012.20. Disponível em: https://doi.org/10.1057/jos.2012.20. Acesso em: 5 nov. 2026.

SHORTLE, John F.; THOMPSON, James M.; GROSS, Donald; HARRIS, Carl M. **Fundamentals of queueing theory**. 5. ed.
Hoboken: Wiley, 2018. (Wiley Series in Probability and Statistics). DOI: 10.1002/9781119453765. Disponível em:
https://doi.org/10.1002/9781119453765. Acesso em: 5 nov. 2026.

SIMPY. **SimPy**: discrete event simulation for Python. Documentação. [*S. l.*]: Read the Docs, 2026. Disponível em:
https://simpy.readthedocs.io/. Acesso em: 5 nov. 2026.

VOSE, David. **Risk analysis**: a quantitative guide. 3. ed. Chichester: Wiley, 2008. ISBN 978-0-470-51284-5.
