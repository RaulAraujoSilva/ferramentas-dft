# 2.1 Identificação da curva de distribuição de tempo das atividades

> Módulo `dft_ferramentas/distribuicoes.py` · exemplo `exemplos/2_1_distribuicoes.py` · dados `dados/distribuicoes_*.csv` ·
> testes `tests/test_distribuicoes.py` · referências `docs/metodos/2_1_distribuicoes_referencias.json`.
> Dados sempre fictícios. Etapa do curso: **Fase 6 — Estimativa de esforço** (Aula 3, Bloco B), alimentando o
> **Produto 3** (planilha de estimativas do setor-recorte: demanda projetada + esforço por entrega) e, pela fórmula do
> quadro, a **Fase 8**.

## 1. O problema que esta etapa resolve

O quadro de pessoal depende de quanto tempo custa cada entrega. Esse tempo **varia**: o mesmo parecer leva 4 horas num
caso simples e 16 num caso enrolado. O artigo-base resolve a variabilidade lendo um percentil alto: o esforço unitário é
"o valor necessário a cobrir 80% dos menores tempos observados" (RODRIGUES et al., 2021, Fase 6). Para ler um percentil
é preciso conhecer a **curva de distribuição** do tempo — e a curva sai de um de dois lugares:

- **Há registro de tempo** (cronoanálise, apontamento de horas, extração de sistema de processos). Então a curva é
  *ajustada aos dados*, pelo procedimento consagrado de modelagem de entradas de simulação (LAW, 2015, cap. 6;
  NIST/SEMATECH, 2012).
- **Não há registro** — a situação mais comum num município. Então a curva é *elicitada* de quem executa, por três
  pontos (mínimo, mais provável, máximo), e convertida em distribuição: beta-PERT, triangular ou log-normal deslocada
  (MALCOLM et al., 1959; VOSE, 2000; RODRIGUES et al., 2021).

Este documento descreve os dois caminhos, as decisões D1 e D3 do curso que os amarram, e o que o script faz em cada um.

## 2. Padrões usuais e fontes

### 2.1 Estudo de tempos clássico (cronoanálise)

É a medição do trabalho consagrada pela engenharia industrial e padronizada nos manuais da Organização Internacional
do Trabalho (OIT, 1992) e em Barnes (1980). O analista divide a tarefa em elementos, cronometra vários ciclos,
**avalia o ritmo** do executante (fator de ritmo, em que 1,00 é o ritmo normal) e soma **tolerâncias** (pausas pessoais,
fadiga, esperas inevitáveis):

```
tempo normal  TN = média dos tempos observados × fator de ritmo
tempo-padrão  TP = TN × (1 + tolerâncias)
```

O número de ciclos não é arbitrário. A forma tabelada nos manuais, para 95% de confiança (k ≈ 2) e erro de ±5%, é

```
N' = [ 40 · √(n·Σx² − (Σx)²) / Σx ]²
```

em que `n` é o número de ciclos da amostra piloto. Algebricamente, `N' = (k · CV / E)²`, com `CV` o coeficiente de
variação da amostra piloto. É a mesma lógica do tamanho de amostra para estimar uma média (seção 4.1).

**Leitura para o setor público.** Cronometrar trabalho intelectual (parecer, análise de processo) é caro e sujeito ao
efeito observador; a cronoanálise convém a tarefas curtas e repetitivas (protocolo, atendimento de balcão, conferência
documental). O fator de ritmo é julgamento do analista e precisa ser declarado. E as tolerâncias **não** podem ser
contadas duas vezes: se a disponibilidade efetiva da Fase 7 já desconta pausas e ausências, o tempo-padrão da Fase 6
usa tolerância zero para esses itens.

### 2.2 Amostragem do trabalho (*work sampling*)

Em vez de cronometrar, o observador registra, em instantes aleatórios, **o que cada pessoa está fazendo**. A proporção
de observações numa categoria estima a fração do tempo dedicada a ela (BARNES, 1980; OIT, 1992). Pelletier e Duffield
(2003) mostram o uso em serviços — enfermagem — e comparam a técnica com o estudo de tempos e movimentos: é mais
barata, menos intrusiva e aplicável a equipes, mas dá **proporções**, não tempos por unidade. O tamanho da amostra vem
da binomial:

```
erro absoluto E (pontos percentuais):  n = z² · p(1 − p) / E²     (OIT: n = 4p(100 − p)/L², com z ≈ 2 e p, L em %)
erro relativo e (fração de p):         n = z² · (1 − p) / (e² · p)
tempo unitário = p × horas-pessoa observadas × ritmo / quantidade produzida × (1 + tolerâncias)
```

No DFT, a amostragem do trabalho serve a duas coisas: **validar** o percentual de alocação declarado (que o L1 classifica
como fonte "média-baixa") e **medir atividades indiretas** (reuniões, tarefas administrativas), que a decisão D2 cataloga
como entrega própria.

### 2.3 Ajuste de distribuições pelo procedimento usual de simulação

Law (2015, cap. 6; 2016) organiza a modelagem de entradas em três atividades, e a prática de software aberto
(DELIGNETTE-MULLER; DUTANG, 2015) e o manual do NIST (2012) seguem o mesmo roteiro:

1. **Hipotetizar famílias** a partir de estatística descritiva e histograma. Média maior que a mediana e assimetria
   positiva indicam cauda à direita — o formato típico de tempo de tarefa (os exemplos reais de tempos de processamento
   e de reparo em Law, 2016, são assimétricos à direita). Coeficiente de variação perto de 1 sugere exponencial; bem abaixo de 1, gama, Weibull ou
   log-normal com forma maior que 1.
2. **Estimar parâmetros** por máxima verossimilhança (MLE). O NIST (2012, §1.3.6.5.2) registra as vantagens
   (consistência, normalidade assintótica, aplicação direta a dados censurados) e o cuidado com amostras pequenas, em
   que os estimadores podem ser enviesados.
3. **Verificar a representatividade**: graficamente (histograma com densidades sobrepostas, gráfico Q-Q e P-P, cujo
   alinhamento à diagonal indica boa aderência — NIST, 2012, §1.3.3.22; WILK; GNANADESIKAN, 1968) e por testes de
   aderência.

**Testes de aderência — o que cada um olha:**

| Teste | Estatística | Ponto forte | Limitação | Fonte |
|---|---|---|---|---|
| Kolmogorov-Smirnov (KS) | maior distância entre a FDA empírica e a ajustada | não depende de classes | mais sensível ao centro que às caudas; com parâmetros estimados da amostra os valores críticos tabelados deixam de valer | NIST §1.3.5.16; LILLIEFORS (1967) |
| Anderson-Darling (AD) | `A² = −n − (1/n)·Σ(2i − 1)[ln F(x₍ᵢ₎) + ln(1 − F(x₍ₙ₊₁₋ᵢ₎))]` | **pesa mais as caudas** — onde está o P80 — e tem mais poder que o KS | valores críticos dependem da família | ANDERSON; DARLING (1954); STEPHENS (1974); NIST §1.3.5.14 |
| Qui-quadrado | `Σ (Oⱼ − Eⱼ)²/Eⱼ` em k classes | vale para qualquer distribuição, inclusive discreta | resultado depende das classes; graus de liberdade `k − 1 − nº de parâmetros` | NIST §1.3.5.15; LAW (2015): classes equiprováveis, `n·pⱼ ≥ 5` |

Law (2016) lembra três limites dos testes: a hipótese nula quase nunca é exatamente verdadeira; com amostras pequenas
ou moderadas o poder é baixo (não rejeitar não prova aderência); com amostras grandes qualquer desvio mínimo leva à
rejeição. Por isso os testes **não escolhem** a distribuição sozinhos — servem de filtro.

**Escolha entre candidatas que passam no filtro: critérios de informação.**

```
AIC = 2k − 2 ln L      (AKAIKE, 1974)          BIC = k ln n − 2 ln L      (SCHWARZ, 1978)
```

em que `k` é o número de parâmetros e `L` a verossimilhança maximizada. Menor é melhor; ambos penalizam parâmetros
extras, o BIC com mais força. Burnham e Anderson (2002) dão a régua usual de leitura das diferenças: modelos com
`ΔAIC ≤ 2` têm suporte substancial e são, na prática, equivalentes.

**Candidatas usuais para tempo de tarefa.** Log-normal, gama e Weibull (suporte positivo, cauda à direita, forma
flexível); exponencial (caso particular, CV = 1, raramente adequado a tarefa humana); log-normal de três parâmetros
(com deslocamento λ, o "tempo mínimo físico"). Trietsch et al. (2012) defendem, com base teórica e de campo, a
log-normal como núcleo do tempo de atividade e documentam duas distorções típicas de dados reais: o **arredondamento**
e o **efeito Parkinson** (o trabalho se expande até o prazo concedido). A normal entra apenas como referência: Law
(2016) a aponta como a escolha errada mais comum para tempos de serviço — é simétrica e atribui probabilidade a tempos
negativos.

**Distribuições de suporte limitado (triangular, beta).** São as curvas naturais da elicitação sem dados (Law, 2016,
recomenda a triangular nesse caso). Com dados, porém, o MLE dos limites é **não regular**: os limites estimados "colam"
nos extremos da amostra e a teoria que sustenta o AIC deixa de valer — o mesmo problema da verossimilhança ilimitada
quando há origem deslocada desconhecida, analisado por Cheng e Amin (1983). E o máximo estimado **trunca a cauda** que
define o P80. Decisão deste módulo: triangular e beta ficam disponíveis sob pedido, fora do ranking padrão.

### 2.4 Dados censurados e arredondados

- **Censura à direita.** Na data de corte, parte dos processos ainda está aberta: sabe-se apenas que o tempo real é
  **maior** que o decorrido. O NIST (2012, §8.1.3.1) classifica esse caso como censura do tipo I. Descartar os abertos,
  ou tratá-los como concluídos, apaga justamente a cauda longa e **subestima** o esforço. O MLE trata a censura somando
  `ln S(t)` (sobrevivência) para os abertos: `ln L = Σ_concluídos ln f(tᵢ) + Σ_abertos ln S(tᵢ)`.
- **Arredondamento.** Quem anota em blocos de 15 minutos produz empates artificiais, que distorcem o KS/AD e podem
  degenerar a log-normal de três parâmetros (TRIETSCH et al., 2012). A solução é tratar cada registro como um
  **intervalo** `[x − passo/2, x + passo/2]` e maximizar `Σ ln[F(sᵢ) − F(iᵢ)]` — o caso "readout/interval data" do NIST
  (2012, §8.1.3.1).

### 2.5 Sem dados: elicitação de três pontos

O PERT nasceu no programa Polaris da Marinha norte-americana, com estimativas de tempo obtidas dos responsáveis
técnicos e expressas em termos probabilísticos (MALCOLM et al., 1959). A média e o desvio clássicos são

```
média = (a + 4m + b)/6        desvio ≈ (b − a)/6
```

e pressupõem uma beta (CLARK, 1962). Os três pontos viram distribuição de três maneiras, todas no script:

| Distribuição | Parâmetros | Suporte | Papel no curso (D3) |
|---|---|---|---|
| **beta-PERT, método de Vose** | `α = (μ − a)(2m − a − b)/[(m − μ)(b − a)]`; `β = α(b − μ)/(μ − a)`; se `m = μ`, `α = 1 + k/2` | `[a, b]` | **padrão**: preserva a moda informada |
| beta-PERT clássica | `α = [(μ − a)/(b − a)]·[(μ − a)(b − μ)/sd² − 1]`, `sd = (b − a)/6` | `[a, b]` | comparação (preserva o desvio) |
| triangular | `a, m, b` diretos; média `(a + m + b)/3` | `[a, b]` | só sensibilidade, nunca número oficial |
| **log-normal deslocada** | `λ, μ, σ` resolvidos (abaixo) | `[λ, ∞)` | **opção avançada**, fiel ao artigo-base |

Fórmulas da beta-PERT: documentação da função `betaPERT` do pacote R `prevalence`, que codifica Malcolm et al. (1959) e
Vose (2000), com `μ = (a + km + b)/(k + 2)` e `k = 4`.

**Log-normal deslocada (parametrização do curso, L2 §3.6 — derivação própria).** O artigo-base critica os "extremos
fixos" do PERT e adota uma "log normal generalizada deslocada" (RODRIGUES et al., 2021), sem dar a parametrização. O
curso lê o mínimo e o máximo como os quantis 5% e 95% e o mais provável como a moda:

```
λ + e^(μ − zσ) = a     λ + e^(μ + zσ) = b     λ + e^(μ − σ²) = m        (z = Φ⁻¹(0,95) = 1,644854)
⇒ μ(λ) = ½ ln[(a − λ)(b − λ)];  σ(λ) = ln[(b − λ)/(a − λ)]/(2z);  resolve-se λ + e^(μ(λ) − σ(λ)²) = m em [0, a)
```

Se a equação não tem raiz em `[0, a)` (estimativa simétrica ou assimétrica à esquerda), o script recusa a log-normal e
recomenda a beta-PERT — o mesmo comportamento do motor do Sistema DFT.

**A qualidade dos três pontos.** A literatura de elicitação adverte que extremos são mal estimados: intervalos ditos
"de 90%" valem cerca de 50% (TEIGEN; JØRGENSEN, 2005); o procedimento de quatro passos (limites realistas, melhor
palpite, confiança declarada) reduz o excesso de confiança (SPEIRS-BRIDGE et al., 2010); aproximações por percentis
são mais acuradas que as fórmulas PERT clássicas (KEEFER; VERDINI, 1993); e o julgamento deve seguir protocolo
estruturado (O'HAGAN, 2019). Isso sustenta ler `a` e `b` como quantis 5%/95% (log-normal) em vez de limites absolutos.

## 3. Passo a passo

**Com dados (caminho a):**

1. **Delimitar a entrega** (fronteira "começa quando / termina quando") e conferir que os registros são da mesma
   entrega. Misturas produzem curvas bimodais que nenhuma família ajusta.
2. **Limpar**: tempos nulos ou negativos, registros abertos (censura), passo de arredondamento.
3. **Descrever**: n, média, mediana, desvio, CV, assimetria, P80 empírico.
4. **Conferir o tamanho de amostra** para a precisão desejada (seção 4.1). Se faltar, seguir com aviso de
   provisoriedade e programar coleta.
5. **Ajustar as candidatas por MLE** (log-normal, gama, Weibull, exponencial, log-normal deslocada; normal como
   referência). Com censura ou arredondamento, usar o MLE adequado (seção 2.4).
6. **Olhar os gráficos** (histograma com densidades, Q-Q, P-P) antes dos números.
7. **Testar a aderência** — AD como teste principal, KS e qui-quadrado como complemento — com p-valores por
   *bootstrap* paramétrico quando os parâmetros vêm da amostra.
8. **Escolher por AIC** entre as não rejeitadas; empate técnico (`ΔAIC ≤ 2`) resolve-se pela família mais simples.
9. **Ler o P80** da escolhida e compará-lo com o P80 empírico e com o P80 das demais plausíveis (faixa de
   sensibilidade).
10. **Registrar** a escolha, a justificativa e a semente do *bootstrap* na memória de cálculo.

**Sem dados (caminho b):**

1. Coletar de quem executa, na ordem **mais provável → mínimo → máximo** (D1; o mínimo primeiro ancora para baixo).
2. Perguntar pelo **valor típico** (moda), não pelo "valor que cobre 2/3".
3. Fazer a **pergunta-filtro** de D3: "existe caso, ainda que raro, que demore mais que o máximo informado?".
   Sim → log-normal deslocada; não → beta-PERT (Vose).
4. Calcular as duas curvas e o P80 de cada uma, **lado a lado**, com o delta percentual.
5. Registrar a distribuição adotada e a justificativa.

## 4. Fórmulas

### 4.1 Tamanho de amostra para estimar a média

```
n = (z · CV / E)²
```

com `E` o erro relativo admitido e `CV = s/x̄`. Com amostra piloto pequena, troca-se `z` pelo `t` de Student com
`n − 1` graus de liberdade e itera-se até estabilizar. Exemplo: CV = 0,5, E = 5%, 95% → `(1,96 × 0,5/0,05)² = 384,1` →
**385** observações. É a mesma regra das eqs. 4–6 de Ichsan, Isvara e Karim (2025) para o número de iterações de Monte
Carlo, lá com `z = 3`.

**Atenção:** essa regra dimensiona a amostra para a **média**. O P80 é mais exigente — seu erro-padrão é
`√(p(1 − p)/n)/f(Q_p)` — e distinguir famílias parecidas exige ainda mais dados, porque o poder dos testes é baixo
em amostras pequenas e moderadas (LAW, 2016). Com menos de 30 registros, o resultado do ranking é provisório.

### 4.2 Erro de Monte Carlo do percentil

```
SE(Q̂_p) ≈ √( p(1 − p)/N ) / f(Q_p)
```

No CT-1, `f(9,075) = 0,107836`, logo, com `N = 10.000`, `SE ≈ 0,0371 h` (≈ 2,2 minutos) — muito menor que a incerteza
dos próprios três pontos (L2 §3.7).

### 4.3 Casos canônicos do curso (reproduzidos pelos testes)

Entrada: parecer técnico de licitação, `a = 4 h`, `m = 6 h`, `b = 16 h`.

| Grandeza | beta-PERT (Vose) — CT-1 | log-normal deslocada — CT-2 |
|---|---|---|
| parâmetros | α = 1,666667; β = 4,333333 | λ = 1,545912; μ = 1,784366; σ = 0,539021 |
| média | 7,333333 h | 8,432930 h |
| P50 | 7,023448 h | 7,501716 h |
| **P80** | **9,074998 h** | **10,920644 h** |
| P90 | 10,223219 h | 13,429399 h |
| P95 | 11,151745 h | 16,000000 h (= máximo, por construção) |

Δ P80 = +20,3%. Com 2.637,19 pareceres/ano (CT-3), a diferença é de cerca de **4.867 h/ano** — o script calcula
`1,845646 × 2.637,1918 = 4.867,3 h`. A beta-PERT clássica dá α = 1,728395; β = 4,493827; P80 = 9,045662 h.

## 5. Critérios de decisão

| Situação | Caminho | Critério |
|---|---|---|
| Há ≥ 30 registros da mesma entrega | ajuste (caminho a) | AD não rejeita (p *bootstrap* ≥ 0,05) **e** menor AIC; `ΔAIC ≤ 2` → família mais simples |
| Há 5 a 29 registros | ajuste com aviso **ou** três pontos | testes com baixo poder; usar o ajuste para calibrar os três pontos, não para substituí-los |
| Não há registro | três pontos (caminho b) | beta-PERT (Vose) por padrão; log-normal deslocada se a pergunta-filtro de D3 for "sim" |
| Há processos abertos na data de corte | MLE com censura | nunca descartar nem tratar como concluídos |
| Tempos anotados em blocos (15 min, meia hora) | MLE intervalar | não confiar em KS/AD sobre dados com muitos empates |
| Todas as plausíveis rejeitadas pelo AD | rever a fronteira da entrega | mistura de tipos; subdividir (Fase 3) ou usar a distribuição empírica |
| Três pontos simétricos ou com cauda à esquerda | beta-PERT | a log-normal deslocada não tem solução com λ ≥ 0 |
| Escolher entre AIC e BIC | AIC como principal | BIC como conferência; se divergirem, preferir a família mais simples e registrar |

**Qual teste de aderência?** AD como principal, porque o P80 está na cauda e o AD pesa as caudas (STEPHENS, 1974; NIST,
2012). KS como complemento intuitivo (a "maior distância" é fácil de explicar a gestor). Qui-quadrado só com amostras
maiores e classes equiprováveis. Com parâmetros estimados da própria amostra — o caso normal — os p-valores tabelados
são otimistas; o script oferece *bootstrap* paramétrico.

**P80 ajustado ou empírico?** Com muitos dados da mesma entrega, o P80 empírico é a leitura literal do artigo-base e é
válido. O ajustado suaviza a amostra, permite extrapolar e entra na simulação conjunta da carteira (L2 §3.8). Reportar
os dois; adotar o ajustado quando a aderência for boa.

## 6. Armadilhas

1. **Substituir a distribuição pela média.** É a "armadilha 1" de Law (2016): a variabilidade, e não só a média,
   determina filas e atrasos. No DFT, dimensionar pela média aceita que metade das entregas estoure o orçamento de
   horas.
2. **Usar a normal para tempo de tarefa** ("armadilha 2" de Law, 2016). Ela subestima a cauda e admite tempos
   negativos.
3. **Ler o p-valor do KS como se os parâmetros fossem conhecidos.** Com parâmetros estimados, o teste fica
   conservador demais e quase nunca rejeita (LILLIEFORS, 1967; NIST, 2012).
4. **Tomar "não rejeitado" por "verdadeiro".** Com 30 registros quase nada é rejeitado; com 3.000, quase tudo é.
5. **Escolher a distribuição depois de ver qual P80 agrada.** O critério (AIC + AD) e o percentil são fixados
   **antes**, como manda o L1 sobre o percentil.
6. **Descartar processos em aberto.** No exemplo fictício, o P80 da log-normal cai de 16,33 h para 13,17 h (−19%) se
   os 24 abertos forem ignorados.
7. **Ignorar o arredondamento.** Empates artificiais derrubam o KS/AD e podem fazer a log-normal de três parâmetros
   "explodir" (λ colado no mínimo).
8. **Ajustar triangular ou beta com limites livres.** Os limites colam nos extremos da amostra e o AIC fica
   artificialmente bom; o máximo estimado corta a cauda.
9. **Misturar entregas diferentes na mesma amostra.** A curva fica bimodal. Divergência grande é sinal de fronteira
   mal definida (L1, Fase 6).
10. **Contar tolerâncias duas vezes** (cronoanálise e disponibilidade efetiva).
11. **Tratar o "mais provável" como quantil 2/3.** Ver D1 e a seção 7.
12. **Monte Carlo sobre três pontos inventados.** O método não melhora dado ruim; o protocolo de elicitação importa
    mais que o número de iterações.

## 7. Como a etapa entra no DFT

- **Fase 6 (Aula 3, Bloco B).** É o núcleo do "esforço unitário com tratamento de variabilidade". O aluno preenche, no
  **Produto 3**, para cada entrega do setor-recorte: a fonte do tempo (registro, amostragem, três pontos), a curva
  adotada, a justificativa e o P80.
- **Fase 4 → Fase 6.** O P80 multiplica a demanda projetada; a diferença entre curvas vira horas e servidores (no
  CT-1 × CT-2, ~4.867 h/ano, ~4,2 servidores).
- **Fase 7.** A cronoanálise e a amostragem do trabalho alimentam e validam a disponibilidade efetiva (pausas,
  atividades indiretas) — sem dupla contagem.
- **Fase 8.** A memória de cálculo registra a distribuição adotada (D3), a semente, os testes e a faixa de
  sensibilidade.
- **Do primeiro ao segundo ciclo.** No primeiro ciclo o município trabalha quase só com três pontos; o L1 recomenda
  apontamento de horas nas entregas de maior carga a partir do segundo. O caminho (a) é a ponte: os primeiros registros
  servem para **calibrar** os três pontos (o máximo informado foi ultrapassado? a moda bate?).

**Decisão D1 e uma observação técnica.** O curso trata o ponto central como **moda**. A função
`moda_implicita_quantil` mostra a consequência numérica da ambiguidade: com `(4; ?; 16)`, se o informante entendeu
"valor que cobre 2/3" e respondeu 8 h, ler 8 como moda dá P80 = 10,65 h, enquanto a moda coerente com esse quantil é
5,96 h e o P80 correspondente é 9,04 h. Em tempos assimétricos à direita, portanto, **ler como moda um valor pensado
como quantil superestima** o P80; e **ler como quantil um valor típico subestima** (no CT-1, 6 h nem é atingível como
quantil 2/3 com esses extremos — a moda teria de ficar abaixo do mínimo). O sentido do erro depende de qual lado da
ambiguidade ocorre; a regra prática de D1 — perguntar pelo valor típico e tratá-lo como moda — elimina as duas
versões.

## 8. Exemplo numérico (fictício)

Noventa tempos de "parecer técnico de licitação" (`dados/distribuicoes_tempos_parecer.csv`): média 7,38 h, mediana
6,60 h, CV 0,44, assimetria 0,79, P80 empírico 10,20 h. Para estimar a média com ±5% seriam precisos 299 registros
(±10%: 77 — a amostra basta). Ranking por AIC: log-normal 451,3; log-normal deslocada 452,0 (ΔAIC 0,66); gama 453,7;
Weibull 461,1; normal 470,0; exponencial 541,8. O AD com *bootstrap* (199 réplicas) não rejeita a log-normal
(p = 0,19) e rejeita Weibull, normal e exponencial. Recomendação: **log-normal** (empate técnico com a deslocada,
vence pela parcimônia), P80 = 9,68 h; entre as plausíveis o P80 vai de 9,65 a 9,68 h.

Licença ambiental com 24 de 60 processos abertos (`distribuicoes_censurados.csv`): P80 com censura 16,33 h × 13,17 h
descartando os abertos. Ouvidoria anotada em blocos de 15 min (`distribuicoes_arredondados.csv`): MLE intervalar,
gama, Weibull e log-normal com AIC 265,9 / 266,5 / 267,0 e P80 entre 1,05 e 1,11 h. Amostragem do trabalho (400
observações): análise de processos 46,2% (IC 95% 41,4%–51,1%); com 960 horas-pessoa e 95 processos concluídos,
4,67 h/processo.

## 9. O que o script faz

| Função / modo | O que faz | Referência |
|---|---|---|
| `estatisticas_descritivas` | n, média, mediana, desvio, CV, assimetria, curtose, P80 empírico | LAW (2015), Atividade I |
| `tamanho_amostra_media` | `n = (z·CV/E)²`, com correção t iterada para amostra piloto | BARNES (1980); OIT (1992); ICHSAN et al. (2025) |
| `ciclos_cronoanalise_oit`, `tempo_padrao` | fórmula tabelada de ciclos (fator 40); TN e TP com ritmo e tolerâncias | OIT (1992); BARNES (1980) |
| `amostragem_trabalho_n`, `..._proporcoes`, `tempo_unitario_amostragem` | n binomial; proporções com IC de Wilson; tempo unitário | BARNES (1980); OIT (1992) |
| `ajustar_candidatas`, `ajustar_candidata` | MLE de exponencial, gama, log-normal, Weibull, log-normal deslocada (verossimilhança perfilada), normal; triangular e beta sob pedido; AIC, BIC, KS, AD, qui-quadrado equiprovável, r(Q-Q), P80; p-valores por *bootstrap* | LAW (2015); NIST (2012); AKAIKE (1974); SCHWARZ (1978); STEPHENS (1974) |
| `recomendar` | ΔAIC ≤ 2 → plausíveis; filtro AD; parcimônia; texto de justificativa | BURNHAM; ANDERSON (2002) |
| `ajustar_censurado`, `ajustar_intervalar` | MLE com censura à direita e com dados arredondados/intervalares | NIST (2012, §8.1.3.1); TRIETSCH et al. (2012) |
| `beta_pert`, `lognormal_deslocada_tres_pontos`, `triangular`, `pert_classico`, `resumo_tres_pontos` | três pontos → parâmetros, média, desvio, P50/P80/P90/P95; delta D3 | MALCOLM et al. (1959); VOSE (2000); L2 §3.5–3.6 |
| `moda_implicita_quantil` | efeito numérico da ambiguidade moda × quantil 2/3 (D1) | RODRIGUES et al. (2021); L2 §3.1 |
| `erro_padrao_quantil` | erro de Monte Carlo do percentil | L2 §3.7 |
| `grafico_ajuste`, `grafico_tres_pontos` | histograma + densidades + Q-Q + P-P; curvas D3 com P80 | — |
| `cli` | `dft distribuicoes arquivo.csv` (tempos), `--censura`, `--arredondamento`, `--amostragem-trabalho`, CSV de três pontos, `--tres-pontos 4 6 16`, `--demanda`, `--bootstrap`, `--grafico` | — |

Os testes (`tests/test_distribuicoes.py`) reproduzem CT-1 e CT-2 com tolerância de `1e-6` (identidades da log-normal
com `1e-9`), o erro de Monte Carlo 0,0371 h, as fórmulas de amostra da OIT e a recuperação de parâmetros em amostras
sintéticas (inclusive com censura e arredondamento).

## Referências

AKAIKE, Hirotugu. A new look at the statistical model identification. **IEEE Transactions on Automatic Control**,
v. 19, n. 6, p. 716-723, 1974. DOI: 10.1109/TAC.1974.1100705. Disponível em: https://doi.org/10.1109/TAC.1974.1100705.
Acesso em: 5 nov. 2026.

ANDERSON, T. W.; DARLING, D. A. A test of goodness of fit. **Journal of the American Statistical Association**, v. 49,
n. 268, p. 765-769, 1954. DOI: 10.1080/01621459.1954.10501232. Disponível em:
https://doi.org/10.1080/01621459.1954.10501232. Acesso em: 5 nov. 2026.

BARNES, Ralph M. **Motion and time study**: design and measurement of work. 7. ed. New York: John Wiley & Sons, 1980.

BURNHAM, Kenneth P.; ANDERSON, David R. **Model selection and multimodel inference**: a practical information-theoretic
approach. 2. ed. New York: Springer, 2002. DOI: 10.1007/b97636. Disponível em: https://doi.org/10.1007/b97636. Acesso
em: 5 nov. 2026.

CHENG, R. C. H.; AMIN, N. A. K. Estimating parameters in continuous univariate distributions with a shifted origin.
**Journal of the Royal Statistical Society: Series B (Methodological)**, v. 45, n. 3, p. 394-403, 1983. DOI:
10.1111/j.2517-6161.1983.tb01268.x. Disponível em: https://doi.org/10.1111/j.2517-6161.1983.tb01268.x. Acesso em:
5 nov. 2026.

CLARK, Charles E. The PERT model for the distribution of an activity time. **Operations Research**, v. 10, n. 3,
p. 405-406, 1962. DOI: 10.1287/opre.10.3.405. Disponível em: https://doi.org/10.1287/opre.10.3.405. Acesso em:
5 nov. 2026.

DELIGNETTE-MULLER, Marie Laure; DUTANG, Christophe. fitdistrplus: an R package for fitting distributions. **Journal of
Statistical Software**, v. 64, n. 4, p. 1-34, 2015. DOI: 10.18637/jss.v064.i04. Disponível em:
https://doi.org/10.18637/jss.v064.i04. Acesso em: 5 nov. 2026.

ICHSAN, Mohammad; ISVARA, Wisnu; KARIM, Syaeful. Monte Carlo simulation for enhancing the schedule completion forecast
of Jakarta Central Railway Station construction project. **Applied Sciences**, v. 15, n. 13, 7464, 2025. DOI:
10.3390/app15137464. Disponível em: https://doi.org/10.3390/app15137464. Acesso em: 5 nov. 2026.

KEEFER, Donald L.; VERDINI, William A. Better estimation of PERT activity time parameters. **Management Science**,
v. 39, n. 9, p. 1086-1091, 1993. DOI: 10.1287/mnsc.39.9.1086. Disponível em: https://doi.org/10.1287/mnsc.39.9.1086.
Acesso em: 5 nov. 2026.

LAW, Averill M. **Simulation modeling and analysis**. 5. ed. New York: McGraw-Hill, 2015.

LAW, Averill M. A tutorial on how to select simulation input probability distributions. In: WINTER SIMULATION
CONFERENCE, 2016, Washington, DC. **Proceedings** [...]. Piscataway: IEEE, 2016. p. 103-117. DOI:
10.1109/WSC.2016.7822083. Disponível em: https://doi.org/10.1109/WSC.2016.7822083. Acesso em: 5 nov. 2026.

LILLIEFORS, Hubert W. On the Kolmogorov-Smirnov test for normality with mean and variance unknown. **Journal of the
American Statistical Association**, v. 62, n. 318, p. 399-402, 1967. DOI: 10.1080/01621459.1967.10482916. Disponível
em: https://doi.org/10.1080/01621459.1967.10482916. Acesso em: 5 nov. 2026.

MALCOLM, D. G.; ROSEBOOM, J. H.; CLARK, C. E.; FAZAR, W. Application of a technique for research and development
program evaluation. **Operations Research**, v. 7, n. 5, p. 646-669, 1959. DOI: 10.1287/opre.7.5.646. Disponível em:
https://doi.org/10.1287/opre.7.5.646. Acesso em: 5 nov. 2026.

NIST/SEMATECH. **e-Handbook of statistical methods**. Gaithersburg: National Institute of Standards and Technology,
2012. Seções 1.3.3.22 (Probability plot), 1.3.5.14 (Anderson-Darling test), 1.3.5.15 (Chi-square goodness-of-fit
test), 1.3.5.16 (Kolmogorov-Smirnov goodness-of-fit test), 1.3.6.5.2 (Maximum likelihood) e 8.1.3.1 (Censoring).
Disponível em: https://www.itl.nist.gov/div898/handbook/. Acesso em: 5 nov. 2026.

O'HAGAN, Anthony. Expert knowledge elicitation: subjective but scientific. **The American Statistician**, v. 73,
supl. 1, p. 69-81, 2019. DOI: 10.1080/00031305.2018.1518265. Disponível em: https://doi.org/10.1080/00031305.2018.1518265.
Acesso em: 5 nov. 2026.

ORGANIZAÇÃO INTERNACIONAL DO TRABALHO (OIT). **Introduction to work study**. Edited by George Kanawaty. 4. ed. rev.
Geneva: International Labour Office, 1992. ISBN 92-2-107108-1.

PELLETIER, Dianne; DUFFIELD, Christine. Work sampling: valuable methodology to define nursing practice patterns.
**Nursing & Health Sciences**, v. 5, n. 1, p. 31-38, 2003. DOI: 10.1046/j.1442-2018.2003.00132.x. Disponível em:
https://doi.org/10.1046/j.1442-2018.2003.00132.x. Acesso em: 5 nov. 2026.

RODRIGUES, Adriana Maciel; CAVALCANTI, Marcos do Couto Bezerra; ARAÚJO DA SILVA, Raul; MACEDO, Valéria. Proposta de
metodologia de dimensionamento da força de trabalho para o TCE-RJ. **Revista do Tribunal de Contas do Estado do Rio de
Janeiro**, Rio de Janeiro, v. 2, n. 1, p. 72-83, 2021. DOI: 10.70690/xt46c612. Disponível em:
https://doi.org/10.70690/xt46c612. Acesso em: 5 nov. 2026.

SCHWARZ, Gideon. Estimating the dimension of a model. **The Annals of Statistics**, v. 6, n. 2, p. 461-464, 1978. DOI:
10.1214/aos/1176344136. Disponível em: https://doi.org/10.1214/aos/1176344136. Acesso em: 5 nov. 2026.

SPEIRS-BRIDGE, Andrew; FIDLER, Fiona; McBRIDE, Marissa; FLANDER, Louisa; CUMMING, Geoff; BURGMAN, Mark. Reducing
overconfidence in the interval judgments of experts. **Risk Analysis**, v. 30, n. 3, p. 512-523, 2010. DOI:
10.1111/j.1539-6924.2009.01337.x. Disponível em: https://doi.org/10.1111/j.1539-6924.2009.01337.x. Acesso em:
5 nov. 2026.

STEPHENS, M. A. EDF statistics for goodness of fit and some comparisons. **Journal of the American Statistical
Association**, v. 69, n. 347, p. 730-737, 1974. DOI: 10.1080/01621459.1974.10480196. Disponível em:
https://doi.org/10.1080/01621459.1974.10480196. Acesso em: 5 nov. 2026.

TEIGEN, Karl Halvor; JØRGENSEN, Magne. When 90% confidence intervals are 50% certain: on the credibility of credible
intervals. **Applied Cognitive Psychology**, v. 19, n. 4, p. 455-475, 2005. DOI: 10.1002/acp.1085. Disponível em:
https://doi.org/10.1002/acp.1085. Acesso em: 5 nov. 2026.

TRIETSCH, Dan; MAZMANYAN, Lilit; GEVORGYAN, Lilit; BAKER, Kenneth R. Modeling activity times by the Parkinson
distribution with a lognormal core: theory and validation. **European Journal of Operational Research**, v. 216, n. 2,
p. 386-396, 2012. DOI: 10.1016/j.ejor.2011.07.054. Disponível em: https://doi.org/10.1016/j.ejor.2011.07.054. Acesso
em: 5 nov. 2026.

VIRTANEN, Pauli et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. **Nature Methods**, v. 17,
n. 3, p. 261-272, 2020. DOI: 10.1038/s41592-019-0686-2. Disponível em: https://doi.org/10.1038/s41592-019-0686-2. Acesso
em: 5 nov. 2026.

VOSE, David. **Risk analysis**: a quantitative guide. 2. ed. Chichester: John Wiley & Sons, 2000. (Método "vose" de
parametrização da beta-PERT, conforme a documentação da função `betaPERT` do pacote R `prevalence`, disponível em:
https://search.r-project.org/CRAN/refmans/prevalence/html/betaPERT.html. Acesso em: 5 nov. 2026.)

WILK, M. B.; GNANADESIKAN, R. Probability plotting methods for the analysis of data. **Biometrika**, v. 55, n. 1,
p. 1-17, 1968. DOI: 10.1093/biomet/55.1.1. Disponível em: https://doi.org/10.1093/biomet/55.1.1. Acesso em: 5 nov. 2026.
