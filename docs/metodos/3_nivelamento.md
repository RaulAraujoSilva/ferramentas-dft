# 3. Nivelamento de recursos e Teoria das Restrições

**Módulo:** `dft_ferramentas/nivelamento.py` · **Exemplo:** `exemplos/3_nivelamento.py` · **Dados fictícios:**
`dados/nivelamento_atividades.csv`, `dados/nivelamento_capacidades.csv` e `dados/nivelamento_fluxo.csv` ·
**Testes:** `tests/test_nivelamento.py`

> Dados e exemplos são fictícios. Nenhum município real é nomeado e nenhuma decisão do TCE-RJ é citada.

## 1. Para que serve esta ferramenta no DFT

A fórmula do quadro, Q = Σ Dᵢ·Eᵢ / H, devolve a quantidade **média** de pessoas que a carga exige no período. Ela
parte de duas premissas que quase nunca são verificadas:

1. **A carga pode ser distribuída no tempo.** Se as entregas se concentram em poucas semanas (campanha de alvarás,
   recadastramento, fechamento de exercício), a média esconde picos. Dimensionar pela média deixa o pico descoberto.
   Dimensionar pelo pico paga ociosidade no resto do ano.
2. **Toda hora de trabalho vale o mesmo.** Num fluxo em etapas, a vazão do serviço é limitada pela etapa mais fraca.
   Uma hora a mais numa etapa folgada não muda nada para o cidadão. A mesma hora na etapa que trava muda tudo.

O módulo trata essas duas premissas com dois instrumentos da gestão de operações:

- **Parte A — suavização e nivelamento de recursos.** É a otimização de recursos do PMBOK, que mostra quanto do
  quadro "pelo pico" é efeito do calendário e não da carga.
- **Parte B — Teoria das Restrições (TOC).** São os cinco passos de focalização de Goldratt. Eles mostram onde cada
  servidor a mais (ou realocado) produz resultado.

O artigo-base não trata nenhum dos dois (RODRIGUES et al., 2021): calcula o quadro por carga agregada. Este módulo
**não substitui** a fórmula. Ele a testa contra o calendário e contra o fluxo antes da Fase 8.

## 2. Padrões usuais e fontes

| Padrão | O que estabelece | Fonte principal |
|---|---|---|
| Otimização de recursos (PMBOK) | **Nivelamento**: ajusta datas à disponibilidade de recursos e pode alongar o prazo e mudar o caminho crítico. **Suavização**: usa só as folgas, sem mexer no prazo nem no caminho crítico | PMI (2017), §6.5.2.3 |
| Histograma de recursos | Demanda por recurso em cada período, dado o cronograma | PMI (2017); Harris (1990) |
| Mínima soma dos quadrados (Burgess) | Com a área fixa, Σr(t)² é mínima quando o perfil é plano. Desloca as atividades do fim para o início, dentro da folga | Burgess e Killebrew (1962); Neumann e Zimmermann (2000) |
| Regras de prioridade e esquema serial | Programa as atividades na ordem da regra, no 1.º período em que cabem na capacidade. A menor folga (MINSLK) está entre as melhores regras | Davis e Patterson (1975); Kolisch e Hartmann (2006) |
| Heijunka (nivelamento da produção) | Nivelar volume e mix para eliminar a irregularidade (*mura*) e a sobrecarga (*muri*) | Ohno (1988); Liker (2004) |
| Suavização da demanda em serviços | A variabilidade "artificial" criada pela própria agenda é gerenciável e gera picos evitáveis | McManus et al. (2003); Litvak e Fineberg (2013) |
| Polivalência e balanceamento | Flexibilidade parcial e encadeada captura quase todo o ganho da flexibilidade total | Jordan e Graves (1995); Hopp, Tekin e Van Oyen (2004) |
| Teoria das Restrições — 5 passos | Identificar, explorar, subordinar, elevar a restrição e voltar ao início sem deixar a inércia virar restrição | Goldratt (1990); Goldratt e Cox (2004); Rahman (1998) |
| Tambor-pulmão-corda e gerenciamento do pulmão | A restrição dita o ritmo (tambor), um pulmão de tempo a protege e a liberação de trabalho segue o tambor (corda). Zonas do pulmão em terços | Cox e Schleier (2010); Watson, Blackstone e Gardiner (2007) |
| Contabilidade de ganhos | T (ganho), I (inventário/investimento) e OE (despesa operacional). Uma decisão é boa se ΔT > ΔOE | Corbett (1998); Goldratt e Cox (2004) |
| Corrente crítica | Estimativas agressivas, margens agregadas num pulmão de projeto, dependências de recurso | Goldratt (1997); Newbold (1998); Leach (1999); Herroelen e Leus (2001) |
| TOC em saúde e serviços públicos | Ganhos de vazão e de tempo de espera, com evidência ainda concentrada em relatos de caso | Lubitsh, Doyle e Valentine (2005); Bacelar-Silva, Cox e Rodrigues (2022); Datt et al. (2024) |

O que a evidência permite afirmar sobre a TOC:

- **Relatos de caso favoráveis.** A meta-análise de Mabin e Balderstone (2003) reuniu mais de 80 aplicações com
  melhoras operacionais e financeiras significativas e não encontrou relato de fracasso. A ausência de fracassos é,
  em si, sinal de viés de publicação.
- **Saúde.** A revisão sistemática de Bacelar-Silva, Cox e Rodrigues (2022) analisou 42 implementações, todas com
  resultado positivo. As médias relatadas foram 50 % de redução do tempo de espera e 34 % de aumento da vazão.
- **Desenho mais robusto.** O estudo de série temporal interrompida em um hospital público britânico (LUBITSH;
  DOYLE; VALENTINE, 2005) encontrou impacto em dois de três departamentos. É resultado mais modesto e mais crível.
- **Cautela com o lean no setor público.** Radnor e Osborne (2013) advertem contra transplantar ferramentas de
  manufatura para serviços públicos sem a lógica do serviço.

## 3. Conceitos e fórmulas

### 3.1 Histograma de recursos

Para um cronograma com inícios sᵢ, a demanda do recurso *k* no período *t* é

```
r_k(t) = Σ_{i: recurso(i)=k, sᵢ ≤ t < sᵢ + dᵢ} cargaᵢ        (servidores no período t)
```

A **área** Σ_t r_k(t) é o esforço total em servidor-semanas. Ela não muda quando as atividades são deslocadas. O
**pico** e a **soma dos quadrados** mudam. Indicadores usados:

- pico, média = área ÷ horizonte e razão pico/média;
- Σr²;
- semanas acima da capacidade.

### 3.2 Suavização × nivelamento

| | Suavização (*resource smoothing*) | Nivelamento (*resource leveling*) |
|---|---|---|
| O que fica fixo | Prazo do projeto e janelas | Capacidade de cada recurso |
| O que pode mudar | Datas, só dentro da folga | Datas e o prazo final |
| Pergunta que responde | "Qual o menor pico possível sem atrasar?" | "Quanto atrasa se eu só tenho *c* pessoas?" |
| Objetivo usual | Minimizar Σr² ou o pico | Respeitar r_k(t) ≤ c_k em todo t |

### 3.3 Método de Burgess (mínima soma dos quadrados)

O ponto de partida é o cronograma "mais cedo" do CPM (ESᵢ, LSᵢ, folga = LSᵢ − ESᵢ). O procedimento:

1. Percorrer as atividades da última para a primeira (ordem topológica reversa).
2. Para cada atividade, calcular a janela viável. O início mínimo é o término das predecessoras já posicionadas e a
   liberação. O início máximo é o início das sucessoras, o prazo e a semana-limite, menos a duração.
3. Testar cada início *t* da janela e escolher o que minimiza Σ_t r_k(t)². Em caso de empate, ficar com o menor pico
   e, depois, com o início mais cedo. Mover só se a melhora for estrita.
4. Repetir as passadas até que nenhuma atividade se mova.

A soma dos quadrados é o critério clássico porque, com a área fixa, Σr² = T·(média² + variância). Minimizar Σr²
equivale a minimizar a variância do perfil. É uma heurística: o problema de nivelamento é NP-difícil e o ótimo
global não é garantido (NEUMANN; ZIMMERMANN, 2000).

### 3.4 Nivelamento por esquema serial e regra de prioridade

O esquema serial de geração de cronograma (KOLISCH; HARTMANN, 2006) funciona assim:

1. Entre as atividades **elegíveis** (todas as predecessoras já programadas), escolher a de maior prioridade.
2. Posicioná-la no primeiro *t* ≥ max(liberação, término das predecessoras) em que r_k(τ) + carga ≤ c_k para todo
   τ ∈ [t, t + d).
3. Repetir até programar todas.

O módulo oferece três regras: menor folga total (MINSLK, padrão), menor término tardio (LFT) e menor início tardio
(LST). A regra de menor folga está entre as de melhor desempenho no estudo comparativo de Davis e Patterson (1975).

### 3.5 Heijunka e nivelamento da demanda em serviços

No Sistema Toyota, *heijunka* é nivelar volume e mix da produção para que o ritmo de trabalho seja estável (OHNO,
1988). Liker (2004) o coloca como o 4.º princípio e associa a ele dois males além do desperdício (*muda*): a
irregularidade (*mura*) e a sobrecarga (*muri*).

Em serviços públicos, a maior parte da demanda não é controlável. Uma parte relevante, porém, é **gerada pela
própria administração**:

- prazos legais concentrados no mesmo mês;
- campanhas;
- agendas de vistoria;
- mutirões;
- calendário de renovação de licenças.

Na saúde, a variação da agenda cirúrgica programada pesa sobre o acesso à terapia intensiva (MCMANUS et al., 2003).
Suavizar a agenda de eletivas é, segundo Litvak e Fineberg (2013), alavanca simultânea de qualidade, segurança e
economia. A tradução municipal: **antes de pedir gente para o pico, verifique se o pico foi criado pelo calendário
da própria prefeitura** (escalonar vencimentos por dígito final da inscrição, por exemplo).

### 3.6 Balanceamento entre equipes e polivalência

Balancear é aproximar as utilizações das etapas ou equipes. Realocar quem tem folga para onde há fila é a resposta
de custo zero, mas exige competência (Fase 5) e tem limite. Jordan e Graves (1995) mostram que flexibilidade
parcial, organizada em **cadeia**, obtém quase todo o benefício da flexibilidade total. Hopp, Tekin e Van Oyen
(2004) comparam duas formas de polivalência:

- *cherry picking*: treinar só para socorrer o gargalo;
- *skill chaining*: cada pessoa domina a sua etapa e a vizinha.

O encadeamento sai mais robusto. Hopp e Van Oyen (2004) dão uma estrutura para decidir quanto treinar e como
coordenar. O módulo testa a realocação só **dentro do mesmo grupo de competência** e mostra o
efeito colateral: tirar gente de uma etapa pode criar nova restrição.

### 3.7 Teoria das Restrições — os cinco passos

A capacidade de cada etapa *i*, em casos do sistema por semana, e a sua utilização são:

```
capacidadeᵢ = nᵢ · Hᵢ / (tᵢ · fᵢ)
ρᵢ = D · fᵢ · tᵢ / (nᵢ · Hᵢ)
```

onde nᵢ é o número de servidores, Hᵢ as horas efetivas por servidor na semana, tᵢ as horas por caso, fᵢ a fração
dos casos que passam pela etapa e D a demanda semanal. A vazão do sistema em série é

```
X = min( D , minᵢ capacidadeᵢ )
```

1. **Identificar** a restrição: a etapa de menor capacidade, que é a de maior ρ. Se D for menor que todas as
   capacidades, a restrição é externa (a própria demanda).
2. **Explorar**: tirar da restrição todo trabalho que não precisa ser feito ali. Exemplos: retrabalho por processo
   incompleto, conferências administrativas, esperas por documento. O "kit completo" de Ronen (1992), que consiste em
   não iniciar a análise sem o processo instruído, é a forma clássica em serviços, também usada em saúde (RONEN; PLISKIN; PASS, 2006). Custo de pessoal: zero.
3. **Subordinar**: as demais etapas trabalham no ritmo da restrição. Ociosidade numa não-restrição é **capacidade
   protetiva**, não desperdício. Metas de produção por etapa que estimulem "empurrar" trabalho são revistas.
4. **Elevar**: aumentar a capacidade da restrição. As opções vão de horas adicionais e realocação a apoio em tarefas
   não técnicas, terceirização de parte da atividade e, por fim, provimento. São as quatro respostas da ISO
   30409:2016: realocação, requalificação, terceirização e recrutamento.
5. **Voltar ao passo 1** sem deixar que a inércia, isto é, as políticas desenhadas para a restrição antiga, vire a
   nova restrição (GOLDRATT, 1990).

### 3.8 Tambor-pulmão-corda e tamanho do pulmão

- **Tambor:** a restrição, que dita o ritmo X.
- **Pulmão:** tempo (ou fila de casos prontos) antes da restrição, para que ela nunca fique parada por atraso a
  montante.
- **Corda:** novos casos são liberados ao fluxo no ritmo do tambor.

O módulo dimensiona o pulmão de duas formas e converte o tempo em casos pela lei de Little, L = λ·W (LITTLE, 1961):

```
LT_montante = Σ atravessamentoᵢ (etapas antes da restrição, em dias úteis)
Regra prática:  B = fator · LT_montante           (fator declarado; 0,5 por padrão)
Estatístico:    B = z · σ_LT,  σ_LT = √Σ (cvᵢ · atravessamentoᵢ)²
Casos no pulmão = (X / dias úteis por semana) · B
```

O pulmão inicial é calibrado pelo **gerenciamento do pulmão**, com o pulmão dividido em terços (COX; SCHLEIER,
2010):

- **verde:** fila confortável;
- **amarelo:** acompanhar;
- **vermelho:** a restrição corre risco de parar, e é hora de acionar a recuperação.

Se a zona vermelha é frequente, o pulmão aumenta. Se a fila nunca sai do verde, ele diminui. O gerenciamento por
zonas já foi aplicado a serviços hospitalares (UMBLE; UMBLE, 2006).

A corda não resolve o déficit. Ela **transfere** o acúmulo do interior do fluxo para uma fila de entrada visível,
triada e priorizada, e torna previsível o prazo do que já está em andamento. Só elevar a restrição fecha a diferença
D − X.

### 3.9 Contabilidade de ganhos

Medidas de Goldratt e Cox (2004) e Corbett (1998):

- **T (ganho)** = receita − custos totalmente variáveis, por unidade de tempo;
- **I** = inventário/investimento;
- **OE** = despesa operacional (inclui a folha);
- lucro = T − OE; ROI = (T − OE)/I.

A regra de decisão é avaliar ΔT, ΔI e ΔOE. +1 servidor numa não-restrição dá ΔT = 0 e ΔOE > 0, sempre ruim. Na
restrição, depende de quanto a vazão sobe.

No setor público, "T" raramente é receita. Pode ser:

- a taxa líquida arrecadada;
- o custo evitado (multas, judicialização);
- um valor público convencionado e **declarado na Fase 1**.

A lógica de ΔT × ΔOE continua válida mesmo com T em "casos concluídos".

### 3.10 Corrente crítica e pulmões de projeto

Goldratt (1997) propôs três mudanças no planejamento de projetos:

- estimativas agressivas, sem margem individual;
- a margem retirada agregada num **pulmão de projeto** ao fim da corrente, mais pulmões de convergência onde
  caminhos alimentadores entram na corrente;
- a corrente crítica, que é o caminho mais longo considerando também as dependências de recurso.

Há dois dimensionamentos usuais do pulmão:

```
Corte de 50 %:  pulmão = 0,5 · Σ agressivas         (Goldratt; Leach, 1999)
RSS:            pulmão = √Σ (seguraᵢ − agressivaᵢ)²  (Newbold, 1998)
```

Herroelen e Leus (2001) mostram que a regra de 50 % pode superestimar seriamente a proteção em correntes longas e
que a atualização periódica do cronograma-base é o que mais ajuda. O consumo do pulmão também é acompanhado em
terços (LEACH, 1999).

## 4. Passo a passo

**Parte A — calendário de entregas**

1. Liste as entregas que têm calendário: campanhas, operações anuais, projetos. Dê a cada uma duração em semanas,
   predecessoras, perfil que executa, carga em servidores por semana e janela (semana mais cedo e semana-limite).
2. Rode o CPM: prazo, caminho crítico, folgas. Folga negativa significa janela inviável e precisa ser resolvida antes
   de tudo.
3. Gere o histograma "mais cedo". Anote, por perfil, o pico, a média e a razão pico/média.
4. **Suavize** (Burgess). O pico que sobra é o mínimo que o calendário impõe sem atrasar.
5. Se o pico suavizado ainda supera o quadro disponível, **nivele** com a capacidade real e leia o atraso
   resultante. A decisão passa a ser um *trade-off* explícito: aceitar o atraso, reforçar temporariamente o perfil ou
   mudar o calendário (heijunka).
6. Leve à Fase 8 a média (quadro permanente), o pico suavizado (necessidade temporária) e o prazo nivelado.

**Parte B — fluxo de atendimento**

1. Monte a tabela de etapas: servidores, horas por caso (Fase 6), fração de roteamento, horas efetivas (Fase 7),
   grupo de competência (Fase 5) e, se houver, tempo de atravessamento e sua variabilidade, extraídos do sistema de
   protocolo.
2. Passo 1: calcule capacidade e utilização, identifique a restrição e a vazão.
3. Passo 2: liste as perdas de capacidade da restrição (retrabalho, kit incompleto, tarefas administrativas) e
   simule a sua retirada ou transferência.
4. Passo 3: dimensione o pulmão e defina a corda (regra de liberação/triagem na entrada).
5. Passo 4: compare +1 servidor em cada etapa e a realocação de polivalentes. Calcule a elevação **mínima**
   necessária.
6. Passo 5: recalcule e identifique a nova restrição. Dimensione cada etapa para a vazão-alvo com utilização alta na
   restrição e capacidade protetiva nas demais.

## 5. Critérios de decisão

| Situação | Decisão |
|---|---|
| Pico/média ≤ 1,2 depois da suavização | Quadro pela média (fórmula do curso) é exequível |
| Pico/média alto mesmo suavizado, com picos curtos | Quadro pela média + reforço temporário (força-tarefa, horas adicionais, apoio de outra unidade) |
| Pico criado por calendário da própria administração | Heijunka: rever calendário antes de rever quadro |
| Utilização da restrição > 1 | Há acúmulo permanente: explorar e, se não bastar, elevar |
| Restrição com ρ entre 0,85 e 1 | Opera, mas qualquer variabilidade gera fila: proteger com pulmão |
| Não-restrição com ρ > 0,85 | Pouca capacidade protetiva: candidata a próxima restrição |
| ΔT de +1 na restrição < ΔOE | Preferir realocação parcial, horas adicionais ou apoio, antes de provimento |
| Pulmão: com σ de atravessamento conhecido | Usar o pulmão estatístico (z = 1,645 ≈ 95 %); sem σ, regra prática e calibração por zonas |
| Corrente crítica com muitas atividades | Preferir RSS ao corte de 50 % (HERROELEN; LEUS, 2001) |
| Escolha da restrição | Se possível, que seja **estratégica**: o recurso mais caro ou escasso, de propósito |

Utilização-alvo padrão do script: 0,95 na restrição e 0,85 nas demais. Os valores são **parâmetros declarados** a
aprovar na Fase 1, não norma.

## 6. Armadilhas

- **Dimensionar pelo pico do cronograma "mais cedo".** O CPM coloca tudo o mais cedo possível. No exemplo, isso
  sugere 8 fiscais quando a suavização mostra que 5 bastam no mesmo prazo e 4 bastam com uma semana a mais.
- **Confundir suavização com nivelamento.** Suavizar nunca atrasa. Nivelar pode atrasar. Relatar um nivelamento como
  se o prazo fosse mantido é erro de comunicação na Fase 8.
- **Média anual que esconde sazonalidade.** A fórmula do quadro com H anual dilui campanhas. Faça a conta na
  granularidade em que o serviço acontece.
- **Reforçar a etapa errada.** +1 servidor numa não-restrição não aumenta a vazão em nada (ΔT = 0) e eleva a
  despesa. É a mensagem central da TOC para o DFT.
- **Buscar alta utilização em todas as etapas.** Num fluxo com variabilidade, isso destrói o tempo de atravessamento.
  A folga nas não-restrições é proteção.
- **Remanejar sem olhar o fluxo.** No exemplo, mover um engenheiro da vistoria para a análise técnica derruba a vazão
  de 27,45 para 15,56 casos/semana, porque a vistoria vira a restrição.
- **Elevar antes de explorar.** No exemplo, explorar (kit completo) rende +4,12 casos/semana sem custo de pessoal.
  Depois disso, o servidor adicional rende só +0,55.
- **Achar que a corda resolve déficit.** Ela organiza a fila; não cria capacidade.
- **Restrição que "anda".** Depois de elevar, a restrição migra (no exemplo, para a demanda externa, com a vistoria
  como etapa interna mais carregada). Sem o passo 5, a organização otimiza o lugar errado.
- **Pulmão de projeto de 50 % automático.** Tende a superdimensionar (HERROELEN; LEUS, 2001).
- **Transplante acrítico do lean/TOC.** Serviço público tem coprodução com o cidadão, prazos legais e demanda não
  recusável (RADNOR; OSBORNE, 2013).
- **"T" monetário mal definido.** Taxa arrecadada não mede valor público. Declare o critério na Fase 1 ou use
  "casos concluídos".

## 7. Como a etapa entra no DFT

| Fase | Uso do módulo | Produto do aluno |
|---|---|---|
| 5 — Tipo de recurso | Coluna `recurso` (Parte A) e `grupo` (Parte B). A realocação só ocorre dentro do mesmo grupo de competência | Vinculação entrega ↔ perfil; mapa de polivalência |
| 6 — Esforço | `carga` por semana (A) e `tempo_h` por caso (B) vêm do esforço estimado (PERT/P80, módulos 2.1 e 5) | Esforço por entrega e por etapa |
| 7 — Disponibilidade | `horas_servidor_semana` por etapa/perfil; capacidade do recurso no nivelamento | H efetiva por perfil |
| 8 — Consolidação | Três números por perfil: média (quadro permanente), pico suavizado (reforço temporário) e prazo nivelado. Na Parte B: restrição, vazão, dimensionamento pela restrição e alternativas de elevação com ΔT × ΔOE | Relatório com leitura gerencial e alternativas à recomposição por provimento |

A ligação conceitual com o dimensionamento tem quatro pontos:

1. **A restrição define a capacidade do sistema.** A vazão do serviço é a da etapa mais fraca, não a soma das
   capacidades. Um quadro total "suficiente" pode conviver com fila crônica.
2. **Dimensiona-se a não-restrição pela restrição.** Cada etapa é calculada para a vazão-alvo do sistema, com
   utilização alta na restrição e **capacidade protetiva** nas demais. A fórmula é a mesma do curso, aplicada por
   etapa: Qᵢ = D·fᵢ·tᵢ / (Hᵢ·ρᵢ). Com ρ = 1, ela devolve o número canônico do exemplo de nove passos, 7.770 ÷ 928,2 =
   8,37 servidores (teste automatizado).
3. **O quadro é uma alavanca de elevação, não a primeira.** Elevar vem depois de explorar e subordinar. Entre as
   formas de elevar, a ISO 30409:2016 lista realocação, requalificação, terceirização e recrutamento. O DFT que
   recomenda remanejamento não fracassou: aplicou a ordem certa.
4. **A suavização protege a média.** A fórmula Σ D·E / H só é verdadeira se o calendário permitir que a carga seja
   distribuída. O módulo mostra quando isso acontece e quanto custa quando não acontece.

## 8. O que o script faz

| Função | O que faz |
|---|---|
| `atividades_de_dataframe`, `etapas_de_dataframe` | Leem as tabelas (CSV `;` e vírgula decimal) |
| `cpm`, `caminho_critico` | CPM com janelas: ES, EF, LS, LF, folga total e livre, caminho crítico |
| `perfil_recursos`, `indicadores_perfil`, `soma_quadrados` | Histograma e indicadores (área, pico, média, pico/média, Σr², semanas acima da capacidade) |
| `suavizar` | Suavização de prazo fixo pelo método de Burgess |
| `nivelar` | Nivelamento com capacidade, esquema serial, regras `folga`/`lft`/`lst` |
| `pulmao_projeto` | Pulmão da corrente crítica (corte de 50 % ou RSS) |
| `analise_fluxo`, `vazao_sistema` | Passo 1: capacidade, utilização, restrição, vazão, acúmulo |
| `explorar` | Passo 2: reduz o tempo na restrição e, opcionalmente, transfere o trabalho para outra etapa |
| `pulmao_tambor`, `efeito_subordinar` | Passo 3: pulmão (regra e estatístico, em dias e casos, zonas) e efeito da corda sobre WIP e atravessamento |
| `cenarios_elevar`, `cenarios_realocar` | Passo 4: +n servidor em cada etapa (com ΔT, ΔOE e ΔT − ΔOE) e realocação de polivalentes |
| `dimensionar_pela_restricao`, `servidores_necessarios` | Quadro por etapa para a vazão-alvo, com utilização-alvo diferenciada |
| `grafico_perfis`, `grafico_toc` | Histogramas antes/depois e capacidades por cenário |
| `cli` | `dft nivelamento ARQUIVO.csv [opções]`; detecta o modo pelas colunas (`--modo recursos|toc` força) |

Uso:

```
dft nivelamento dados/nivelamento_atividades.csv --capacidades dados/nivelamento_capacidades.csv --grafico a.png
dft nivelamento dados/nivelamento_fluxo.csv --destino-explorar "Análise documental" \
    --custo-servidor 3200 --ganho-por-caso 900 --grafico b.png
```

### Resultado do exemplo fictício A — operação de renovação de alvarás sanitários (10 entregas)

- CPM: 13 semanas. Caminho crítico A01 → A02 → A06 → A08 → A10.
- Fiscais: no cronograma mais cedo, pico de **8** (Σr² total = 184). Suavizado, pico de **5** no mesmo prazo (Σr² =
  134). O pico residual de 5 é inevitável: A03 e A04 disputam a mesma janela de 6 semanas com 7 semanas de trabalho.
- Nivelado com 4 fiscais e 2 administrativos: pico de 4 e término na **semana 14** (+1 semana).
- A carga média de fiscais é de 2,46 servidores. É ela, e não o pico, que entra na fórmula do quadro permanente.
- Corrente crítica ilustrativa: corrente de 13 semanas. Pulmão de 3,25 semanas pelo corte de 50 % e de 3,04 semanas
  por RSS.

### Resultado do exemplo fictício B — licenciamento de obras (5 etapas, 28 processos/semana, H = 28 h/semana)

- **Identificar.** As capacidades são:

    | Etapa | Capacidade (casos/semana) |
    |---|---|
    | Protocolo | 93,33 |
    | Análise documental | 42,00 |
    | Análise técnica | **23,33** (ρ = 1,20) |
    | Vistoria | 31,11 |
    | Emissão | 35,00 |

    A vazão é de 23,33 e acumulam-se 4,67 processos/semana (56 em 12 semanas).

- **Explorar.** Kit completo: −15 % do tempo da análise técnica, transferido para a análise documental. A vazão sobe
    para **27,45** (+4,12) sem servidor novo.
- **Subordinar.** Lead time a montante de 5 dias. O pulmão pela regra é de 2,5 dias (≈ 13,7 processos). O
    estatístico é de 4,0 dias (≈ 22,1 processos; σ = 2,45 d, z = 1,645).
- **Elevar.**
    - Sem explorar, +1 na análise técnica leva a vazão a 28 (ΔT − ΔOE = +R$ 1.000/semana com os parâmetros fictícios).
    - Depois de explorar, o mesmo servidor rende +0,55 (ΔT − ΔOE = −R$ 2.705,88). A elevação mínima necessária é de
      +0,06 servidor (≈ 1,7 h/semana).
    - +1 em qualquer não-restrição: ΔT = 0, ΔOE = +R$ 3.200.
    - Realocar um engenheiro da vistoria: vazão cai para 15,56 (nova restrição).
- **Dimensionar pela restrição** (ρ 0,95/0,85): 12 servidores necessários contra 11 atuais, com a análise técnica +1,
  a vistoria +1 e o protocolo −1.

## Referências

BACELAR-SILVA, Gustavo M.; COX, James F.; RODRIGUES, Pedro Pereira. Outcomes of managing healthcare services using the
theory of constraints: a systematic review. **Health Systems**, [s. l.], v. 11, n. 1, p. 1-16, 2022. DOI:
https://doi.org/10.1080/20476965.2020.1813056. Acesso em: 5 nov. 2026.

BURGESS, A. R.; KILLEBREW, J. B. Variation in activity level on a cyclical arrow diagram. **Journal of Industrial
Engineering**, [s. l.], v. 13, n. 2, p. 76-83, 1962.

CORBETT, Thomas. **Throughput accounting**: TOC's management accounting system. Great Barrington, MA: North River
Press, 1998.

COX, James F., III; SCHLEIER, John G., Jr. (ed.). **Theory of constraints handbook**. New York: McGraw-Hill, 2010.

DATT, Mohit; GUPTA, Ajay; MISRA, Sushendra Kumar; GUPTA, Mahesh. Theory of constraints in healthcare: a systematic
literature review. **International Journal of Quality & Reliability Management**, [s. l.], v. 41, n. 6,
p. 1417-1456, 2024. DOI: https://doi.org/10.1108/IJQRM-02-2022-0056. Acesso em: 5 nov. 2026.

DAVIS, Edward W.; PATTERSON, James H. A comparison of heuristic and optimum solutions in resource-constrained project
scheduling. **Management Science**, [s. l.], v. 21, n. 8, p. 944-955, 1975. DOI:
https://doi.org/10.1287/mnsc.21.8.944. Acesso em: 5 nov. 2026.

GOLDRATT, Eliyahu M. **Critical chain**. Great Barrington, MA: North River Press, 1997.

GOLDRATT, Eliyahu M. **What is this thing called theory of constraints and how should it be implemented?**
Croton-on-Hudson, NY: North River Press, 1990.

GOLDRATT, Eliyahu M.; COX, Jeff. **The goal**: a process of ongoing improvement. 3. ed. rev. Great Barrington, MA:
North River Press, 2004.

HARRIS, Robert B. Packing method for resource leveling (PACK). **Journal of Construction Engineering and
Management**, [s. l.], v. 116, n. 2, p. 331-350, 1990. DOI: https://doi.org/10.1061/(ASCE)0733-9364(1990)116:2(331).
Acesso em: 5 nov. 2026.

HERROELEN, Willy; LEUS, Roel. On the merits and pitfalls of critical chain scheduling. **Journal of Operations
Management**, [s. l.], v. 19, n. 5, p. 559-577, 2001. DOI: https://doi.org/10.1016/S0272-6963(01)00054-7. Acesso em:
5 nov. 2026.

HOPP, Wallace J.; TEKIN, Eylem; VAN OYEN, Mark P. Benefits of skill chaining in serial production lines with
cross-trained workers. **Management Science**, [s. l.], v. 50, n. 1, p. 83-98, 2004. DOI:
https://doi.org/10.1287/mnsc.1030.0166. Acesso em: 5 nov. 2026.

HOPP, Wallace J.; VAN OYEN, Mark P. Agile workforce evaluation: a framework for cross-training and coordination.
**IIE Transactions**, [s. l.], v. 36, n. 10, p. 919-940, 2004. DOI: https://doi.org/10.1080/07408170490487759.
Acesso em: 5 nov. 2026.

INTERNATIONAL ORGANIZATION FOR STANDARDIZATION. **ISO 30409:2016**: human resource management: workforce planning.
Geneva: ISO, 2016. Disponível em: https://www.iso.org/standard/64150.html. Acesso em: 5 nov. 2026.

JORDAN, William C.; GRAVES, Stephen C. Principles on the benefits of manufacturing process flexibility.
**Management Science**, [s. l.], v. 41, n. 4, p. 577-594, 1995. DOI: https://doi.org/10.1287/mnsc.41.4.577. Acesso
em: 5 nov. 2026.

KOLISCH, Rainer; HARTMANN, Sönke. Experimental investigation of heuristics for resource-constrained project
scheduling: an update. **European Journal of Operational Research**, [s. l.], v. 174, n. 1, p. 23-37, 2006. DOI:
https://doi.org/10.1016/j.ejor.2005.01.065. Acesso em: 5 nov. 2026.

LEACH, Larry P. Critical chain project management improves project performance. **Project Management Journal**,
[s. l.], v. 30, n. 2, p. 39-51, 1999. DOI: https://doi.org/10.1177/875697289903000207. Acesso em: 5 nov. 2026.

LIKER, Jeffrey K. **The Toyota way**: 14 management principles from the world's greatest manufacturer. New York:
McGraw-Hill, 2004.

LITTLE, John D. C. A proof for the queuing formula: L = λW. **Operations Research**, [s. l.], v. 9, n. 3,
p. 383-387, 1961. DOI: https://doi.org/10.1287/opre.9.3.383. Acesso em: 5 nov. 2026.

LITVAK, Eugene; FINEBERG, Harvey V. Smoothing the way to high quality, safety, and economy. **New England Journal of
Medicine**, [s. l.], v. 369, n. 17, p. 1581-1583, 2013. DOI: https://doi.org/10.1056/NEJMp1307699. Acesso em:
5 nov. 2026.

LUBITSH, Guy; DOYLE, Christine; VALENTINE, John. The impact of theory of constraints (TOC) in an NHS trust.
**Journal of Management Development**, [s. l.], v. 24, n. 2, p. 116-131, 2005. DOI:
https://doi.org/10.1108/02621710510579482. Acesso em: 5 nov. 2026.

MABIN, Victoria J.; BALDERSTONE, Steven J. The performance of the theory of constraints methodology: analysis and
discussion of successful TOC applications. **International Journal of Operations & Production Management**,
[s. l.], v. 23, n. 6, p. 568-595, 2003. DOI: https://doi.org/10.1108/01443570310476636. Acesso em: 5 nov. 2026.

MCMANUS, Michael L. et al. Variability in surgical caseload and access to intensive care services.
**Anesthesiology**, [s. l.], v. 98, n. 6, p. 1491-1496, 2003. DOI:
https://doi.org/10.1097/00000542-200306000-00029. Acesso em: 5 nov. 2026.

NEUMANN, K.; ZIMMERMANN, J. Procedures for resource leveling and net present value problems in project scheduling
with general temporal and resource constraints. **European Journal of Operational Research**, [s. l.], v. 127, n. 2,
p. 425-443, 2000. DOI: https://doi.org/10.1016/S0377-2217(99)00498-1. Acesso em: 5 nov. 2026.

NEWBOLD, Robert C. **Project management in the fast lane**: applying the theory of constraints. Boca Raton:
St. Lucie Press, 1998. Disponível em:
https://www.routledge.com/Project-Management-in-the-Fast-Lane-Applying-the-Theory-of-Constraints/Newbold/p/book/9781574441956.
Acesso em: 5 nov. 2026.

OHNO, Taiichi. **Toyota production system**: beyond large-scale production. Cambridge, MA: Productivity Press, 1988.

PROJECT MANAGEMENT INSTITUTE. **A guide to the project management body of knowledge (PMBOK guide)**. 6. ed. Newtown
Square, PA: Project Management Institute, 2017. Disponível em: https://www.pmi.org/standards/pmbok. Acesso em:
5 nov. 2026.

RADNOR, Zoe; OSBORNE, Stephen P. Lean: a failed theory for public services? **Public Management Review**, [s. l.],
v. 15, n. 2, p. 265-287, 2013. DOI: https://doi.org/10.1080/14719037.2012.748820. Acesso em: 5 nov. 2026.

RAHMAN, Shams-ur. Theory of constraints: a review of the philosophy and its applications. **International Journal of
Operations & Production Management**, [s. l.], v. 18, n. 4, p. 336-355, 1998. DOI:
https://doi.org/10.1108/01443579810199720. Acesso em: 5 nov. 2026.

RODRIGUES, Adriana Maciel; CAVALCANTI, Marcos do Couto Bezerra; ARAÚJO DA SILVA, Raul; MACEDO, Valéria. Proposta de
metodologia de dimensionamento da força de trabalho para o TCE-RJ. **Revista do Tribunal de Contas do Estado do Rio
de Janeiro**, Rio de Janeiro, v. 2, n. 1, p. 72-83, 2021. DOI: https://doi.org/10.70690/xt46c612. Acesso em: 5 nov.
2026.

RONEN, Boaz. The complete kit concept. **International Journal of Production Research**, [s. l.], v. 30, n. 10,
p. 2457-2466, 1992. DOI: https://doi.org/10.1080/00207549208948166. Acesso em: 5 nov. 2026.

RONEN, Boaz; PLISKIN, Joseph S.; PASS, Shimeon. **Focused operations management for health services
organizations**. San Francisco: Jossey-Bass, 2006.

UMBLE, Michael; UMBLE, Elisabeth J. Utilizing buffer management to improve performance in a healthcare environment.
**European Journal of Operational Research**, [s. l.], v. 174, n. 2, p. 1060-1075, 2006. DOI:
https://doi.org/10.1016/j.ejor.2005.02.059. Acesso em: 5 nov. 2026.

WATSON, Kevin J.; BLACKSTONE, John H.; GARDINER, Stanley C. The evolution of a management philosophy: the theory of
constraints. **Journal of Operations Management**, [s. l.], v. 25, n. 2, p. 387-402, 2007. DOI:
https://doi.org/10.1016/j.jom.2006.04.004. Acesso em: 5 nov. 2026.
