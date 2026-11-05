# 2. Identificação de recursos por atividade

> Módulo `dft_ferramentas/recursos.py` · exemplo `exemplos/2_recursos.py` · dados `dados/recursos_*.csv` ·
> testes `tests/test_recursos.py` · referências `docs/metodos/2_recursos_referencias.json`.
> Dados sempre fictícios. Etapa do curso: **Fase 5 — Identificação do tipo de recurso** (Aula 4), alimentando a
> **Fase 8 — Consolidação** (Produto 4: quadro consolidado em pelo menos duas visões).

## 1. O problema que esta etapa resolve

A fórmula didática do curso, `Q = Σ(Dᵢ × Eᵢ)/H`, responde "quantas pessoas" mas não "quais pessoas". O artigo-base
é autocrítico nesse ponto: na aplicação ao TCE-RJ, o quadro "foi definido apenas com base no número de servidores",
e os modelos comparados "não conseguiram fazer um detalhamento no nível menor do que cargo" (RODRIGUES et al., 2021).
A Portaria SEDGG/ME nº 7.888/2022 torna a lacuna incômoda: o art. 5º, VI liga o dimensionamento a concursos,
contratações temporárias e movimentação de pessoal — e concurso é de cargo, temporário é de função, movimentação é
de perfil. Um quadro em "servidores genéricos" não serve a nenhuma dessas decisões.

A identificação de recursos por atividade é a técnica que fecha essa lacuna: decompõe cada entrega em atividades,
diz **quem** executa cada atividade (perfil, cargo ou competência) e **quanto tempo** cada perfil gasta, e com isso
reparte a carga de trabalho por tipo de recurso. Dela saem quatro das seis visões da Tabela 3 do artigo-base:
quadro por cargo, horas-homem por entrega, servidores por competência e custo por entrega.

A escada de granularidade implícita no artigo — servidor genérico → cargo → nível → competência → competência +
nível — é também uma escada de custo de coleta. O curso recomenda **cargo** no primeiro ciclo e só desce à
competência quando o objetivo declarado na Fase 1 exigir (plano de capacitação, por exemplo).

## 2. Padrões usuais e fontes

Não há um único "método de identificação de recursos" no setor público. A prática combina cinco tradições, cada uma
com um papel preciso no DFT.

### 2.1 Análise do trabalho e descrição de cargos (*job analysis*)

A análise do trabalho é a disciplina que descreve **o que se faz** num posto (tarefas, atividades) e **o que se
exige de quem faz** (conhecimentos, habilidades, capacidades). Sanchez e Levine (2012) registram a passagem da
"análise de cargo" clássica, centrada em descrições estáveis, para a "análise do trabalho" (*work analysis*),
centrada em atividades e papéis que mudam — exatamente o nível em que o DFT opera, pois a unidade de análise é a
entrega, não o cargo.

O padrão de referência internacional é o **O\*NET** (Occupational Information Network), sucessor do *Dictionary of
Occupational Titles* nos Estados Unidos. Peterson et al. (2001) descrevem seu modelo de conteúdo, que separa
descritores **orientados ao trabalhador** (características, requisitos, experiência) de descritores **orientados ao
trabalho** (requisitos ocupacionais, como atividades de trabalho generalizadas e detalhadas, e informação específica
da ocupação, como tarefas). O próprio O\*NET Resource Center apresenta hoje o modelo em seis domínios agrupados
em "sobre a pessoa", "sobre o trabalho" e "sobre o mercado". Para o DFT a lição é a separação: a matriz de recursos
é montada pelo lado do **trabalho** (atividade e tempo); o lado da **pessoa** (competência) é uma segunda camada.

No Brasil, o equivalente funcional é a **Classificação Brasileira de Ocupações** (CBO, 3. ed., 2010), que descreve
cada família ocupacional por áreas de atividade, com descrições produzidas pelo método DACUM, em que trabalhadores
da ocupação descrevem o próprio trabalho em painel. A CBO serve para normalizar a nomenclatura de cargos de uma
prefeitura e para conferir se uma atividade é típica de uma ocupação — não para fixar tempos.

O alerta metodológico mais citado vem de Morgeson e Campion (1997): informações de análise do trabalho coletadas por
autorrelato estão sujeitas a fontes **sociais** (autoapresentação, conformidade ao grupo, desejo de valorizar o
próprio cargo) e **cognitivas** (sobrecarga de informação, heurísticas, categorização) de imprecisão. Na prática do
DFT isso se traduz em atividades infladas, tempos exagerados e atribuição de tarefas simples a cargos mais altos — e
justifica três salvaguardas usadas no curso: coleta de tempos em três pontos com P80 (Fase 6), validação cruzada
entre executores e chefia, e verificação pelo critério do "cargo de menor complexidade possível".

### 2.2 Matriz de responsabilidades (RACI) e matriz atividade × perfil

O instrumento usual para registrar "quem faz o quê" é a **matriz de atribuição de responsabilidades** (RAM). O Guia
PMBOK (PMI, 2017, 6. ed.) apresenta a matriz RACI — *Responsible, Accountable, Consulted, Informed* — como exemplo
de RAM, com a regra de que cada atividade tenha **um único** responsável pela aprovação (A). Em português: R =
executa; A = responde e aprova; C = é consultado (comunicação de mão dupla); I = é informado (mão única).

A RACI qualifica papéis, mas **não mede esforço**. O DFT precisa de uma matriz quantitativa: atividade × perfil com
**tempo**. A decisão do módulo é combinar as duas: cada linha da tabela de atividades é uma combinação (entrega,
atividade, perfil) com tempo e, opcionalmente, a letra RACI. Quem é R ou A tem tempo lançado; C pode ter tempo
(consulta custa tempo); I normalmente tem tempo zero — e o script alerta quando um "I" aparece com tempo. A letra
serve para validar a matriz (um A por atividade, ao menos um R), não para calcular.

Para a camada de competência, a referência brasileira é Brandão e Bahry (2005), que descrevem métodos de mapeamento
de competências (análise documental, entrevistas, grupos focais, questionários) e a redação de competências como
comportamentos observáveis. A matriz atividade × competência tem a mesma forma da matriz atividade × perfil: basta
trocar a coluna `perfil` por `competencia` — o script aceita qualquer rótulo.

### 2.3 Custeio baseado em atividades: ABC e TDABC

O custeio baseado em atividades (ABC) atribui o custo dos recursos às atividades e destas aos produtos. Sua versão
original dependia de entrevistas em que cada pessoa declarava o **percentual** do tempo gasto em cada atividade — o
mesmo vício que o artigo-base critica no "percentual de alocação" (soma sempre 100%, logo nunca revela ociosidade
nem sobrecarga). Kaplan e Anderson (2004; 2007) propuseram o **ABC direcionado pelo tempo** (TDABC), que pede só dois
parâmetros: (1) o **custo por unidade de tempo da capacidade fornecida** e (2) o **tempo necessário** para executar
cada atividade. A taxa de custo da capacidade é o custo do departamento dividido pela **capacidade prática** (não a
teórica): os autores sugerem tipicamente 80% a 85% da capacidade teórica para pessoas, deixando o restante para
pausas, comunicação e treinamento. Os tempos se combinam em **equações de tempo**, com termos condicionais para
variações (por exemplo, "+ 0,8 h se houver diligência"). E a diferença entre capacidade fornecida e capacidade usada
aparece explicitamente como **capacidade não utilizada**, em horas e em reais — em vez de ser rateada aos produtos.

A revisão sistemática de Keel et al. (2017) mostra que o TDABC se difundiu sobretudo em serviços de saúde, com
heterogeneidade de aplicação (muitos estudos medem só parte dos custos ou não descrevem a estimativa da capacidade
prática) — uma advertência útil para o uso em prefeituras: declarar sempre o que entra no custo e como a capacidade
foi estimada.

No modelo referencial federal, o volume de 2022 propõe o custo por entrega `CE = %CT(Eₙ)/Q(Eₙ)` — o percentual do
custo total aplicado a cada entrega dividido pela quantidade de entregas —, considerando na simulação apenas custos
diretos (MENDES; SERRANO; MENESES, 2022). Freitas, Serrano e Ferreira (2021) aplicam a lógica à alocação da força de
trabalho no Executivo federal. **O TDABC é a forma operacional dessa fórmula**: se o custo de pessoal é repartido
pelas horas que cada entrega consome, `%CT(E) × CT = D(E) × Σ tempo × taxa`, e dividir por `Q(E) = D(E)` dá o custo
unitário.

### 2.4 O método WISN da Organização Mundial da Saúde

O WISN (*Workload Indicators of Staffing Need*) é o padrão internacional mais difundido de dimensionamento por carga
de trabalho e por **categoria profissional**. O manual (WHO, 2023, 2. ed., que revisa o manual de Shipp de 1998)
organiza o método em passos: definir categorias e tipos de estabelecimento; estimar o **tempo de trabalho
disponível** por ano; definir os **componentes da carga de trabalho** — atividades de serviço (com estatística de
volume), atividades de **apoio** feitas por todos da categoria (registros, reuniões) e atividades **adicionais**
feitas só por alguns (chefia, comissões); fixar **padrões de atividade** (tempo para executar a atividade com
qualidade profissional nas condições locais); calcular a **carga-padrão** (quanto um profissional produz por ano);
calcular os **fatores de ajuste**; obter o pessoal requerido e interpretar. Na formulação usada na literatura
aplicada (NAJAFPOUR et al., 2023):

- carga-padrão = tempo disponível ÷ padrão de serviço;
- pessoal para serviço = Σ volume anual ÷ carga-padrão;
- CAF = 1 / (1 − ΣCAS/100), em que CAS é o percentual de tempo das atividades de apoio;
- IAF = ΣIAS ÷ tempo disponível, em que IAS são as horas/ano das atividades adicionais;
- requerido = pessoal para serviço × CAF + IAF; **razão WISN** = pessoal atual ÷ requerido.

A aplicação nacional na Namíbia (McQUIDE; KOLEHMAINEN-AITKEN; FORSTER, 2013) mostra os dois usos dos indicadores —
a diferença entre atual e requerido e a razão WISN como medida de pressão de carga — e revela desequilíbrios
**dentro** das categorias e entre tipos de unidade. Carvalho et al. (2022) registram, para o Brasil, que o WISN não
considera demanda reprimida nem programação, e que a maioria dos estudos dimensiona uma só categoria.

Duas lições para o DFT municipal. Primeira: o WISN já é, por construção, um dimensionamento **por perfil** —
nunca agrega categorias. Segunda: o CAF é algebricamente parente do ITP — um ITP de 70% equivale a um CAF com 30% de
apoio (1/0,70 ≈ 1,43). Por isso, no curso (decisão D2), atividade indireta estruturada vira **entrega própria** e não
é contada de novo no CAF nem no ITP; o script oferece `fatores_wisn` só para quem precisa reproduzir o método.

### 2.5 Modelo referencial federal (UnB/Sipec) e Portaria SEDGG/ME nº 7.888/2022

O modelo referencial de dimensionamento da força de trabalho, desenvolvido pela Universidade de Brasília para o
Sipec (SERRANO et al., 2018; MENDES; SERRANO; MENESES, 2022), dimensiona **unidades** a partir de **entregas** e
coleta, à parte, informação de perfil (escolaridade da pessoa e do cargo, vínculos, funções) para a leitura
qualitativa. O banco de entregas é organizado por **categorias de serviço** (transversais, gestão de pessoas,
licitação etc. e grupos finalísticos), e o painel distingue pessoas em apoio exclusivo e em força-tarefa. O tipo de
recurso registrado é "cargo e nível" (RODRIGUES et al., 2021, Tabela 2). Três visões sustentam a consolidação: por
**cargo** (quadro de lotação por cargo), por **entrega** (h/h por entrega e custo) e por **competência**.

A Portaria SEDGG/ME nº 7.888/2022 acrescenta o denominador correto — servidores **em exercício**, não apenas
lotados (art. 4º, IV) — e a finalidade que exige tipo de recurso (art. 5º, VI). E o volume de 2022 estabelece o
limite ético: o DFT indica "necessidades coletivas" de provimento e movimentação, "e não de servidores de forma
isolada" (MENDES; SERRANO; MENESES, 2022). A ISO 30409:2016 completa com a segmentação da força de trabalho como passo
formal (por papel, por empregado, horizontal e vertical) e autoriza organizações pequenas a não segmentar no primeiro
ciclo.

## 3. Passo a passo

1. **Partir do catálogo de entregas** (Fase 3) com a demanda do período já projetada e ajustada por vetores
   (Fase 4). Sem catálogo validado não há matriz.
2. **Decompor cada entrega em atividades** — o suficiente para separar trabalho de perfis diferentes, não mais.
   Regra prática: uma atividade nova só quando muda o executor, muda a exigência legal ou muda a frequência.
3. **Atribuir o perfil de cada atividade** pelas cinco perguntas do L1 (Fase 5, §5.6): reserva legal (privatividade)?
   registro em conselho? formação específica? poder de polícia ou fé pública? Se nenhuma, qual o **cargo de menor
   complexidade** capaz de executá-la? Registrar a resposta na coluna `privativa`.
4. **Registrar papéis RACI** quando houver mais de um perfil na mesma atividade (um único A; ao menos um R).
5. **Lançar o tempo** de cada atividade por perfil (idealmente o P80 da Fase 6) e a **frequência** — quantas vezes a
   atividade ocorre por unidade da entrega (0,25 = um em cada quatro processos tem diligência). É a equação de tempo
   do TDABC escrita como valor esperado.
6. **Montar a matriz** entrega × atividade × perfil e as agregações: horas por entrega, por perfil e a participação
   de cada perfil em cada entrega.
7. **Calcular o denominador por perfil** `H_r` (Fase 7) e `Q_r = carga_r/H_r`; somar os `Q_r` para o total da unidade.
   Declarar a regra de arredondamento antes de rodar.
8. **Custear** pelo TDABC: taxa = custo anual por pessoa ÷ `H_r`; custo da entrega = Σ tempo × taxa.
9. **Comparar com a capacidade em exercício** (lotação × `H_r`): capacidade não utilizada ou sobrecarga, em horas e
   em reais; razão WISN por perfil.
10. **Tratar os alertas** — atividades sem perfil, perfis sem atividade, perfis não cadastrados, entregas sem
    atividade, inconsistências RACI e candidatas à realocação de nível — e só então levar a carga por perfil à
    consolidação.

## 4. Fórmulas

| Símbolo | Definição | Unidade |
|---|---|---|
| `h(e,a,r) = t(e,a,r) × f(e,a,r)` | tempo da atividade *a* da entrega *e* pelo perfil *r* × frequência por unidade | h/unidade |
| `E(e,r) = Σₐ h(e,a,r)` | esforço unitário da entrega *e* no perfil *r* | h/unidade |
| `carga_r = Σₑ Dₑ × E(e,r)` | carga anual do perfil *r* (numerador de `Q_r`) | h/ano |
| `H_r = dias_r × jornada_r × ITP_r` | disponibilidade efetiva de uma pessoa do perfil | h/ano |
| `Q_r = carga_r / H_r`; `Q = Σ_r Q_r` | quantitativo por perfil e da unidade | pessoas |
| `taxa_r = custo_anual_r / H_r` | taxa de custo da capacidade (TDABC) | R$/h |
| `custo(e) = Σ_r E(e,r) × taxa_r` | custo unitário da entrega | R$/unidade |
| `ociosa_r = lotação_r × H_r − carga_r` | capacidade não utilizada (< 0 = sobrecarga) | h/ano |
| `razão WISN_r = lotação_r / Q_r` | pressão de carga (< 1 = pressão) | — |
| `CAF = 1/(1 − ΣCAS/100)`; `IAF = ΣIAS/T` | fatores de ajuste do WISN | — |

A soma dos `Q_r` **não** é igual a `carga_total/H` quando as jornadas diferem — e a repartida é a correta (L1,
Parte 3.2, variante (a)).

## 5. Critérios de decisão

| Decisão | Critério adotado | Por quê |
|---|---|---|
| Nível do recurso | Cargo no 1º ciclo; competência só se o objetivo for capacitação | Custo de coleta e o que o edital de concurso usa |
| Atividade compartilhada | Uma linha por perfil, cada uma com seu tempo | Evita percentuais arbitrários; o percentual sai da conta |
| Tempo a usar | P80 da Fase 6 por atividade (ou da entrega repartido) | Coerência com o diferencial probabilístico do curso |
| RACI | Letra valida a matriz; quem calcula é o tempo | RACI não mede esforço (PMBOK) |
| Capacidade prática no TDABC | `H_r` da Fase 7 (ITP 70% padrão), não os 80–85% de Kaplan e Anderson | Uma só disponibilidade em todo o plano; o custo fica coerente com o quadro |
| Custo-hora | Custo anual ÷ `H_r`, nunca ÷ jornada bruta | A jornada bruta subestima o custo por entrega em ~40% |
| Arredondamento | Declarado na Fase 1; padrão teto; fracionário sempre exibido | Decisão D6 do curso |
| Apoio e atividades indiretas | Entrega própria (D2); CAF/IAF só para reproduzir o WISN | Evita dupla contagem com o ITP |
| Menor complexidade | Não privativa em perfil acima do menor perfil com carga → alerta | Regra do artigo-base; candidata a *redeployment* (ISO 30409) |
| Atividade sem perfil | Mantida na carga da entrega, fora de todos os `Q_r`, com alerta | Mostrar a subestimação em vez de escondê-la |

## 6. Armadilhas

- **Somar jornadas diferentes** num `H` único: `7.770/928,2 = 8,37 → 9`, contra `3,68 + 3,52 → 4 + 4 = 8` no exemplo.
- **Percentual de alocação no lugar de tempo**: soma 100% por construção e esconde ociosidade e sobrecarga — é a
  crítica do artigo-base ao ABC por entrevista, que o TDABC corrige.
- **Alocar tudo no cargo mais alto**: encarece o quadro ideal; a regra é o menor cargo possível.
- **Decompor demais**: atividades de minutos multiplicam erro de estimativa e custo de coleta (MORGESON; CAMPION,
  1997). Decomponha só até separar perfis.
- **Contar cedidos no denominador** ou ignorar terceirizados no numerador: a carga existe independentemente de quem a
  executa; quem se reparte é o denominador (Portaria 7.888/2022, art. 4º, IV).
- **Custo-hora sobre jornada bruta**: R$ 40,23/h contra R$ 67,87/h no exemplo do agente — 41% a menos.
- **Ratear a capacidade ociosa nas entregas**: no TDABC, capacidade não utilizada é informação gerencial separada,
  não custo do produto.
- **Recontar apoio**: CAF ou IAF sobre atividades já catalogadas como entrega própria, ou sobre o que o ITP já cobre.
- **Tratar a sugestão de vínculo da IA como decisão**: privatividade é matéria jurídica; a IA rascunha, a equipe
  valida (o arquivo `recursos_atividades_rascunho.csv` simula esse rascunho).
- **Usar a matriz para avaliar pessoas**: o DFT dimensiona unidades, não servidores.

## 7. Como a etapa entra no DFT

| Fase | Entrada ou saída deste módulo |
|---|---|
| Fase 3 — Levantamento de entregas | Catálogo (`recursos_entregas.csv`): entrega, nome, unidade de medida |
| Fase 4 — Demanda | Coluna `demanda` (já com vetores) |
| **Fase 5 — Tipo de recurso** | **Matriz entrega × atividade × perfil; participação por perfil; alertas** |
| Fase 6 — Esforço | `tempo_h` por atividade (P80 do módulo `distribuicoes`/`simulacao`) |
| Fase 7 — Disponibilidade | `H_r` por perfil (`jornada_h`, `dias_efetivos`, `itp`) |
| Fase 8 — Consolidação | Carga por perfil (`--saida-carga`), `Q_r`, déficit em coluna separada, custo por entrega, capacidade ociosa |

Produto do aluno: a **matriz entrega × tipo de recurso com o percentual de esforço de cada tipo** (L1, Fase 5,
§5.7) e, na Aula 4, o **Produto 4 — quadro consolidado em pelo menos duas visões** (por cargo e por entrega, mais
custo por entrega).

## 8. Exemplo numérico (fictício)

Unidade de licenciamento do exemplo de nove passos (L1, Parte 3.6): cinco entregas, 7.770 h/ano de carga. A
decomposição por atividade reproduz os esforços PERT (1,83; 0,43; 2,50; 1,20; 4,00 h) e a repartição do L1, §5.6:
E1 60% técnica, E3 40% técnica, E5 integralmente técnica.

| Perfil | Carga (h/ano) | `H_r` | `Q_r` | Q (teto) | Lotação | Déficit | Razão WISN |
|---|---|---|---|---|---|---|---|
| Técnico (6 h/dia) | 3.417,6 | 928,2 | 3,68 | 4 | 2 | +2 | 0,54 |
| Agente administrativo (8 h/dia) | 4.352,4 | 1.237,6 | 3,52 | 4 | 5 | −1 | 1,42 |
| Estagiário (sem atividade) | 0 | 618,8 | 0 | 0 | 2 | −2 | — |
| **Total** | **7.770,0** | | **7,20** | **8** | | | |

O total é 8, e não o 9 do cálculo agregado. A leitura gerencial muda: não "falta uma pessoa", mas faltam dois
técnicos e sobra capacidade administrativa — candidata a realocação, não a concurso. No Passo 8 do mesmo exemplo
(todos com `H` = 928,2 h e E3 no agente), o script reproduz 2.517,6 h → 2,71 → 3 técnicos e 5.252,4 h → 5,66 → 6
agentes.

Custo (TDABC, custo anual fictício R$ 168 mil para o técnico e R$ 84 mil para o agente): taxas de R$ 181,00/h e
R$ 67,87/h; vistoria R$ 723,98 por unidade; certidão R$ 29,19. A certidão (E2) tem 3.600 de 6.480 ocorrências
(**56%** do volume) e 20% da carga — alto volume e baixa complexidade, candidata a estudo de automação.
*Nota de conferência:* o L1 (Parte 3.6, Passo 9) registra "46% do volume"; o cálculo correto com os dados do próprio
exemplo é 3.600/6.480 = 55,6%. A conclusão qualitativa não muda.

## 9. O que o script faz

| Função / opção | O que faz | Fonte do método |
|---|---|---|
| `matriz_recursos` | Matriz longa entrega × atividade × perfil, h/unidade e h/ano | RAM (PMI, 2017); L1, Fase 5 |
| `horas_por_entrega` | h/h por entrega, % volume × % carga | Tabela 3 do artigo-base |
| `matriz_entrega_perfil`, `participacao_perfis` | Tabela cruzada e participação de cada perfil | L1, Fase 5, §5.7 |
| `disponibilidade_perfil` | `H = dias × jornada × ITP` | L1, Parte 3.5; L2, §4.2 |
| `taxa_custo_capacidade`, `custo_por_entrega` | Taxa por capacidade prática e custo por entrega | Kaplan e Anderson (2004; 2007); Mendes, Serrano e Meneses (2022) |
| `carga_por_perfil` | `Q_r`, arredondamento declarado, déficit, razão WISN, capacidade ociosa e seu custo | L2, §4.3; WHO (2023); TDABC |
| `alertas` | Sem perfil, perfil sem atividade, não cadastrado, entrega sem atividade, RACI, menor complexidade | L1, Fase 5, §5.7; PMI (2017) |
| `fatores_wisn`, `requerido_wisn` | CAF, IAF e pessoal requerido do WISN | WHO (2023); Najafpour et al. (2023) |
| `dft recursos <atividades.csv>` | Relatório; `--entregas`, `--perfis` (padrão: arquivos irmãos), `--arredondamento`, `--saida-carga`, `--grafico` | — |

Formato dos CSVs (separador `;`, vírgula decimal): **entregas** `entrega;nome;demanda;unidade_medida`;
**atividades** `entrega;atividade;perfil;tempo_h;frequencia;raci;privativa` (as três últimas opcionais);
**perfis** `perfil;cargo;complexidade;jornada_h;dias_efetivos;itp;custo_anual;lotacao_atual` (aceita
`disponibilidade_h` no lugar de jornada e dias, e `custo_hora` no lugar de `custo_anual`).

## Referências

BRANDÃO, Hugo Pena; BAHRY, Carla Patricia. Gestão por competências: métodos e técnicas para mapeamento de
competências. **Revista do Serviço Público**, Brasília, v. 56, n. 2, p. 179-194, abr./jun. 2005. DOI:
10.21874/rsp.v56i2.224. Disponível em: https://doi.org/10.21874/rsp.v56i2.224. Acesso em: 5 nov. 2026.

BRASIL. Ministério da Economia. Secretaria Especial de Desburocratização, Gestão e Governo Digital. **Portaria
SEDGG/ME nº 7.888, de 1º de setembro de 2022**. Estabelece orientações aos órgãos e entidades da administração
pública federal quanto aos procedimentos a serem observados para o dimensionamento da força de trabalho. Diário
Oficial da União: seção 1, Brasília, DF, 2 set. 2022.

BRASIL. Ministério do Trabalho e Emprego. **Classificação Brasileira de Ocupações: CBO 2010**. 3. ed. Brasília, DF:
MTE, 2010.

CARVALHO, Desirée dos Santos et al. Planejamento e dimensionamento da força de trabalho em saúde no Brasil: avanços
e desafios. **Saúde em Debate**, Rio de Janeiro, v. 46, n. 135, p. 1215-1237, out./dez. 2022. DOI:
10.1590/0103-1104202213519. Disponível em: https://doi.org/10.1590/0103-1104202213519. Acesso em: 5 nov. 2026.

FREITAS, Heitor Silveira; SERRANO, André Luiz Marques; FERREIRA, Lucas Oliveira Gomes. Alocação da força de trabalho
em uma amostra do poder executivo brasileiro: levantamento dos custos e avaliação. **Revista do Serviço Público**,
Brasília, v. 72, n. 1, p. 163-199, 2021. DOI: 10.21874/art-6-2022-01-19-61e8631b4c164. Disponível em:
https://doi.org/10.21874/art-6-2022-01-19-61e8631b4c164. Acesso em: 5 nov. 2026.

INTERNATIONAL ORGANIZATION FOR STANDARDIZATION. **ISO 30409:2016**: human resource management: workforce planning.
Geneva: ISO, 2016.

KAPLAN, Robert S.; ANDERSON, Steven R. Time-driven activity-based costing. **Harvard Business Review**, Boston,
v. 82, n. 11, p. 131-138, Nov. 2004.

KAPLAN, Robert S.; ANDERSON, Steven R. **Time-driven activity-based costing**: a simpler and more powerful path to
higher profits. Boston: Harvard Business School Press, 2007. ISBN 978-1-4221-0171-1.

KEEL, George; SAVAGE, Carl; RAFIQ, Muhammad; MAZZOCATO, Pamela. Time-driven activity-based costing in health care:
a systematic review of the literature. **Health Policy**, Amsterdam, v. 121, n. 7, p. 755-763, July 2017. DOI:
10.1016/j.healthpol.2017.04.013. Disponível em: https://doi.org/10.1016/j.healthpol.2017.04.013. Acesso em:
5 nov. 2026.

McQUIDE, Pamela A.; KOLEHMAINEN-AITKEN, Riitta-Liisa; FORSTER, Norbert. Applying the workload indicators of staffing
need (WISN) method in Namibia: challenges and implications for human resources for health policy. **Human
Resources for Health**, London, v. 11, art. 64, 2013. DOI: 10.1186/1478-4491-11-64. Disponível em:
https://doi.org/10.1186/1478-4491-11-64. Acesso em: 5 nov. 2026.

MENDES, Nara Cristina Ferreira; SERRANO, André Luiz Marques; MENESES, Pedro Paulo Murce (org.). **Dimensionamento na
administração pública federal**: avanços e resultados alcançados. Brasília, DF: FT/UnB; Ministério da Economia,
2022. ISBN 978-65-86503-90-6. DOI: 10.26512/9786586503906. Disponível em: https://doi.org/10.26512/9786586503906.
Acesso em: 5 nov. 2026.

MORGESON, Frederick P.; CAMPION, Michael A. Social and cognitive sources of potential inaccuracy in job analysis.
**Journal of Applied Psychology**, Washington, DC, v. 82, n. 5, p. 627-655, Oct. 1997. DOI:
10.1037/0021-9010.82.5.627. Disponível em: https://doi.org/10.1037/0021-9010.82.5.627. Acesso em: 5 nov. 2026.

NAJAFPOUR, Zhila; ZARE NASIRI, Mohsen; NOZARIAN, Mohammad-Hussein; KELIDDAR, Iman; SHAYANFARD, Kamran. Estimating
the number of required nurses in different types of hospitals: an application of the workload indicators of
staffing needs (WISNS) method. **PLOS ONE**, San Francisco, v. 18, n. 12, e0295213, Dec. 2023. DOI:
10.1371/journal.pone.0295213. Disponível em: https://doi.org/10.1371/journal.pone.0295213. Acesso em: 5 nov. 2026.

NATIONAL CENTER FOR O\*NET DEVELOPMENT. **The O\*NET Content Model**. Raleigh: O\*NET Resource Center, [2026].
Disponível em: https://www.onetcenter.org/content.html. Acesso em: 5 nov. 2026.

PETERSON, Norman G. et al. Understanding work using the Occupational Information Network (O\*NET): implications for
practice and research. **Personnel Psychology**, Hoboken, v. 54, n. 2, p. 451-492, June 2001. DOI:
10.1111/j.1744-6570.2001.tb00100.x. Disponível em: https://doi.org/10.1111/j.1744-6570.2001.tb00100.x. Acesso em:
5 nov. 2026.

PROJECT MANAGEMENT INSTITUTE. **A guide to the project management body of knowledge (PMBOK guide)**. 6th ed.
Newtown Square: PMI, 2017. ISBN 978-1-62825-184-5.

RODRIGUES, Adriana Maciel; CAVALCANTI, Marcos do Couto Bezerra; ARAÚJO DA SILVA, Raul; MACEDO, Valéria. Proposta de
metodologia de dimensionamento da força de trabalho para o TCE-RJ. **Revista do Tribunal de Contas do Estado do Rio
de Janeiro**, Rio de Janeiro, v. 2, n. 1, p. 72-83, 2021. DOI: 10.70690/xt46c612. Disponível em:
https://doi.org/10.70690/xt46c612. Acesso em: 5 nov. 2026.

SANCHEZ, Juan I.; LEVINE, Edward L. The rise and fall of job analysis and the future of work analysis. **Annual
Review of Psychology**, Palo Alto, v. 63, p. 397-425, 2012. DOI: 10.1146/annurev-psych-120710-100401. Disponível em:
https://doi.org/10.1146/annurev-psych-120710-100401. Acesso em: 5 nov. 2026.

SERRANO, André Luiz Marques et al. (org.). **Dimensionamento na administração pública federal**: uma ferramenta do
planejamento da força de trabalho. Brasília, DF: Enap, 2018.

WORLD HEALTH ORGANIZATION. **Workload indicators of staffing need (WISN)**: user manual. 2nd ed. Geneva: WHO, 2023.
ISBN 978-92-4-007006-6. Disponível em: https://www.who.int/publications/i/item/9789240070066. Acesso em:
5 nov. 2026.
