# 4. Previsão de demanda — dossiê do método

> **Fase do DFT:** Fase 4 — Estimativa de demanda ("Quantas vezes cada entrega ocorre?").
> **Produto do aluno:** Produto 3 — planilha de estimativas do setor-recorte (demanda projetada + esforço por entrega).
> **Script:** `dft_ferramentas/previsao.py` · `dft previsao dados/previsao_serie.csv --horizonte 12`.
> **Caso canônico:** CT-3 (dossiê L2, §6) — Holt vence SES (RMSE 6,20 < 7,41); Σ projeção de 12 meses = 2.481,05;
> vetores +12% (h ≥ 4) e −5% (h ≥ 7); Σ ajustada = 2.637,19; o quadro passa de 24 para 25 servidores.

Este documento **complementa** o L2 (§§1 e 2) e a apostila do Módulo 3, que já tratam das competições de previsão
(M4), da suavização exponencial amortecida, dos ajustes por julgamento, do MASE e dos serviços de emergência. Aqui entra o
padrão usual de cada técnica que o script implementa — Box-Jenkins, Croston/TSB, STL, origem móvel, bootstrap, hierarquia — e as
decisões de implementação.

---

## 1. O padrão usual e de onde ele vem

### 1.1 O manual de referência

O padrão de fato da previsão de séries temporais em gestão é o livro aberto *Forecasting: principles and practice*,
3.ª edição (FPP3), de Hyndman e Athanasopoulos (2021). O curso já o adota como norma (L2). O fluxo que ele ensina é:
**olhar a série** (gráfico e decomposição) → **métodos de referência** → **modelos** (suavização exponencial, ARIMA,
regressão dinâmica) → **avaliação fora da amostra** → **intervalos de previsão** → **ajuste por julgamento só com
evidência**. O *e-Handbook* do NIST/SEMATECH (seção 6.4) dá a mesma sequência para quem vem do controle de processos:
médias móveis, suavização exponencial simples, dupla e tripla, e depois Box-Jenkins. A revisão enciclopédica de
Petropoulos et al. (2022), assinada por 80 autores, confirma que esse é o núcleo consensual da área.

### 1.2 O que as competições M dizem sobre simples × complexo

As competições de Makridakis são o principal teste empírico de métodos de previsão.

- **M3 (2000), 3.003 séries.** A competição reafirmou as conclusões das anteriores. Métodos estatisticamente sofisticados
  não produzem, necessariamente, previsões mais acuradas que os simples. A ordem dos métodos muda conforme a medida de
  erro. Combinar métodos supera, em média, os métodos combinados. E a acurácia depende do horizonte (MAKRIDAKIS; HIBON,
  2000). O teste estatístico formal de Koning et al. (2005) **confirmou** que a complexidade não se relaciona com a
  acurácia. Mas **não confirmou** a superioridade das combinações. Portanto, a regra "complexo não é melhor" tem lastro
  mais firme que a regra "combinar sempre ganha".
- **M4 (2018–2020), 100.000 séries e 61 métodos.** Já sintetizada no Encontro 3: combinações e híbridos no topo, e
  aprendizado de máquina puro decepcionante (MAKRIDAKIS; SPILIOTIS; ASSIMAKOPOULOS, 2020).
- **M5 (2022), 42.840 séries hierárquicas de vendas de varejo**, muitas intermitentes, com 30.490 séries no nível mais
  baixo (MAKRIDAKIS; SPILIOTIS; ASSIMAKOPOULOS, 2022). Aqui o cenário muda: há milhares de séries **relacionadas** e
  variáveis explicativas (preço, calendário). Nessas condições, métodos de aprendizado de máquina treinados em conjunto
  rendem. A solução que ficou em 2.º lugar combinou árvores de gradiente (LightGBM) nas séries intermitentes da base com
  um alinhamento de cima para baixo da hierarquia (ANDERER; LI, 2022).

**Leitura para o DFT municipal.** Uma unidade típica tem de 5 a 40 entregas, com 12 a 60 meses de histórico, sem
variáveis explicativas registradas. É o oposto do M5. A evidência aponta para **métodos simples bem validados**, com a
complexidade só como candidata que precisa vencer os simples na validação.

### 1.3 Suavização exponencial

Gardner (1985) organizou a prática de suavização exponencial — escolha de parâmetros, inicialização e manutenção de
sistemas de previsão. Em 2006, ao atualizar a revisão, ele registra dois avanços. O teórico foi a fundamentação
estatística completa por modelos de espaço de estados com uma única fonte de erro. O prático foi um método robusto para
tendências amortecidas. Sobre a escolha de modelos, ele conclui que a pesquisa é inconclusiva e que "ainda é difícil
superar a aplicação de uma tendência amortecida a toda série temporal" (GARDNER, 2006, resumo, tradução nossa). A
formulação em espaço de estados (HYNDMAN et al., 2002; HYNDMAN et al., 2008) é o que permite calcular verossimilhança,
AIC, intervalos de previsão e simulação para cada método da família.

### 1.4 Box-Jenkins e ARIMA

O método de Box e Jenkins (BOX et al., 2015, 5. ed.) é o ciclo **identificação → estimação → diagnóstico** de modelos
ARIMA(p, d, q)(P, D, Q)ₘ. Primeiro, diferencia-se a série até a estacionariedade (d, D). Depois, escolhem-se as ordens
autorregressiva (p) e de médias móveis (q), estima-se por máxima verossimilhança e verifica-se se os resíduos são ruído
branco. A versão automática usual é o algoritmo de Hyndman e Khandakar (2008), descrito em FPP3 §9.7:

1. d por testes KPSS sucessivos (H₀: estacionária); D = 1 se a força sazonal F_S ≥ 0,64 (FPP3, §9.1);
2. quatro modelos iniciais — (2,d,2), (0,d,0), (1,d,0), (0,d,1) —, com constante se d ≤ 1;
3. busca passo a passo variando p e q em ±1 e incluindo ou excluindo a constante, enquanto o **AICc** cair.

ARIMA e suavização exponencial se sobrepõem — os modelos ETS lineares são casos de ARIMA —, mas nenhuma família contém a
outra inteira (FPP3, §9.10).

### 1.5 Regressão dinâmica e variáveis externas

Quando há fatores externos **conhecidos para o futuro**, o padrão é a regressão com erros ARIMA (FPP3, cap. 10), já
detalhada no L2 §1.5. Os preditores típicos do setor público são três (FPP3, §7.4):

| Fator | Variável | Exemplo municipal (fictício) |
|---|---|---|
| Feriados e dias úteis | número de dias úteis no mês; dummy de feriado | protocolo que fecha no carnaval |
| Mudança legislativa | **degrau** (0 antes, 1 depois) | nova lei que amplia o rol de pareceres |
| Campanha, mutirão, recadastramento | **pulso** (1 só no mês) | mutirão de regularização fiscal |

O regressor precisa ser conhecido no horizonte. Dias úteis e feriados são; o efeito de uma lei ainda não aplicada não é.
Por isso o DFT trata mudança legislativa como **vetor de demanda** (D4). O vetor é uma intervenção do tipo degrau
aplicada fora do modelo, com magnitude justificada por evidência externa. Larson e Sinclair (2022), já no Encontro 3,
mostram o valor de uma indicadora de data de decreto numa quebra estrutural.

### 1.6 Demanda intermitente

Serviços raros — recursos de segunda instância, perícias, tombamentos — geram séries com muitos zeros. Croston (1972)
mostrou que a suavização exponencial comum produz estoques inadequados nesses casos. Propôs suavizar separadamente o
**tamanho** das demandas não nulas e o **intervalo** entre elas, e prever a razão entre os dois. Syntetos e Boylan
(2005) mostraram que o método é viesado e propuseram a correção SBA (fator 1 − α/2), superior em 3.000 séries reais.
Teunter, Syntetos e Babai (2011) propuseram o **TSB**, que atualiza a **probabilidade** de haver demanda em todo período.
Por isso o TSB é não viesado e decai quando a demanda desaparece (obsolescência). O Croston, ao contrário, congela a
previsão em meses sem demanda. A escolha do método segue a classificação de Syntetos, Boylan e Croston (2005), pelo
intervalo médio entre demandas (ADI) e pelo quadrado do coeficiente de variação dos tamanhos (CV²), com fronteiras
ADI = 1,32 e CV² = 0,49. FPP3 (§13.2) adverte que Croston não corresponde a modelo estatístico e, portanto, não tem
intervalo de previsão algébrico.

### 1.7 Decomposição STL

A STL (Cleveland et al., 1990) separa tendência, sazonalidade e resto por regressões locais (*loess*). FPP3 (§3.6)
lista suas vantagens sobre a decomposição clássica: aceita qualquer período sazonal, deixa a sazonalidade mudar no tempo,
controla a suavidade da tendência e pode ser robusta a valores atípicos. A limitação é que só faz decomposição aditiva; a
multiplicativa sai do logaritmo. Dela saem as **forças** de FPP3 §4.3, usadas na árvore de decisão:
F_T = max(0; 1 − Var(R)/Var(T + R)) e F_S = max(0; 1 − Var(R)/Var(S + R)).

### 1.8 Validação, erro e intervalos

**Origem móvel.** Tashman (2000) mostrou que a avaliação fora da amostra com origem móvel torna o teste mais eficiente e
confiável numa série individual. Ele também constatou que poucos softwares a implementavam corretamente. FPP3 (§5.10) a
chama de validação cruzada de séries temporais. Bergmeir e Benítez (2012) discutem as variantes em blocos. A regra comum
a todos é que **o treino sempre antecede o teste** — nunca se embaralha.

**Medidas.** RMSE para escolher, MAE para comunicar, MAPE só sem zeros e MASE para comparar séries e contagens raras
(L2, §1.7; HYNDMAN; KOEHLER, 2006).

**Intervalos por bootstrap.** Thombs e Schucany (1990) propuseram intervalos de previsão por reamostragem dos resíduos,
sem supor erros normais. FPP3 (§5.5) adota o procedimento para qualquer modelo com resíduos de um passo. Bergmeir,
Hyndman e Benítez (2016) levaram a ideia ao *bagging*: reamostram o resto da STL em blocos e combinam modelos ajustados
às séries reamostradas.

### 1.9 Previsão hierárquica e agrupada

As entregas se somam em unidades, as unidades em secretarias e as secretarias no município. Previsões feitas
separadamente em cada nível não batem entre si. Hyndman et al. (2011) propuseram prever todos os níveis e
**reconciliar** por regressão, com resultado melhor que *bottom-up* (somar a base) e *top-down* (ratear o topo). O método
**MinT** (WICKRAMASURIYA; ATHANASOPOULOS; HYNDMAN, 2019) usa a covariância completa dos erros e tem solução fechada. A
evidência no setor público diverge. Houve ganho em ambulâncias no Reino Unido (ROSTAMI-TABAR; HYNDMAN, 2025) e nenhum
ganho na resolução tática canadense (REZAEI; INGOLFSSON, 2024). O tema fica como controvérsia (Encontro 3).

### 1.10 Julgamento, demanda reprimida e demanda × oferta

Lawrence et al. (2006) revisaram mais de 200 estudos. O julgamento humano pode trazer ganho significativo de acurácia,
mas está sujeito a muitos vieses. Daí o desenho do vetor (D4): ajuste só com evidência, documentado, com antes/depois.
No serviço público há um problema anterior, que é o **registro mostrar o atendido, não o demandado**. Onde há fila, a
série de baixas é teto de capacidade disfarçado de demanda (L1, verbete 5). A literatura de "desrestrição" de demanda é
quase toda de gestão de receitas (PRICE; FOWKES; HOPMAN, 2019, no Encontro 3). Para o município, o padrão prático é usar
**entradas** (protocolos abertos), e não saídas; registrar o estoque represado à parte; e apresentar o resultado com e
sem a demanda reprimida (L1, §3.8.8). Rostami-Tabar et al. (2022) chamam essa agenda de *forecasting for social good*.
Nela, previsão serve a objetivos sociais (acesso, equidade) antes de objetivos financeiros. Prever só o que a capacidade
atual deixou passar perpetua a subprovisão.

---

## 2. Passo a passo

1. **Defina a entrega e a contagem.** Uma série por entrega do catálogo (Fase 3), na unidade de medida definida. Use a
   data de **entrada** (demanda), não a de conclusão (oferta).
2. **Monte a série mensal** e trate a qualidade: meses faltantes, mudança de sistema, reclassificação. Não preencha zero
   onde o dado falta.
3. **Olhe o gráfico** e calcule a decomposição STL (se houver ≥ 2 ciclos), F_T e F_S, e a classificação ADI/CV².
4. **Passe pela árvore de decisão** (seção 4) e obtenha os candidatos.
5. **Valide por origem móvel** com todos os candidatos nas mesmas origens: para t = início…n − 1, ajuste em y₁…y_t e
   preveja y_{t+1}… (ou h passos). Nunca embaralhe.
6. **Escolha pelo menor RMSE.** Em empate, fica o mais simples. Os métodos de referência estão entre os candidatos, então
   o modelo só é escolhido se vencer o ingênuo, a média e o drift.
7. **Reajuste o vencedor na série inteira** e projete o horizonte (12 meses no Produto 3), com intervalo de 80% por
   bootstrap.
8. **Leia os alertas:** parâmetro na fronteira da grade (amostra curta), MAPE indefinido, intervalo que explode.
9. **Aplique vetores** apenas se houver fato novo com fonte (lei, decreto, contrato, plano). Registre o antes/depois e a
   trilha.
10. **Registre** no Produto 3 o método, a validação, a projeção base, a ajustada, o intervalo e a semente. Avalie o acerto
    ao fim do exercício (FPP3, §6.2, princípio 4).

---

## 3. Fórmulas

| Método | Previsão ŷ_{T+h|T} | Atualização |
|---|---|---|
| Média | ȳ | — |
| Ingênuo | y_T | — |
| Ingênuo sazonal | y_{T+h−m(k+1)}, k = ⌊(h−1)/m⌋ | — |
| Drift | y_T + h(y_T − y₁)/(T − 1) | — |
| SES | ℓ_T | ℓ_t = αy_t + (1−α)ℓ_{t−1} |
| Holt | ℓ_T + h·b_T | ℓ_t = αy_t + (1−α)(ℓ_{t−1}+b_{t−1}); b_t = β(ℓ_t−ℓ_{t−1}) + (1−β)b_{t−1} |
| Holt amortecido | ℓ_T + (φ+…+φʰ)b_T | idem, com φ·b_{t−1}; 0,80 ≤ φ ≤ 0,98 |
| Holt-Winters aditivo | ℓ_T + h·b_T + s_{T+h−m(k+1)} | ℓ_t = α(y_t − s_{t−m}) + (1−α)(ℓ_{t−1}+b_{t−1}); s_t = γ(y_t − ℓ_{t−1} − b_{t−1}) + (1−γ)s_{t−m} |
| Croston | q̂/â | q̂ e â por SES nos períodos com demanda |
| SBA | (1 − α/2)·q̂/â | — |
| TSB | p̂·ẑ | p̂_t = β·1[y_t>0] + (1−β)p̂_{t−1} em todo t; ẑ por SES quando y_t > 0 |

**Erros** (e = y − ŷ): MAE = média|e|; RMSE = √média(e²); MAPE = média|100e/y| (só se y > 0); MASE = média|e|/Q,
Q = média|y_t − y_{t−m}| no treino (m = 1 se não sazonal).
**Intervalo normal** (h = 1): ŷ ± 1,28σ̂ (80%), σ̂ = √(Σe²/(n−K)).
**Bootstrap:** para cada caminho, ŷ*_{T+i} = previsão a partir do estado simulado + e*, com e* sorteado com reposição
entre os resíduos de um passo, e o estado atualizado com ŷ*. Os percentis 10 e 90 dos caminhos formam o intervalo.
**Vetores:** F(h) = Π fator_v (vetores vigentes em h); ajustada(h) = base(h)·F(h).

---

## 4. Critérios de decisão

**Árvore por tamanho de série** (amplia L2 §1.10; m = 12 para séries mensais):

```
n < 12  → não modelar: quantidade planejada ou estimativa estruturada de gestores (artigo-base, Fase 4; FPP3 §6.2)
ADI ≥ 1,32 (muitos zeros) → média, ingênuo, Croston, SBA, TSB  (MAE/RMSE; MAPE não se aplica)
12 ≤ n < 2m + 3 → média, ingênuo, drift, SES, Holt, Holt amortecido  (sem Holt-Winters: < 2 ciclos + folga de teste)
n ≥ 2m + 3 → + ingênuo sazonal;
             F_S ≥ 0,64 → + Holt-Winters aditivo (e multiplicativo, se y > 0)
             n ≥ 3m     → + ARIMA automático
Sempre: origem móvel, menor RMSE, empate → mais simples.
```

| Pergunta | Critério | Fonte |
|---|---|---|
| Aditivo ou multiplicativo? | amplitude sazonal constante → aditivo; proporcional ao nível → multiplicativo. Na dúvida, os dois entram e a validação decide | FPP3 §8.3 |
| Holt ou Holt amortecido? | horizonte longo (quadro de 3–5 anos) → amortecido por prudência; Holt puro só se vencer na validação e não houver β na fronteira | Gardner (2006); L2 §1.4.3 |
| ARIMA? | só com ≥ 3 ciclos e se vencer a suavização exponencial; nunca por ser "mais sofisticado" | M3; Koning et al. (2005) |
| Qual métrica escolhe? | RMSE (FPP3 §5.10); MASE para comparar séries; MAPE só sem zeros | Hyndman e Koehler (2006) |
| Regressão dinâmica? | só se o regressor for conhecido no horizonte (dias úteis, feriados) e houver ≥ 3 anos | FPP3 cap. 10 |
| Hierarquia? | reconciliar quando entregas, unidades e secretaria são previstas separadamente e precisam somar | Hyndman et al. (2011) |
| Vetor? | só com fonte e evidência; nunca entre −3% e +3% | D4; FPP3 §6.7 |

---

## 5. Armadilhas

1. **Prever a oferta.** Contar processos concluídos numa unidade com fila mede a capacidade, não a demanda.
2. **Embaralhar a validação** (*shuffle*, *k-fold* comum). Isso vaza o futuro para o treino e faz qualquer método parecer
   bom (HEWAMALAGE; ACKERMANN; BERGMEIR, 2023).
3. **Escolher pelo ajuste dentro da amostra.** O SSE do treino sempre favorece o modelo mais flexível.
4. **Holt puro em horizonte longo.** A tendência cresce sem limite; em 5 anos, o quadro explode.
5. **Holt-Winters com menos de dois ciclos.** A sazonalidade estimada com um único ano é ruído.
6. **MAPE com zeros ou valores pequenos.** É indefinido ou explosivo, e pune assimetricamente.
7. **Intervalo ignorado.** Informar só o ponto esconde que, com 12 meses de histórico, o intervalo de 80% em h = 12 pode
   ter ±20%. No CT-3, o bootstrap dá [123,6; 268,2] em h = 12, contra a base de 227,6, porque β ≈ 1 propaga cada choque
   para a tendência.
8. **Vetor pequeno ou otimista.** "Acho que vai crescer uns 2%" não é vetor (D4).
9. **Contar a mudança duas vezes.** Se a lei já vigora há meses e aparece na série, o modelo já a captou; um vetor por
   cima duplica o efeito.
10. **Somar previsões de níveis diferentes** sem reconciliar. O total da secretaria fica diferente da soma das unidades.
11. **Croston em série que acabou.** Ele não atualiza em meses sem demanda; use TSB quando a entrega pode desaparecer.
12. **Confundir "projeção linear simples" com método.** É o drift — um método de referência, não um modelo.

---

## 6. Como a etapa entra no DFT

- **Fase 4** do artigo-base (RODRIGUES et al., 2021), que critica a média ou a projeção linear dos últimos meses e pede
  "técnicas mais assertivas de previsão". A previsão produz, para cada entrega do catálogo (Fase 3), a **demanda anual
  projetada**.
- **Produto 3** do aluno (Aula 3): a coluna "demanda projetada" da planilha de estimativas, com método, validação,
  intervalo e vetores.
- **Encadeamento:** demanda anual × esforço P80 (Fase 6) = horas; ÷ disponibilidade efetiva (Fase 7) = quantitativo
  (Fase 8). No CT-3, a demanda ajustada de 2.637,19 pareceres × 9,075 h (P80) mais 4.500 h de certidões = 28.432,51 h;
  ÷ 1.153,6 h = 24,65 → **25 servidores**. Sem os vetores, 2.481,05 pareceres → 24 servidores. O relatório da Fase 8
  mostra os dois números.
- **Quantidade planejada** (auditorias, turmas) não passa por este módulo: entra direto do plano.

---

## 7. O que o script faz

| Função | O que faz | Fonte do método |
|---|---|---|
| `prever_media`, `prever_ingenuo`, `prever_sazonal_ingenuo`, `prever_drift`, `metodos_referencia` | Os quatro métodos de referência | FPP3 §5.2 |
| `ajustar_ses`, `ajustar_holt`, `ajustar_holt_amortecido` | Grade α, β (passo 0,01) e φ ∈ {0,80…0,98}; ℓ₀ = y₁, b₀ = y₂ − y₁; alerta de fronteira | FPP3 §§8.1–8.2; L2 §5.4; motor TS |
| `ajustar_holt_winters` | Holt-Winters aditivo/multiplicativo (statsmodels, estado estimado) | FPP3 §8.3 |
| `ajustar_arima_auto`, `numero_diferencas` | ARIMA automático: D por F_S, d por KPSS, busca passo a passo por AICc | Hyndman e Khandakar (2008); FPP3 §§9.1, 9.7 |
| `decompor_stl` | STL robusta + forças F_T e F_S | Cleveland et al. (1990); FPP3 §§3.6, 4.3 |
| `classificar_demanda`, `prever_croston`, `prever_tsb` | ADI/CV²; Croston, SBA e TSB | Syntetos, Boylan e Croston (2005); Croston (1972); Teunter et al. (2011) |
| `medidas_erro`, `escala_mase` | MAE, RMSE, MAPE (guarda de zero), MASE | FPP3 §5.8; Hyndman e Koehler (2006) |
| `recomendar_metodos` | Árvore de decisão por tamanho de série | L2 §1.10, ampliada |
| `avaliar_treino_teste`, `validacao_origem_movel`, `escolher_metodo` | Validação sem embaralhar; menor RMSE; empate → mais simples | FPP3 §§5.8, 5.10; Tashman (2000) |
| `projetar` | Projeção na série inteira + intervalo por bootstrap dos resíduos (semente fixa) | FPP3 §5.5; Thombs e Schucany (1990) |
| `Vetor`, `validar_vetor`, `fator_composto`, `aplicar_vetores`, `ler_vetores_csv` | Vetores D4: travas, composição multiplicativa, antes/depois, trilha, alertas | D4; L2 §§2.3, 5.5; FPP3 §6.7 |
| `grafico`, `cli` | Gráfico antes/depois e relatório em português | — |

**Dados fictícios:** `dados/previsao_ct3.csv` (CT-3), `dados/previsao_vetores_ct3.csv` (V1 e V2),
`dados/previsao_serie.csv` (36 meses, sazonal) e `dados/previsao_intermitente.csv` (24 meses, ADI = 3).
**Exemplo:** `python exemplos/4_previsao.py`. **Testes:** `tests/test_previsao.py`, que reproduzem o CT-3.

**Decisões de implementação.** (i) SES, Holt e Holt amortecido usam a **convenção do L2/motor TS** — grade e estado
inicial fixo —, e não o otimizador do statsmodels. Só assim o CT-3 bate até a 4.ª casa. (ii) O intervalo padrão é o
**bootstrap propagado pelas equações do modelo**, e não a fórmula normal. Isso explica a assimetria e o alargamento com β
alto. (iii) A seleção é por **origem móvel um passo à frente** (`--h-validacao` muda isso), com todos os candidatos nas
mesmas origens. O CT-3 canônico usa treino/teste 9 + 3 (`--validacao treino-teste --n-teste 3`).

---

## Referências

ANDERER, Matthias; LI, Feng. Hierarchical forecasting with a top-down alignment of independent-level forecasts.
**International Journal of Forecasting**, v. 38, n. 4, p. 1405-1414, 2022. DOI: 10.1016/j.ijforecast.2021.12.015.
Disponível em: https://doi.org/10.1016/j.ijforecast.2021.12.015. Acesso em: 5 nov. 2026.

BERGMEIR, Christoph; BENÍTEZ, José M. On the use of cross-validation for time series predictor evaluation.
**Information Sciences**, v. 191, p. 192-213, 2012. DOI: 10.1016/j.ins.2011.12.028. Disponível em:
https://doi.org/10.1016/j.ins.2011.12.028. Acesso em: 5 nov. 2026.

BERGMEIR, Christoph; HYNDMAN, Rob J.; BENÍTEZ, José M. Bagging exponential smoothing methods using STL decomposition and
Box–Cox transformation. **International Journal of Forecasting**, v. 32, n. 2, p. 303-312, 2016. DOI:
10.1016/j.ijforecast.2015.07.002. Disponível em: https://doi.org/10.1016/j.ijforecast.2015.07.002. Acesso em: 5 nov. 2026.

BOX, George E. P.; JENKINS, Gwilym M.; REINSEL, Gregory C.; LJUNG, Greta M. **Time series analysis**: forecasting and
control. 5. ed. Hoboken: John Wiley & Sons, 2015. 712 p. ISBN 978-1-118-67502-1.

CLEVELAND, Robert B.; CLEVELAND, William S.; McRAE, Jean E.; TERPENNING, Irma. STL: a seasonal-trend decomposition
procedure based on loess. **Journal of Official Statistics**, v. 6, n. 1, p. 3-73, 1990. Disponível em:
https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf.
Acesso em: 5 nov. 2026.

CROSTON, J. D. Forecasting and stock control for intermittent demands. **Operational Research Quarterly**, v. 23, n. 3,
p. 289-303, 1972. DOI: 10.1057/jors.1972.50. Disponível em: https://doi.org/10.1057/jors.1972.50. Acesso em: 5 nov. 2026.

FILDES, Robert; GOODWIN, Paul; LAWRENCE, Michael; NIKOLOPOULOS, Konstantinos. Effective forecasting and judgmental
adjustments: an empirical evaluation and strategies for improvement in supply-chain planning. **International Journal of
Forecasting**, v. 25, n. 1, p. 3-23, 2009. DOI: 10.1016/j.ijforecast.2008.11.010. Disponível em:
https://doi.org/10.1016/j.ijforecast.2008.11.010. Acesso em: 5 nov. 2026.

GARDNER JR., Everette S. Exponential smoothing: the state of the art. **Journal of Forecasting**, v. 4, n. 1, p. 1-28,
1985. DOI: 10.1002/for.3980040103. Disponível em: https://doi.org/10.1002/for.3980040103. Acesso em: 5 nov. 2026.

GARDNER JR., Everette S. Exponential smoothing: the state of the art — Part II. **International Journal of Forecasting**,
v. 22, n. 4, p. 637-666, 2006. DOI: 10.1016/j.ijforecast.2006.03.005. Disponível em:
https://doi.org/10.1016/j.ijforecast.2006.03.005. Acesso em: 5 nov. 2026.

GARDNER JR., Everette S.; McKENZIE, Ed. Forecasting trends in time series. **Management Science**, v. 31, n. 10,
p. 1237-1246, 1985. DOI: 10.1287/mnsc.31.10.1237. Disponível em: https://doi.org/10.1287/mnsc.31.10.1237. Acesso em:
5 nov. 2026.

HEWAMALAGE, Hansika; ACKERMANN, Klaus; BERGMEIR, Christoph. Forecast evaluation for data scientists: common pitfalls and
best practices. **Data Mining and Knowledge Discovery**, v. 37, n. 2, p. 788-832, 2023. DOI: 10.1007/s10618-022-00894-5.
Disponível em: https://doi.org/10.1007/s10618-022-00894-5. Acesso em: 5 nov. 2026.

HYNDMAN, Rob J.; AHMED, Roman A.; ATHANASOPOULOS, George; SHANG, Han Lin. Optimal combination forecasts for
hierarchical time series. **Computational Statistics & Data Analysis**, v. 55, n. 9, p. 2579-2589, 2011. DOI:
10.1016/j.csda.2011.03.006. Disponível em: https://doi.org/10.1016/j.csda.2011.03.006. Acesso em: 5 nov. 2026.

HYNDMAN, Rob J.; ATHANASOPOULOS, George. **Forecasting**: principles and practice. 3. ed. Melbourne: OTexts, 2021.
Disponível em: https://otexts.com/fpp3/. Acesso em: 5 nov. 2026.

HYNDMAN, Rob J.; KHANDAKAR, Yeasmin. Automatic time series forecasting: the forecast package for R. **Journal of
Statistical Software**, v. 27, n. 3, p. 1-22, 2008. DOI: 10.18637/jss.v027.i03. Disponível em:
https://doi.org/10.18637/jss.v027.i03. Acesso em: 5 nov. 2026.

HYNDMAN, Rob J.; KOEHLER, Anne B. Another look at measures of forecast accuracy. **International Journal of
Forecasting**, v. 22, n. 4, p. 679-688, 2006. DOI: 10.1016/j.ijforecast.2006.03.001. Disponível em:
https://doi.org/10.1016/j.ijforecast.2006.03.001. Acesso em: 5 nov. 2026.

HYNDMAN, Rob J.; KOEHLER, Anne B.; ORD, J. Keith; SNYDER, Ralph D. **Forecasting with exponential smoothing**: the state
space approach. Berlin: Springer, 2008. (Springer Series in Statistics). DOI: 10.1007/978-3-540-71918-2. Disponível em:
https://doi.org/10.1007/978-3-540-71918-2. Acesso em: 5 nov. 2026.

HYNDMAN, Rob J.; KOEHLER, Anne B.; SNYDER, Ralph D.; GROSE, Simone. A state space framework for automatic forecasting
using exponential smoothing methods. **International Journal of Forecasting**, v. 18, n. 3, p. 439-454, 2002. DOI:
10.1016/S0169-2070(01)00110-8. Disponível em: https://doi.org/10.1016/S0169-2070(01)00110-8. Acesso em: 5 nov. 2026.

KONING, Alex J.; FRANSES, Philip Hans; HIBON, Michèle; STEKLER, H. O. The M3 competition: statistical tests of the
results. **International Journal of Forecasting**, v. 21, n. 3, p. 397-409, 2005. DOI: 10.1016/j.ijforecast.2004.10.003.
Disponível em: https://doi.org/10.1016/j.ijforecast.2004.10.003. Acesso em: 5 nov. 2026.

LARSON, William D.; SINCLAIR, Tara M. Nowcasting unemployment insurance claims in the time of COVID-19.
**International Journal of Forecasting**, v. 38, n. 2, p. 635-647, 2022. DOI: 10.1016/j.ijforecast.2021.01.001.
Disponível em: https://doi.org/10.1016/j.ijforecast.2021.01.001. Acesso em: 5 nov. 2026.

LAWRENCE, Michael; GOODWIN, Paul; O'CONNOR, Marcus; ÖNKAL, Dilek. Judgmental forecasting: a review of progress over the
last 25 years. **International Journal of Forecasting**, v. 22, n. 3, p. 493-518, 2006. DOI:
10.1016/j.ijforecast.2006.03.007. Disponível em: https://doi.org/10.1016/j.ijforecast.2006.03.007. Acesso em: 5 nov. 2026.

MAKRIDAKIS, Spyros; HIBON, Michèle. The M3-Competition: results, conclusions and implications. **International Journal
of Forecasting**, v. 16, n. 4, p. 451-476, 2000. DOI: 10.1016/S0169-2070(00)00057-1. Disponível em:
https://doi.org/10.1016/S0169-2070(00)00057-1. Acesso em: 5 nov. 2026.

MAKRIDAKIS, Spyros; SPILIOTIS, Evangelos; ASSIMAKOPOULOS, Vassilios. The M4 Competition: 100,000 time series and 61
forecasting methods. **International Journal of Forecasting**, v. 36, n. 1, p. 54-74, 2020. DOI:
10.1016/j.ijforecast.2019.04.014. Disponível em: https://doi.org/10.1016/j.ijforecast.2019.04.014. Acesso em: 5 nov. 2026.

MAKRIDAKIS, Spyros; SPILIOTIS, Evangelos; ASSIMAKOPOULOS, Vassilios. M5 accuracy competition: results, findings, and
conclusions. **International Journal of Forecasting**, v. 38, n. 4, p. 1346-1364, 2022. DOI:
10.1016/j.ijforecast.2021.11.013. Disponível em: https://doi.org/10.1016/j.ijforecast.2021.11.013. Acesso em: 5 nov. 2026.

NIST; SEMATECH. **e-Handbook of statistical methods**: 6.4 Introduction to time series analysis. Gaithersburg: National
Institute of Standards and Technology, [2012]. Disponível em: https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm.
Acesso em: 5 nov. 2026.

PETROPOULOS, Fotios et al. Forecasting: theory and practice. **International Journal of Forecasting**, v. 38, n. 3,
p. 705-871, 2022. DOI: 10.1016/j.ijforecast.2021.11.001. Disponível em: https://doi.org/10.1016/j.ijforecast.2021.11.001.
Acesso em: 5 nov. 2026.

PRICE, Ilan; FOWKES, Jaroslav; HOPMAN, Daniel. Gaussian processes for unconstraining demand. **European Journal of
Operational Research**, v. 275, n. 2, p. 621-634, 2019. DOI: 10.1016/j.ejor.2018.11.065. Disponível em:
https://doi.org/10.1016/j.ejor.2018.11.065. Acesso em: 5 nov. 2026.

REZAEI, Mostafa; INGOLFSSON, Armann. Forecasting to support EMS tactical planning: what is important and what is not.
**Health Care Management Science**, v. 27, n. 4, p. 604-630, 2024. DOI: 10.1007/s10729-024-09690-7. Disponível em:
https://doi.org/10.1007/s10729-024-09690-7. Acesso em: 5 nov. 2026.

RODRIGUES, Adriana Maciel; CAVALCANTI, Marcos do Couto Bezerra; ARAÚJO DA SILVA, Raul; MACEDO, Valéria. Proposta de
metodologia de dimensionamento da força de trabalho para o TCE-RJ. **Revista do Tribunal de Contas do Estado do Rio de
Janeiro**, Rio de Janeiro, v. 2, n. 1, p. 72-83, 2021. DOI: 10.70690/xt46c612. Disponível em:
https://doi.org/10.70690/xt46c612. Acesso em: 5 nov. 2026.

ROSTAMI-TABAR, Bahman; ALI, Mohammad M.; HONG, Tao; HYNDMAN, Rob J.; PORTER, Michael D.; SYNTETOS, Aris. Forecasting
for social good. **International Journal of Forecasting**, v. 38, n. 3, p. 1245-1257, 2022. DOI:
10.1016/j.ijforecast.2021.02.010. Disponível em: https://doi.org/10.1016/j.ijforecast.2021.02.010. Acesso em: 5 nov. 2026.

ROSTAMI-TABAR, Bahman; HYNDMAN, Rob J. Hierarchical time series forecasting in emergency medical services. **Journal of
Service Research**, v. 28, n. 2, p. 278-295, 2025. DOI: 10.1177/10946705241232169. Disponível em:
https://doi.org/10.1177/10946705241232169. Acesso em: 5 nov. 2026.

SEABOLD, Skipper; PERKTOLD, Josef. Statsmodels: econometric and statistical modeling with Python. In: PYTHON IN SCIENCE
CONFERENCE, 9., 2010, Austin. **Proceedings** [...]. Austin: SciPy, 2010. p. 92-96. DOI: 10.25080/Majora-92bf1922-011.
Disponível em: https://doi.org/10.25080/Majora-92bf1922-011. Acesso em: 5 nov. 2026.

SYNTETOS, Aris A.; BOYLAN, John E. The accuracy of intermittent demand estimates. **International Journal of
Forecasting**, v. 21, n. 2, p. 303-314, 2005. DOI: 10.1016/j.ijforecast.2004.10.001. Disponível em:
https://doi.org/10.1016/j.ijforecast.2004.10.001. Acesso em: 5 nov. 2026.

SYNTETOS, A. A.; BOYLAN, J. E.; CROSTON, J. D. On the categorization of demand patterns. **Journal of the Operational
Research Society**, v. 56, n. 5, p. 495-503, 2005. DOI: 10.1057/palgrave.jors.2601841. Disponível em:
https://doi.org/10.1057/palgrave.jors.2601841. Acesso em: 5 nov. 2026.

TASHMAN, Leonard J. Out-of-sample tests of forecasting accuracy: an analysis and review. **International Journal of
Forecasting**, v. 16, n. 4, p. 437-450, 2000. DOI: 10.1016/S0169-2070(00)00065-0. Disponível em:
https://doi.org/10.1016/S0169-2070(00)00065-0. Acesso em: 5 nov. 2026.

TEUNTER, Ruud H.; SYNTETOS, Aris A.; BABAI, M. Zied. Intermittent demand: linking forecasting to inventory obsolescence.
**European Journal of Operational Research**, v. 214, n. 3, p. 606-615, 2011. DOI: 10.1016/j.ejor.2011.05.018.
Disponível em: https://doi.org/10.1016/j.ejor.2011.05.018. Acesso em: 5 nov. 2026.

THOMBS, Lori A.; SCHUCANY, William R. Bootstrap prediction intervals for autoregression. **Journal of the American
Statistical Association**, v. 85, n. 410, p. 486-492, 1990. DOI: 10.1080/01621459.1990.10476225. Disponível em:
https://doi.org/10.1080/01621459.1990.10476225. Acesso em: 5 nov. 2026.

WICKRAMASURIYA, Shanika L.; ATHANASOPOULOS, George; HYNDMAN, Rob J. Optimal forecast reconciliation for hierarchical and
grouped time series through trace minimization. **Journal of the American Statistical Association**, v. 114, n. 526,
p. 804-819, 2019. DOI: 10.1080/01621459.2018.1448825. Disponível em: https://doi.org/10.1080/01621459.2018.1448825.
Acesso em: 5 nov. 2026.
