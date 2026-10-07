# tmw-churn-case
TMW Churn Loyalty — Estudo Preditivo com Databricks e MLflow
Projeto de Data Science aplicado à análise de comportamento, retenção e propensão a churn no ecossistema TMW, integrando dados do TMW Loyalty System e da TMW Education Platform.
O estudo foi desenvolvido em Databricks, com uso de SQL, Python, scikit-learn e MLflow, cobrindo desde o entendimento do negócio e construção da Analytical Base Table (ABT) até comparação de modelos, calibração probabilística, definição de faixas de risco e scoring operacional.
---
1. Objetivo do projeto
O projeto busca responder duas perguntas principais:
Quais características comportamentais estão associadas ao churn?
Quais usuários apresentam maior probabilidade estimada de churn e devem ser priorizados em ações de retenção?
O problema foi estudado a partir da jornada de uso do Loyalty, utilizando o Education como fonte complementar para explicar comportamento.
---
2. Entendimento do negócio
Antes da modelagem, foram levantadas questões fundamentais com o cliente.
Definição oficial de churn
O negócio considera churn quando o usuário permanece:
> **28 dias consecutivos sem atividade.**
O que conta como atividade?
Qualquer transação de pontos registrada no Loyalty é considerada atividade, independentemente do valor da transação.
Sistema de pontos
As mecânicas gerais permanecem estáveis, embora novos produtos possam surgir ao longo do tempo.
A expectativa de negócio é:
> **quanto maior o engajamento, maior a quantidade de pontos esperada.**
Entretanto, diferentes ações possuem pesos muito diferentes. Por isso, o estudo mostrou que o volume bruto de pontos não deve ser interpretado isoladamente como medida perfeita de engajamento.
Relação Loyalty × Education
O Loyalty é o coração do projeto e é nele que atividade e churn são definidos.
O Education funciona como fonte complementar de informação, utilizada para explicar melhor o comportamento do usuário.
Picos de atividade
Picos de uso tendem a ocorrer em períodos de cursos ao vivo. O pico observado em agosto de 2025, por exemplo, esteve associado a um curso de SQL ao vivo.
Intervenção de churn
Não existe atualmente uma estratégia estruturada de prevenção ou recuperação de churn.
O negócio consegue se comunicar com o usuário:
durante sua interação nas lives;
quando utiliza a plataforma de cursos.
A janela de intervenção deve ocorrer antes de completar os 28 dias sem atividade.
Teste A/B
É possível realizar testes A/B de comunicação no chat ou em outros pontos de contato, embora a implementação seja considerada operacionalmente complexa.
---
3. Observação metodológica importante
Embora a definição oficial de churn informada pelo negócio seja 28 dias sem atividade, o notebook final utiliza um target experimental construído com a seguinte lógica:
```text
Janela de observação: D0–D30
Janela de resultado:  D31–D90
```
A variável resposta é:
```text
churn_90d = 1
→ nenhuma transação entre D31 e D90

churn_90d = 0
→ pelo menos uma transação entre D31 e D90
```
Portanto, este notebook deve ser interpretado como um estudo de propensão à inatividade/churn no horizonte D31–D90.
Uma evolução natural do projeto é reconstruir o target aderente à regra oficial de 28 dias consecutivos sem atividade.
---
4. Fontes de dados
O estudo utiliza duas bases principais.
TMW Loyalty System
Fonte principal do comportamento do usuário.
Exemplos de informações utilizadas:
transações;
pontos;
produtos;
datas de atividade;
frequência;
streaks;
recência;
diversidade de interações.
TMW Education Platform
Fonte complementar de comportamento.
Exemplos:
episódios concluídos;
cursos;
dias ativos;
atividade educacional.
A integração é realizada por meio da relação entre:
```text
usuarios_tmw.idUsuario
        ↕
usuarios_tmw.idTMWCliente
```
---
5. Validação da integração
A ponte entre Education e Loyalty apresentou boa consistência.
Resultados observados:
```text
2.414 usuários Education
2.413 encontrados no Loyalty
1 não encontrado
```
Na base Loyalty:
```text
5.742 clientes
│
├── 2.413 (42%)
│   └── Loyalty + Education
│
└── 3.329 (58%)
    └── somente Loyalty
```
Também foram realizadas verificações de:
cardinalidade;
duplicidades;
valores nulos;
cobertura entre as bases.
---
6. Principais aprendizados exploratórios
6.1 Pontos não contam toda a história
Inicialmente, a quantidade de pontos parecia uma candidata natural para representar engajamento.
Entretanto, diferentes ações possuem pesos muito distintos.
Isso significa que dois usuários podem acumular a mesma quantidade de pontos a partir de comportamentos completamente diferentes.
O estudo passou então a priorizar variáveis de:
recência;
frequência;
dias distintos de atividade;
persistência;
gaps entre interações;
streak;
tipo de interação.
O total de pontos permaneceu como variável complementar.
6.2 Ativação e churn são fenômenos diferentes
Foi necessário separar:
```text
Usuário nunca ativado
≠
Usuário que utilizou e depois abandonou
```
Essa distinção evitou misturar falha de onboarding com churn.
6.3 Formação de hábito
A análise mostrou que o principal problema não era apenas fazer o usuário começar a utilizar o Loyalty.
O desafio era transformar a primeira interação em:
```text
Ativação
   ↓
Retorno
   ↓
Regularidade
   ↓
Persistência
   ↓
Hábito
```
O principal insight do estudo foi:
> **O churn está mais associado à falta de continuidade do que ao volume bruto de pontos acumulados.**
---
7. Analytical Base Table — ABT
As tabelas transacionais foram transformadas em uma Analytical Base Table (ABT).
A granularidade utilizada foi:
```text
1 linha = 1 usuário
```
Cada linha contém:
```text
Comportamento passado
        +
Resultado futuro
```
A ABT histórica utilizada na modelagem contém:
```text
223 usuários
```
Distribuição do target:
```text
157 churners
66 retidos
Taxa de churn ≈ 70,4%
```
---
8. Feature Engineering
A ABT completa chegou a aproximadamente 31 colunas.
As principais dimensões de features foram:
Frequência
`tx_0_7`
`tx_8_30`
`tx_0_30`
Intensidade e tendência
`taxa_tx_d0_7`
`taxa_tx_d8_30`
`momentum_atividade`
`delta_taxa_atividade`
Regularidade e persistência
`dias_ativos_30d`
`dia_ultima_atividade_30d`
`dias_sem_atividade_ate_d30`
`maior_gap_dias`
`media_gap_dias`
Valor
`pontos_30d`
Diversidade
`produtos_distintos_30d`
Tipo de interação
`qtd_chatmessage_30d`
`qtd_lista_presenca_30d`
`qtd_streak_30d`
`fl_streak_30d`
Education
`episodios_education_30d`
`cursos_education_30d`
`dias_ativos_education_30d`
---
9. Validação da ABT
Antes da modelagem foram realizadas verificações de:
unicidade de usuário;
duplicidades;
valores nulos;
mínimos e máximos;
percentis;
heavy users;
consistência temporal;
comparação churn × retidos.
Não foi necessária imputação estatística tradicional.
Ausências semanticamente equivalentes a zero foram tratadas ainda na construção da ABT.
---
10. Análise Exploratória de Dados
A EDA incluiu:
estatísticas descritivas;
histogramas;
boxplots;
comparação churn × retidos;
matriz de correlação;
correlação individual com churn.
O principal padrão encontrado foi:
> **regularidade e persistência discriminam churn melhor do que pontos acumulados.**
Entre os sinais mais relevantes estavam:
`dias_sem_atividade_ate_d30`;
`dias_ativos_30d`;
`dia_ultima_atividade_30d`;
atividade inicial;
consumo de episódios no Education.
O volume bruto de pontos apresentou baixa capacidade explicativa isolada.
---
11. Pré-processamento
Algumas variáveis apresentaram forte assimetria devido à presença de heavy users.
Foi aplicada transformação:
```python
np.log1p()
```
em variáveis como:
`tx_0_7`
`tx_8_30`
`pontos_30d`
`qtd_streak_30d`
`episodios_education_30d`
A divisão treino/teste foi realizada com:
```python
test_size=0.25
random_state=42
stratify=y
```
---
12. Regressão Logística V1
A primeira regressão utilizou um pipeline:
```text
StandardScaler
      ↓
LogisticRegression
```
Resultados no holdout:
Métrica	Resultado
Accuracy	0.804
Precision	0.818
Recall	0.923
F1-score	0.867
ROC-AUC	0.796
---
13. Multicolinearidade e seleção de features
A análise de VIF mostrou redundância principalmente entre:
`dias_ativos_30d`
`log_tx_8_30`
Foi removida:
```text
log_tx_8_30
```
e mantida:
```text
dias_ativos_30d
```
por ser mais interpretável e representar diretamente recorrência e formação de hábito.
---
14. Regressão Logística V2 — modelo principal
Features utilizadas:
```python
features_modelo_v2 = [
    "dias_ativos_30d",
    "dias_sem_atividade_ate_d30",
    "produtos_distintos_30d",
    "dias_ativos_education_30d",
    "log_tx_0_7",
    "log_pontos_30d",
    "log_qtd_streak_30d",
    "log_episodios_education_30d"
]
```
Resultados:
Métrica	Resultado
Accuracy	0.804
Precision	0.818
Recall	0.923
F1-score	0.867
ROC-AUC holdout	0.798
ROC-AUC CV médio	0.794
ROC-AUC CV std	0.058
A Regressão Logística V2 foi escolhida como modelo principal por combinar:
melhor desempenho generalizado em validação cruzada;
alto recall;
interpretabilidade;
estabilidade;
boa calibração.
---
15. Outros modelos testados
Além da regressão, foram avaliados outros algoritmos.
Modelo	ROC-AUC teste	ROC-AUC CV	CV std	Accuracy	Precision	Recall	F1
Logistic Regression V2	0.798	0.794	0.058	0.804	0.818	0.923	0.867
Random Forest	0.801	0.778	0.049	0.821	0.822	0.949	0.881
Naive Bayes	0.744	0.775	0.027	0.804	0.804	0.949	0.871
Gradient Boosting	0.729	0.723	0.035	0.786	0.814	0.897	0.854
Decision Tree	0.703	0.722	0.067	0.804	0.818	0.923	0.867
SVM RBF	0.701	0.715	0.092	0.821	0.809	0.974	0.884
Interpretação
O Random Forest apresentou o melhor ROC-AUC no holdout, porém a Logistic Regression V2 apresentou o melhor ROC-AUC médio em validação cruzada.
O Naive Bayes apresentou boa estabilidade, com menor desvio padrão entre os folds.
O SVM RBF apresentou recall muito alto, porém com maior variabilidade e menor ROC-AUC.
O modelo final escolhido foi:
> **Logistic Regression V2**
---
16. Decision Tree como modelo explicativo
A árvore de decisão foi mantida como ferramenta de interpretação.
Configuração:
```python
DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)
```
A árvore destacou principalmente:
tempo sem atividade;
dias distintos de atividade;
consumo de episódios no Education.
Um dos principais cortes ocorreu em aproximadamente:
```text
15,5 dias sem atividade até D30
```
A árvore foi utilizada como modelo de apoio à comunicação das regras de negócio, enquanto a regressão permaneceu como modelo oficial de previsão.
---
17. Variáveis mais importantes
Os coeficientes da Logistic Regression V2 foram:
Feature	Coeficiente	Odds Ratio	Interpretação
`dias_sem_atividade_ate_d30`	+0.812	2.25	forte aumento do risco
`dias_ativos_30d`	-0.718	0.49	forte associação com retenção
`produtos_distintos_30d`	+0.662	1.94	associação positiva condicional
`log_episodios_education_30d`	-0.460	0.63	maior consumo Education reduz risco
`log_tx_0_7`	-0.390	0.68	maior atividade inicial reduz risco
`log_qtd_streak_30d`	+0.129	1.14	efeito pequeno
`log_pontos_30d`	-0.067	0.93	efeito independente pequeno
`dias_ativos_education_30d`	-0.038	0.96	efeito pequeno
Como o modelo utiliza `StandardScaler`, os coeficientes devem ser interpretados na escala padronizada.
Importante:
> **associação não implica causalidade.**
Em especial, o coeficiente positivo de `produtos_distintos_30d` não deve ser interpretado como evidência de que diversidade de produtos causa churn.
---
18. Probabilidades Out-of-Fold
Para evitar probabilidades in-sample na avaliação dos usuários históricos, foram utilizadas probabilidades:
```text
Out-of-Fold (OOF)
```
geradas com:
```python
cross_val_predict(...)
```
Assim, cada usuário recebe uma probabilidade produzida por um modelo que não foi treinado naquele próprio usuário.
---
19. Calibração probabilística
Além da capacidade de discriminar churners de retidos, foi avaliada a qualidade das probabilidades estimadas.
Resultados:
```text
Brier Score = 0.149
Brier baseline ≈ 0.208
Brier Skill Score ≈ 28.5%
ECE ≈ 0.043
```
Interpretação:
o modelo reduz aproximadamente 28,5% do erro probabilístico em relação a uma referência que atribui a taxa média de churn a todos;
o erro médio ponderado de calibração ficou em aproximadamente 4,3 pontos percentuais.
Por isso, o `predict_proba()` foi considerado suficientemente calibrado para ser apresentado como:
> **probabilidade estimada de churn**
e não apenas como score ordinal.
---
20. Threshold e faixas de risco
Foram testados thresholds entre:
```text
0.10 e 0.90
```
avaliando:
precision;
recall;
F1;
especificidade;
NPV;
percentual de usuários alertados.
Também foram construídas faixas operacionais de risco.
No scoring final foram utilizadas:
```text
BAIXO   < 30%
MÉDIO   30% a < 85%
ALTO    >= 85%
```
---
21. Simulação econômica
O notebook também contém uma análise exploratória de custo-benefício.
Foram simulados cenários hipotéticos considerando:
custo de intervenção;
custo de churn perdido;
eficácia da intervenção.
Exemplos:
```text
Intervenção barata:
custo intervenção = 1
custo churn = 10
eficácia = 20%

Intermediário:
custo intervenção = 1
custo churn = 5
eficácia = 30%

Intervenção cara:
custo intervenção = 3
custo churn = 5
eficácia = 40%
```
Também foi calculado o threshold econômico teórico:
```text
Cenário barato       → 0.50
Cenário intermediário → 0.67
Cenário caro          → 1.50 → economicamente inviável
```
Esses valores são hipóteses de simulação, pois o negócio não possui ainda histórico real de custo e eficácia de ações de retenção.
---
22. Scoring operacional
Após a validação, o modelo foi aplicado a uma população operacional.
Critério:
```text
usuários que já completaram D30
e ainda não completaram D90
```
No momento do estudo:
```text
32 usuários elegíveis
19 usuários de alto risco (>= 85%)
```
Maior probabilidade estimada:
```text
96,85%
```
Top 5 probabilidades estimadas:
```text
96,85%
94,72%
94,02%
92,48%
91,74%
```
Por privacidade e segurança, identificadores individuais não são publicados neste repositório.
---
23. MLflow
O MLflow foi utilizado para rastrear os experimentos e tornar a modelagem reproduzível.
Foram registrados:
parâmetros;
features;
modelos;
métricas;
curvas ROC;
Precision-Recall;
matriz de confusão;
VIF;
coeficientes;
feature importance;
validação cruzada;
calibração;
Brier Score;
scoring operacional.
Experimento utilizado:
```text
/Shared/TMW_Churn_Loyalty
```
---
24. Tecnologias utilizadas
Databricks
Spark SQL
Python
pandas
NumPy
scikit-learn
matplotlib
MLflow
Modelos avaliados:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
SVM RBF
Gaussian Naive Bayes
---
25. Estrutura sugerida do repositório
```text
tmw-churn-loyalty/
│
├── README.md
├── notebooks/
│   └── ml_flow_formatado.ipynb
│
├── docs/
│   └── presentation/
│
├── images/
│   └── ...
│
├── requirements.txt
└── .gitignore
```
O notebook pode permanecer como principal artefato analítico.
Em uma evolução posterior, as rotinas de feature engineering, treinamento e scoring podem ser modularizadas em scripts Python.
---
26. Como executar
Pré-requisitos
O notebook foi desenvolvido em Databricks e depende de acesso às tabelas dos schemas:
```text
tmw_loyalty
tmw_education
```
Fluxo de execução
Importar o notebook no Databricks.
Garantir acesso aos schemas necessários.
Executar as validações de integração.
Executar a análise exploratória.
Construir a ABT histórica.
Validar a ABT.
Carregar os dados para Python.
Executar pré-processamento.
Treinar os modelos.
Executar validação cruzada.
Comparar os algoritmos.
Avaliar calibração.
Definir thresholds e faixas de risco.
Construir a população de scoring.
Aplicar o modelo final.
Registrar experimentos e artefatos no MLflow.
---
27. Principais conclusões
O estudo mostrou que:
> **o churn está principalmente associado à baixa continuidade durante os primeiros 30 dias.**
Usuários de maior risco tendem a:
deixar de interagir cedo;
apresentar poucos dias distintos de atividade;
permanecer longos períodos sem interação;
apresentar menor atividade inicial.
Usuários com maior tendência à retenção apresentam:
maior recorrência;
maior persistência;
atividade mais próxima do fim do primeiro mês;
maior atividade inicial;
maior consumo de episódios no Education.
O volume bruto de pontos, isoladamente, apresentou baixa capacidade explicativa.
---
28. Limitações
Principais limitações do estudo:
amostra de modelagem relativamente pequena: 223 usuários;
target experimental D31–D90 diferente da definição operacional oficial de 28 dias;
estudo observacional;
associação não implica causalidade;
ausência de histórico real de intervenção;
custos de churn e eficácia de retenção ainda não foram medidos;
thresholds econômicos são cenários hipotéticos;
eventos como cursos ao vivo podem afetar o comportamento agregado.
---
29. Próximos passos
Evoluções recomendadas:
reconstruir o target aderente à definição oficial de 28 dias sem atividade;
ampliar a população para todos os usuários ativados no Loyalty;
manter o Education como fonte complementar;
incorporar calendário de cursos ao vivo e campanhas;
recalcular scores de risco de forma recorrente;
disparar intervenção antes do churn;
implementar testes A/B;
medir eficácia real das comunicações;
estimar custo por churn evitado;
evoluir o modelo somente após aumento da base histórica.
---
Autor
André Guimarães
Projeto desenvolvido como estudo prático de Data Science aplicado a comportamento, retenção e churn, com foco em integração de dados, feature engineering, modelagem interpretável, MLflow e aplicação operacional.
