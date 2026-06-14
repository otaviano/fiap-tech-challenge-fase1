# Tech Challenge — Fase 1
## Sistema Inteligente de Suporte ao Diagnóstico Oncológico

**Programa:** FIAP Postech — IA para Devs  
**Fase:** 1 — Machine Learning aplicado à Saúde  
**Aluno:** Otaviano Montes Zibetti | Felipe Squarizi
**Email:** otavianomontes@gmail.com | fesqua@yahoo.com.br 
**Data:** Junho de 2026

---

## Links

| Recurso | URL |
|---------|-----|
| Repositório GitHub | https://github.com/otaviano/fiap-tech-challenge-fase1 |
| Notebook principal | https://github.com/otaviano/fiap-tech-challenge-fase1/blob/main/breast_cancer_ml.ipynb |
| Dataset (UCI / sklearn) | https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29 |
| Dataset (Kaggle) | https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data |

---

## Sumário

1. [Contexto e Problema](#1-contexto-e-problema)
2. [Dataset](#2-dataset)
3. [Exploração dos Dados (EDA)](#3-exploração-dos-dados-eda)
4. [Pré-processamento](#4-pré-processamento)
5. [Modelagem](#5-modelagem)
6. [Treinamento e Avaliação](#6-treinamento-e-avaliação)
7. [Explicabilidade com SHAP](#7-explicabilidade-com-shap)
8. [Discussão Crítica](#8-discussão-crítica)
9. [Instruções de Execução](#9-instruções-de-execução)
10. [Roteiro do Vídeo](#10-roteiro-do-vídeo)

---

## 1. Contexto e Problema

Um hospital universitário enfrenta crescente volume de exames e precisas de soluções de IA que acelerem a triagem e apoiem as decisões médicas. Nesta fase, o objetivo é construir a base do sistema focado em **Machine Learning** para classificação automática de resultados de exames.

### Problema escolhido

**Diagnóstico de câncer de mama — classificação maligno vs. benigno**

O câncer de mama é o tipo de câncer mais frequente entre mulheres no Brasil e no mundo. O diagnóstico precoce é o principal fator que eleva as taxas de sobrevivência. Biópsias por agulha geram características numéricas dos núcleos celulares que podem ser analisadas por algoritmos de Machine Learning — permitindo apoio ao diagnóstico do patologista.

> **Por que Recall é a métrica principal?**  
> Em diagnóstico oncológico, um **falso negativo** (classificar maligno como benigno e liberar o paciente sem tratamento) tem custo humano muito maior do que um falso positivo (acionar novas investigações em tumor benigno). Portanto, a estratégia de avaliação prioriza **maximizar o Recall da classe maligna**, mantendo Precision e F1-Score em níveis clinicamente aceitáveis.

---

## 2. Dataset

**Nome:** Wisconsin Breast Cancer Diagnostic Dataset (WBCD)  
**Fonte original:** UCI Machine Learning Repository  
**Disponível via:** `sklearn.datasets.load_breast_cancer()`

### Características gerais

| Atributo | Valor |
|----------|-------|
| Total de amostras | 569 |
| Features numéricas | 30 |
| Classes | 2 (Maligno / Benigno) |
| Valores nulos | 0 (nenhum) |
| Distribuição | Maligno: 212 (37,3%) / Benigno: 357 (62,7%) |

### Origem das features

Cada tumor foi digitalmente fotografado e, a partir da imagem, foram calculadas 10 características dos núcleos celulares:

| Característica | Descrição |
|----------------|-----------|
| radius | Média das distâncias do centro à periferia |
| texture | Desvio padrão dos valores de escala de cinza |
| perimeter | Perímetro do núcleo |
| area | Área do núcleo |
| smoothness | Variação local nos comprimentos dos raios |
| compactness | (perímetro² / área) − 1,0 |
| concavity | Gravidade das porções côncavas do contorno |
| concave points | Número de porções côncavas do contorno |
| symmetry | Simetria do núcleo |
| fractal dimension | "Aproximação da linha de costa" − 1 |

Cada uma das 10 características é representada em **3 versões**: média (`mean`), erro padrão (`error`) e pior valor (`worst`) — totalizando **30 features**.

---

## 3. Exploração dos Dados (EDA)

### 3.1 Distribuição das classes

O dataset apresenta **leve desbalanceamento** (37% maligno / 63% benigno), que foi tratado com split estratificado para garantir proporções equivalentes em treino e teste.

```
Maligno (0):  212 amostras — 37,3%
Benigno (1):  357 amostras — 62,7%
```

### 3.2 Estatísticas descritivas (features mean)

As features de raio, perímetro e área apresentam grande variação de escala (ex.: `mean area` varia de 143 a 2.501 mm²) em relação a features como `mean smoothness` (0,05 a 0,16). Isso justifica o uso de **StandardScaler** antes de modelos sensíveis a escala (SVM, KNN, Regressão Logística).

### 3.3 Análise de correlação

As features mais correlacionadas com o target (correlação absoluta) foram:

| Rank | Feature | |Correlação com target| |
|------|---------|----------------------|
| 1 | worst concave points | 0,794 |
| 2 | worst perimeter | 0,783 |
| 3 | mean concave points | 0,777 |
| 4 | worst radius | 0,776 |
| 5 | mean perimeter | 0,743 |

> Features do grupo `worst` (pior valor observado) têm maior poder discriminativo que as médias — indicando que o comportamento no pior caso é mais diagnóstico que a tendência central.

**Multicolinearidade observada:** features de raio, perímetro e área são altamente correlacionadas entre si (r > 0,95), o que é esperado geometricamente. Isso não prejudica modelos baseados em árvores (Random Forest, Gradient Boosting), mas pode afetar a interpretabilidade de modelos lineares.

---

## 4. Pré-processamento

### 4.1 Pipeline de pré-processamento

O pré-processamento foi encapsulado em `sklearn.Pipeline`, garantindo que transformações sejam aplicadas corretamente dentro do cross-validation (sem data leakage):

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', <algoritmo>)
])
```

### 4.2 Divisão treino/teste

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% treino, 20% teste
    random_state=42,
    stratify=y           # mantém proporção das classes
)
```

| Conjunto | Amostras | Maligno | Benigno |
|----------|----------|---------|---------|
| Treino | 455 | 170 | 285 |
| Teste | 114 | 42 | 72 |

### 4.3 Normalização

`StandardScaler` foi aplicado apenas às features (nunca ao target), com `fit` exclusivo no conjunto de treino e `transform` aplicado ao teste — evitando vazamento de informação:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)         # apenas transform
```

**Antes do scaling:** `mean area` varia de 143 a 2501 (escala grande)  
**Após o scaling:** todas as features centradas em 0 com desvio padrão ≈ 1

### 4.4 Variáveis categóricas

O dataset não contém variáveis categóricas — todas as 30 features são numéricas contínuas. Nenhuma codificação (one-hot, label encoding) foi necessária.

### 4.5 Valores ausentes

```
✅ Nenhum valor nulo encontrado no dataset.
```

---

## 5. Modelagem

### 5.1 Modelos selecionados

Foram treinados e comparados **6 algoritmos de classificação**:

| Modelo | Justificativa |
|--------|--------------|
| **Logistic Regression** | Baseline linear; interpretável e eficiente; bom ponto de partida para problemas de classificação binária |
| **Decision Tree** | Modelo explicável (regras de decisão); útil para entender o problema, mesmo que não seja o mais preciso |
| **Random Forest** | Ensemble que reduz overfitting via bagging; robusto e com feature importance nativa; escolhido para SHAP |
| **Gradient Boosting** | Ensemble sequencial com alto poder preditivo; bom em datasets tabulares estruturados |
| **SVM** | Alta performance em espaços de alta dimensão; robusto a outliers; selecionado como modelo final por melhor Recall |
| **KNN** | Modelo baseado em distância; simples e sem suposições paraméricas; útil como comparativo |

### 5.2 Validação cruzada

Foi utilizada **StratifiedKFold com 5 folds** para garantir:
- Avaliação robusta sem overfitting ao conjunto de teste
- Manutenção da proporção das classes em cada fold
- Estimativas de variância nos resultados (desvio padrão)

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 6. Treinamento e Avaliação

### 6.1 Resultados do Cross-Validation (5-Fold)

> Recall e F1-Score referem-se **à classe MALIGNA (pos_label=0)** — o erro crítico a minimizar.

| Modelo | Recall Maligno | F1 Maligno | Accuracy | AUC |
|--------|---------------|-----------|----------|-----|
| **SVM** | **0,958 ± 0,037** | **0,969** | **0,977** | **0,995** |
| Logistic Regression | 0,944 ± 0,052 | 0,963 | 0,974 | 0,995 |
| Random Forest | 0,939 ± 0,050 | 0,941 | 0,956 | 0,989 |
| KNN | 0,925 ± 0,046 | 0,949 | 0,963 | 0,985 |
| Gradient Boosting | 0,906 ± 0,066 | 0,929 | 0,949 | 0,993 |
| Decision Tree | 0,878 ± 0,082 | 0,899 | 0,928 | 0,907 |

**Modelo selecionado: SVM** — maior Recall médio (0,958) e menor variância (± 0,037), indicando estabilidade e confiabilidade.

### 6.2 Avaliação do SVM no Test Set

```
Classification Report — SVM
═══════════════════════════════════════════════════
              precision    recall  f1-score   support

     Maligno       0.98      0.98      0.98        42
     Benigno       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

### 6.3 Matriz de Confusão (SVM — Test Set)

```
                  Predito
                Maligno  Benigno
Real  Maligno |   41   |    1   |   ← 1 Falso Negativo (crítico)
      Benigno |    1   |   71   |   ← 1 Falso Positivo
```

| Métrica no Test Set | Valor |
|---------------------|-------|
| Accuracy | 98,25% |
| Recall (Maligno) | 97,62% |
| Precision (Maligno) | 97,62% |
| F1-Score (Maligno) | 97,62% |
| AUC-ROC | 0,9950 |
| Falsos Negativos | **1** |
| Falsos Positivos | 1 |

> **1 falso negativo em 42 casos malignos = 97,6% de sensibilidade** — clinicamente expressivo como ferramenta de triagem.

### 6.4 Curva ROC

O SVM atingiu **AUC = 0,9950**, próximo de 1,0 (classificação perfeita). A curva ROC demonstra que o modelo é capaz de discriminar tumores malignos de benignos em quase todos os pontos de corte de probabilidade.

Ranking AUC no test set:
1. SVM: 0,9950
2. Logistic Regression: ~0,995
3. Gradient Boosting: ~0,993
4. Random Forest: 0,9937
5. KNN: ~0,985
6. Decision Tree: ~0,907

---

## 7. Explicabilidade com SHAP

Para garantir a interpretabilidade clínica, foi aplicado **SHAP (SHapley Additive exPlanations)** usando o `TreeExplainer` sobre o **Random Forest** (modelos tree-based são mais eficientes com SHAP).

SHAP quantifica a contribuição marginal de cada feature para cada predição individual, baseado na teoria de valores de Shapley de jogos cooperativos.

### 7.1 Importância Global das Features (Top 10)

| Rank | Feature | Mean |SHAP| |
|------|---------|------|
| 1 | worst area | 0,312 |
| 2 | worst concave points | 0,298 |
| 3 | worst radius | 0,241 |
| 4 | mean concave points | 0,198 |
| 5 | worst perimeter | 0,187 |
| 6 | mean area | 0,145 |
| 7 | worst concavity | 0,132 |
| 8 | mean radius | 0,121 |
| 9 | mean perimeter | 0,118 |
| 10 | area error | 0,089 |

### 7.2 Interpretação Clínica

O SHAP Beeswarm plot revela a **direção** do impacto de cada feature:

- **Valores altos de `worst area`** (vermelho) → empurram a predição para **Maligno**
- **Valores baixos de `worst concave points`** (azul) → empurram para **Benigno**
- Features do grupo `worst` consistentemente dominam — confirmando que o comportamento no pior núcleo celular é mais informativo que as médias

### 7.3 Convergência SHAP vs. Feature Importance nativa

Tanto o SHAP quanto o feature importance nativo do Random Forest concordam nas features mais relevantes (`worst area`, `worst concave points`, `worst radius`), o que aumenta a confiança na interpretação. A diferença é que o SHAP revela **direção e magnitude por amostra**, enquanto o importance nativo fornece apenas uma importância global agregada.

### 7.4 Explicação individual (Force Plot)

Para a primeira amostra do test set (Maligno, classificado corretamente como Maligno):
- Probabilidade de benigno: 0,000 (máxima confiança)
- Features que mais contribuíram para Maligno: `worst area` alto, `worst concave points` alto, `worst radius` alto

---

## 8. Discussão Crítica

### 8.1 O modelo pode ser utilizado na prática?

**Sim, como ferramenta de apoio à decisão — não como substituto ao médico.**

Com **97,6% de Recall** (sensibilidade) e **AUC 0,995**, o modelo tem performance comparável a sistemas de CAD (Computer-Aided Detection) relatados na literatura. Resultados semelhantes são encontrados em artigos publicados utilizando o mesmo dataset (UCI Wisconsin).

### 8.2 Como integrá-lo clinicamente?

**Cenário recomendado — triagem assistida:**

1. Patologista realiza biópsia por agulha → sistema extrai as 30 características celulares
2. Modelo classifica o tumor como provável maligno/benigno com probabilidade associada
3. **Casos de alta probabilidade maligna → prioridade na fila de análise do médico**
4. **Médico sempre tem a palavra final** — o modelo é um alerta, não um diagnóstico

### 8.3 Limitações

| Limitação | Impacto | Mitigação |
|-----------|---------|-----------|
| Dataset pequeno (569 amostras) | Risco de resultados inflados | Validar em datasets maiores e multicêntricos |
| Dados de único centro (Wisconsin, 1995) | Viés demográfico e temporal | Retreinar com dados contemporâneos e diversificados |
| Features extraídas manualmente de imagens digitalizadas | Dependência de software de análise de imagem | Integrar com pipeline de visão computacional |
| Sem validação prospectiva | Performance em dados do mundo real pode diferir | Estudo piloto hospitalar necessário |
| Desbalanceamento leve | Pode subestimar Recall em produção | Investigar SMOTE e threshold tuning |

### 8.4 Próximos passos técnicos

| Técnica | Benefício esperado |
|---------|-------------------|
| GridSearchCV para SVM (C, gamma, kernel) | Otimização de hiperparâmetros |
| Threshold tuning (ex.: 0,3 em vez de 0,5) | Aumentar Recall à custa de Precision |
| SMOTE | Balancear classes para reduzir viés |
| PCA para visualização | Entender a separabilidade das classes em 2D |
| Ensemble SVM + LR | Combinar os dois melhores modelos |
| MLflow | Rastreamento de experimentos e versionamento |
| FastAPI + Docker | Deploy como microserviço hospitalar |

### 8.5 Aspectos éticos

O uso de IA em diagnóstico médico exige:
- **Transparência:** o modelo deve ser explicável ao médico (SHAP atende a isso)
- **Responsabilidade:** falsos negativos podem ter consequências legais — documentar limitações
- **Consentimento:** pacientes devem ser informados do uso de IA na análise de seus exames
- **Supervisão contínua:** monitorar drift de performance ao longo do tempo

---

## 9. Instruções de Execução

### 9.1 Requisitos

- Python 3.11+
- pip ou conda

### 9.2 Instalação local

```bash
# Clonar o repositório
git clone https://github.com/otaviano/fiap-pos-ia.git
cd fiap-pos-ia

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar o notebook
jupyter notebook breast_cancer_ml.ipynb
```

### 9.3 requirements.txt

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.43.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

### 9.4 Execução via Docker

```bash
# Build
docker build -t breast-cancer-ml .

# Executar (Jupyter Lab na porta 8888)
docker run -p 8888:8888 breast-cancer-ml
```

Acesse `http://localhost:8888` no navegador.

### 9.5 Sobre o dataset

O dataset **Wisconsin Breast Cancer** é carregado automaticamente via `sklearn.datasets.load_breast_cancer()` — **não é necessário fazer download manual**. Ele está embutido na biblioteca scikit-learn.

Para referência, os mesmos dados estão disponíveis em:
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## 10. Roteiro do Vídeo

> **Duração estimada:** 12–15 minutos  
> **Formato:** screenshare do Jupyter Notebook + câmera (rosto opcional)  
> **Plataforma:** YouTube (não listado) ou Vimeo

---

### Bloco 1 — Introdução (0:00–1:30)

**O que mostrar:** tela do notebook fechado + slide de abertura

**Script:**
> "Olá! Neste vídeo apresento meu Tech Challenge da Fase 1 do programa IA para Devs da FIAP Postech. O desafio é construir um sistema de suporte ao diagnóstico médico usando Machine Learning.
>
> Escolhi o problema de **diagnóstico de câncer de mama** — classificar tumores como malignos ou benignos a partir de características celulares extraídas de biópsias.
>
> O dataset utilizado é o **Wisconsin Breast Cancer**, com 569 amostras e 30 features numéricas. Vou mostrar todo o pipeline: exploração dos dados, pré-processamento, comparação de 6 modelos e explicabilidade com SHAP."

---

### Bloco 2 — Dataset e EDA (1:30–4:00)

**O que mostrar:** Célula 6, 7, 11, 12, 13, 14 do notebook

**Script:**
> "Começo carregando o dataset. São 569 amostras, sem valores nulos, com 30 features numéricas e 2 classes: Maligno e Benigno.
>
> [Mostrar gráfico de barras] O dataset tem leve desbalanceamento: 37% malignos e 63% benignos. Importante para a estratégia de avaliação.
>
> [Mostrar distribuições] As features do grupo 'mean' mostram separação clara entre as classes — tumores malignos tendem a ter raio, perímetro e área maiores.
>
> [Mostrar heatmap de correlação] Algumas features são altamente correlacionadas entre si — raio, perímetro e área são geometricamente relacionados. Isso é esperado e não representa problema para modelos baseados em árvores.
>
> [Mostrar ranking] As features mais correlacionadas com o diagnóstico são do grupo 'worst' — o comportamento no pior núcleo celular é mais informativo que as médias."

---

### Bloco 3 — Pré-processamento (4:00–5:30)

**O que mostrar:** Células 16 e 17

**Script:**
> "O pré-processamento tem dois componentes principais.
>
> Primeiro, o **split estratificado**: 80% para treino e 20% para teste, garantindo que a proporção de malignos e benignos seja mantida em ambos os conjuntos.
>
> Segundo, o **StandardScaler**: normaliza as features para média zero e desvio padrão um. Isso é essencial para algoritmos sensíveis a escala como SVM e KNN — sem isso, a feature 'mean area' dominaria as distâncias.
>
> [Mostrar boxplot antes/depois] O efeito é visível: features com escalas muito diferentes passam a ter distribuições comparáveis.
>
> Todo o scaling é encapsulado em um Pipeline do sklearn, o que evita data leakage no cross-validation — o scaler é ajustado apenas nos dados de treino de cada fold."

---

### Bloco 4 — Treinamento e Comparação de Modelos (5:30–8:30)

**O que mostrar:** Células 19 e 20

**Script:**
> "Treinei e comparei 6 modelos: Regressão Logística, Árvore de Decisão, Random Forest, Gradient Boosting, SVM e KNN.
>
> A avaliação usa **StratifiedKFold com 5 folds** para estimativas robustas. A métrica principal é o **Recall da classe Maligna** — porque um falso negativo, classificar um tumor maligno como benigno, é o erro mais grave em oncologia.
>
> [Mostrar tabela de resultados] O SVM lidera com **Recall de 0,958** e o menor desvio padrão (± 0,037), indicando não apenas alta performance mas também consistência.
>
> [Mostrar gráfico comparativo] Em AUC, tanto SVM quanto Regressão Logística atingem 0,995 — excelente capacidade discriminativa. A Árvore de Decisão simples fica para trás, com AUC de apenas 0,907.
>
> Pela combinação de maior Recall e menor variância, o **SVM foi selecionado como modelo final**."

---

### Bloco 5 — Avaliação no Test Set (8:30–10:30)

**O que mostrar:** Células 23, 24 e 25

**Script:**
> "Avaliando o SVM no test set — dados que o modelo nunca viu durante o treinamento.
>
> [Mostrar classification report] **98% de accuracy, 98% de Recall e 99% de F1** para ambas as classes. Resultado sólido e consistente com o cross-validation.
>
> [Mostrar matriz de confusão] Em 114 amostras de teste, o modelo cometeu apenas **1 falso negativo**: um tumor maligno classificado como benigno. Também apenas 1 falso positivo.
>
> 1 falso negativo em 42 casos malignos equivale a **97,6% de sensibilidade** — performance clinicamente relevante para uma ferramenta de triagem.
>
> [Mostrar curva ROC] O AUC de 0,9950 confirma que o modelo discrimina excelentemente os dois tipos de tumor em qualquer ponto de corte de probabilidade."

---

### Bloco 6 — Explicabilidade com SHAP (10:30–12:30)

**O que mostrar:** Células 29, 30, 31 e 32

**Script:**
> "Para que o modelo possa ser utilizado em contexto clínico, precisamos de **explicabilidade** — o médico não pode usar uma 'caixa preta'.
>
> Usei o SHAP com TreeExplainer sobre o Random Forest. O SHAP calcula a contribuição de cada feature para cada predição individual, baseado em teoria de jogos.
>
> [Mostrar bar chart SHAP] As features mais importantes globalmente são: **worst area, worst concave points e worst radius**. Isso faz sentido clínico: tumores malignos tendem a ter células maiores e contornos mais irregulares.
>
> [Mostrar beeswarm] O beeswarm plot revela a direção: valores altos de 'worst area' (vermelho) empurram fortemente para Maligno. Valores baixos de 'worst concave points' (azul) indicam Benigno.
>
> [Mostrar force plot] Para esta amostra maligna específica, o modelo teve probabilidade zero de ser benigno. As features que mais contribuíram para esse diagnóstico foram área, pontos côncavos e raio — todas com valores altos.
>
> Essa explicabilidade permite que o médico valide ou questione a predição do modelo."

---

### Bloco 7 — Conclusão e Próximos Passos (12:30–14:00)

**O que mostrar:** Célula 34 + slide de encerramento

**Script:**
> "Para resumir: construí um pipeline completo de Machine Learning para diagnóstico de câncer de mama, desde a exploração dos dados até a explicabilidade com SHAP.
>
> O SVM atingiu **98% de accuracy e AUC de 0,995** no test set, com apenas 1 falso negativo em 42 casos malignos.
>
> **O modelo pode ser utilizado como ferramenta de triagem**, priorizando casos de alta probabilidade maligna na fila de análise médica. O médico sempre tem a palavra final no diagnóstico.
>
> Próximos passos: otimização de hiperparâmetros com GridSearchCV, ajuste de threshold para maximizar ainda mais o Recall, e exploração da parte extra de visão computacional com CNN.
>
> O código completo está disponível no GitHub: github.com/otaviano/fiap-pos-ia. Obrigado!"

---

*Tech Challenge Fase 1 — FIAP Postech IA para Devs — Junho de 2026*
