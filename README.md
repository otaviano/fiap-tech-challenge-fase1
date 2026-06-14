# Tech Challenge — Fase 1: Diagnóstico de Câncer de Mama com Machine Learning

> **FIAP Postech — IA para Devs**  
> Fase 1 — Machine Learning aplicado à Saúde

---

## Sobre o projeto

Este projeto implementa um **sistema de suporte ao diagnóstico oncológico** usando Machine Learning para classificar tumores de mama como malignos ou benignos, com base em características celulares extraídas de biópsias (dataset Wisconsin Breast Cancer).

O modelo atingiu **98% de accuracy** e **AUC-ROC de 0,9950** no conjunto de teste, com apenas **1 falso negativo** em 42 casos malignos.

---

## Estrutura do repositório

```
.
├── breast_cancer_ml.ipynb    # Notebook principal (pipeline completo)
├── RELATORIO_TECH_CHALLENGE.md  # Relatório técnico completo
├── requirements.txt          # Dependências Python
├── Dockerfile                # Container para execução isolada
└── README.md
```

---

## Dataset

**Wisconsin Breast Cancer Diagnostic Dataset**  
569 amostras | 30 features numéricas | 2 classes (Maligno / Benigno)

O dataset é carregado automaticamente via `sklearn.datasets.load_breast_cancer()` — **não é necessário download manual**.

Fontes externas para referência:
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## Pipeline implementado

```
Carregamento → EDA → Pré-processamento → Treinamento (6 modelos)
    → Cross-Validation 5-Fold → Avaliação no Test Set → SHAP
```

### Modelos comparados

| Modelo | Recall Maligno (CV) | AUC (CV) |
|--------|---------------------|----------|
| **SVM** ✅ | **0,958 ± 0,037** | **0,995** |
| Logistic Regression | 0,944 ± 0,052 | 0,995 |
| Random Forest | 0,939 ± 0,050 | 0,989 |
| KNN | 0,925 ± 0,046 | 0,985 |
| Gradient Boosting | 0,906 ± 0,066 | 0,993 |
| Decision Tree | 0,878 ± 0,082 | 0,907 |

> **Recall da classe Maligna** é a métrica principal: falsos negativos (maligno classificado como benigno) têm custo clínico muito maior que falsos positivos.

### Resultados finais — SVM no test set

| Métrica | Valor |
|---------|-------|
| Accuracy | 98,25% |
| Recall (Maligno) | 97,62% |
| Precision (Maligno) | 97,62% |
| F1-Score (Maligno) | 97,62% |
| AUC-ROC | 0,9950 |
| Falsos Negativos | **1** de 42 |
| Falsos Positivos | 1 de 72 |

---

## Como executar

### Opção 1 — Local

```bash
# 1. Clonar o repositório
git clone https://github.com/otaviano/fiap-tech-challenge-fase1.git
cd fiap-tech-challenge-fase1

# 2. Criar ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Abrir o notebook
jupyter notebook breast_cancer_ml.ipynb
```

### Opção 2 — Docker

```bash
# Build da imagem
docker build -t breast-cancer-ml .

# Executar container (Jupyter na porta 8888)
docker run -p 8888:8888 breast-cancer-ml
```

Acesse `http://localhost:8888` no navegador e abra `breast_cancer_ml.ipynb`.

### Opção 3 — Google Colab

Clique no badge abaixo para abrir diretamente no Colab (sem instalação):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otaviano/fiap-tech-challenge-fase1/blob/main/breast_cancer_ml.ipynb)

---

## Dependências

```
scikit-learn >= 1.3.0
pandas       >= 2.0.0
numpy        >= 1.24.0
matplotlib   >= 3.7.0
seaborn      >= 0.12.0
shap         >= 0.43.0
jupyter      >= 1.0.0
```

---

## Explicabilidade (SHAP)

O modelo Random Forest foi analisado com **SHAP (SHapley Additive exPlanations)** para garantir interpretabilidade clínica. As 3 features mais influentes foram:

1. `worst area` — área do pior núcleo celular observado
2. `worst concave points` — pontos côncavos do pior núcleo
3. `worst radius` — raio do pior núcleo celular

Valores altos nessas features indicam forte predição de tumor **Maligno**.

---

## Considerações clínicas

Este modelo é uma **ferramenta de apoio ao diagnóstico**, não um substituto ao médico:

- Alto Recall (97,6%) minimiza falsos negativos — o erro mais crítico em oncologia
- Explicabilidade via SHAP permite que o médico valide a predição
- **O médico sempre tem a palavra final no diagnóstico**

---

## Relatório técnico

O relatório completo com análise de pré-processamento, justificativa dos modelos, interpretação dos resultados e discussão crítica está em [RELATORIO_TECH_CHALLENGE.md](RELATORIO_TECH_CHALLENGE.md).

---

*FIAP Postech — IA para Devs | Tech Challenge Fase 1 | Junho 2026*
