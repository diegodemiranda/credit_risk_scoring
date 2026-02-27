# 🏦 Credit Risk Scoring: Modelagem Probabilística de Inadimplência

Este repositório contém a implementação, a arquitetura de dados e os artefatos de treino para um sistema avançado de *score* de risco voltado à concessão de crédito.

A solução foi desenhada com princípios de arquitetura de software robusta para superar as limitações de sistemas puramente binários (aprovado/rejeitado). O modelo realiza a modelagem probabilística da inadimplência, estimando a probabilidade condicional $P(Y=1|x)$ e entregando um *score* de risco contínuo no intervalo $[0,1]$.

---

## 🏗 Arquitetura do Sistema e Modelos

A base de código implementa redes neurais profundas, organizadas de forma modular para lidar com a não linearidade de dados estruturados financeiros.

### 1. Modelo Principal: TabNet
A arquitetura de eleição para o *score* contínuo em produção.
* **Atenção Sequencial e Seleção Esparsa:** Utiliza a função *sparsemax* para uma seleção adaptativa e dinâmica das características de entrada $x$ mais relevantes em cada etapa de decisão.
* **Interpretabilidade (XAI):** As máscaras de atenção aprendidas permitem extrair a importância exata de cada variável, garantindo a explicabilidade exigida por regulações financeiras.

### 2. Abordagem Híbrida Experimental (DWT + CNN + Transformers)
Uma rota de extração avançada de *features* para cenários de alta complexidade.
* **Transformada Wavelet Discreta (DWT):** Pré-processamento que decompõe o sinal financeiro em coeficientes multirresolução de aproximação (baixa frequência) e de detalhe (alta frequência).
* **Redes Convolucionais (CNN) e Transformers:** Atuam sobre os coeficientes para capturar padrões locais e relações latentes complexas de longo alcance.

---

## 📊 Conjunto de Dados

Validado com o **Realistic Loan Approval Dataset - US and Canada** (Kaggle). O vetor $x$ abrange:

* **Perfil Demográfico:** Idade, tempo de experiência profissional.
* **Saúde Financeira:** Renda anual, patrimônio/poupança, pontuação de crédito base, dívida atual.
* **Comportamento e Histórico:** Histórico de crédito, anotações negativas, inadimplências registradas.

## 🧮 Pipeline de Treino e Otimização Matemática

O processo de treino foi desenhado com rigor matemático para garantir a calibração probabilística do risco:

* **Função de Perda (Binary Cross-Entropy):** Derivada da Máxima Verossimilhança, penaliza fortemente previsões erradas de alta confiança. A função para $n$ observações é dada por:

$$\mathcal{L}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log f_\theta(x_i) + (1-y_i) \log(1-f_\theta(x_i)) \right]$$

* **Otimização (Adam):** Combina momentos de primeira e segunda ordem para garantir estabilidade perante escalas distintas das variáveis financeiras. A atualização dos parâmetros segue a regra:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

* **Regularização:** Utilização de *Early Stopping* condicionado à perda de validação ($\mathcal{L}_{val}^{(t)} > \mathcal{L}_{val}^{(t-k)}$) para evitar *overfitting* do modelo em relação aos dados de treino.

## 📈 Performance

Os modelos apresentam capacidade de isolamento de risco de padrão bancário de excelência:

| Arquitetura | KS (Kolmogorov-Smirnov) | ROC AUC | Destaque Arquitetural |
| :--- | :--- | :--- | :--- |
| **TabNet** | **0.82** | **0.9754** | Fronteira de decisão altamente precisa; baixíssima taxa de falsos negativos. |
| **Híbrido (DWT+CNN+TF)** | **0.79** | **0.9652** | Excelente captura de não linearidades complexas via multirresolução. |

---

## ⚙️ Pré-requisitos e Configurações

Para garantir a reprodutibilidade e o isolamento do ambiente de desenvolvimento, siga os pré-requisitos abaixo.

### Dependências de Sistema
* **Sistema Operacional:** Linux (Ubuntu 20.04/22.04), macOS ou Windows (via WSL2).
* **Python:** Versão `3.10` ou superior.
* **Hardware:** GPU com suporte a NVIDIA CUDA 11.8+ (Altamente recomendado para o treino das arquiteturas com Transformers e CNNs).

### Bibliotecas Principais (Core Stack)
* `torch` e `torchvision` (Motor de Deep Learning)
* `pytorch-tabnet` (Implementação otimizada da arquitetura TabNet)
* `scikit-learn` (Métricas de avaliação e *splits* de dados)
* `pywavelets` (Para a Transformada Wavelet Discreta - DWT)
* `pandas` e `numpy` (Manipulação de tensores e matrizes)

---

 ## 💻 Estrutura do Diretório

```
├── data/
│   ├── raw/                 # Dados brutos
│   └── processed/           # Matrizes e tensores tratados
├── notebooks/               # Análise exploratória e métricas de validação
├── src/
│   ├── config/              # YAMLs de hiperparâmetros e variáveis
│   ├── data/                # Dataloaders e Transformada Wavelet (DWT)
│   ├── models/              # Arquiteturas: tabnet.py, hybrid.py
│   ├── train.py             # Lógica de otimização (BCE Loss, Early Stopping)
│   └── predict.py           # Endpoint de inferência
├── .env.example             # Template de variáveis de ambiente
├── requirements.txt         # Gestão de dependências
└── README.md                # Documentação técnica
````

---

## 🚀 Como Executar (Instalação e Uso)

### 1. Configuração do Ambiente de Desenvolvimento

**Passo 1: Clonar o repositório:**
```bash
git clone [https://github.com/seu-usuario/credit-risk-scoring.git](https://github.com/seu-usuario/credit-risk-scoring.git)
cd credit-risk-scoring
```


**Passo 2: Criar e ativar um ambiente virtual:**
```bash
#Usando Python:
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

#Usando Conda:
conda create -n credit_score_env python=3.10 -y
conda activate credit_score_env
```

**Passo 3: Instalar as dependências:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
``` 

**Passo 4: Configuração de Variáveis de Ambiente (.env):**
```bash
DATA_PATH=./data/raw/loan_approval_dataset.csv
MODEL_SAVE_PATH=./src/models/saved_weights/
RANDOM_SEED=42
BATCH_SIZE=256
MAX_EPOCHS=200]
``` 
___

### 2. Iniciar o Treino (Training Pipeline)
Para treinar um modelo do zero, execute o módulo de treino passando a arquitetura desejada como argumento:

```bash
python src/train.py --model tabnet --config src/config/train_params.yaml
```

___

### 3. Inferência e Geração de Score
Para calcular a probabilidade de risco contínua P(Y=1∣x) para novos clientes num cenário de produção:

```bash
python src/predict.py --input_json data/samples/new_client.json --model_weights src/models/saved_weights/tabnet_best.pt
```

## 👥 Autores e Pesquisadores

 Este projeto é fruto de pesquisa aplicada e desenvolvimento arquitetural na interseção entre Ciência da Computação e Matemática Estatística, desenvolvido na **Universidade Federal de Uberlândia (UFU)**[cite: 3, 4]:

*  **Diego de Miranda da Silva** - *Faculdade de Computação* [cite: 2, 3]
*  **Giovanna Nucci P. de Oliveira** - *Instituto de Matemática e Estatística* [cite: 2, 4]
*  **Isadora Pfeifer Spirandelli** - *Instituto de Matemática e Estatística* [cite: 2, 4]
*  **Matheus Vinicius Maximo Santos** - *Faculdade de Computação* [cite: 2, 3]

## 📚 Referências Científicas

A fundamentação teórica e as decisões arquiteturais deste repositório baseiam-se na seguinte literatura técnico-científica:

1. **Arik, S. Ö.; Pfister, T. (2021).** *TabNet: Attentive Interpretable Tabular Learning*.  Proceedings of the AAAI Conference on Artificial Intelligence[cite: 319, 320].
2. **Chen, X. et al. (2024).** *Credit Rating Model Based on Improved TabNet*.  Mathematics, MDPI[cite: 321].
3. **Alexandridis, A. K.; Zapranis, A. D. (2014).** *Wavelet Neural Networks: With Applications in Financial Engineering, Chaos, and Classification*.  Wiley[cite: 323, 324].
4. **Jarrah, M. (2019).** *A recurrent neural network and a discrete wavelet transform to predict the Saudi stock price trends*.  IJACSA[cite: 325, 326].
5. **Pishchulin, L. et al. (2021) .** *Credit scoring using neural networks and SURE posterior probability calibration*[cite: 327, 328].
6.  **Kaggle (2025).** *Synthetic/Realistic Loan Approval Dataset - US and Canada*[cite: 318].

## 🤝 Como Contribuir

Contribuições são muito bem-vindas, especialmente em frentes de otimização de MLOps e novas arquiteturas. 
1. Faça um *Fork* do projeto.
2. Crie uma *Branch* para sua *Feature* (`git checkout -b feature/NovaArquitetura`).
3. Faça o *Commit* das suas alterações (`git commit -m 'feat: Adicionando camada de Reinforcement Learning'`).
4. Faça o *Push* para a *Branch* (`git push origin feature/NovaArquitetura`).
5. Abra um *Pull Request*.

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---
*Construindo sistemas de crédito mais justos e precisos através da inferência probabilística e Inteligência Artificial de ponta.* 🚀
