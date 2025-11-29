# 🍎🍌🍊 Classificador de Frutas com CNN  

Este projeto implementa um **classificador de imagens de frutas** utilizando **Redes Neurais Convolucionais (CNN)**.  
O notebook `Classify product.ipynb` conduz todo o fluxo completo:  
- preparação e padronização do dataset  
- aumento de dados (data augmentation)  
- geração de variações no disco  
- treinamento do modelo  
- exportação final (`modelo_frutas.h5`)

Além disso, o repositório inclui uma **API FastAPI** capaz de receber imagens, processá-las e retornar a fruta prevista pelo modelo treinado.

---

## 📌 Objetivo
Treinar e servir um modelo de deep learning capaz de **classificar imagens de frutas** usando um dataset público do Kaggle, aplicando boas práticas de preparação de dados e implantação.

---

# 🧠 O que o Notebook Faz

## 1. Download Automático do Dataset (Kaggle)
O notebook utiliza a API oficial do Kaggle para:
- baixar o dataset  
- extrair os arquivos  
- organizar automaticamente a estrutura de diretórios  

---

## 2. Padronização dos Nomes das Classes
O dataset original possui variações e inconsistências.  
O notebook realiza:

- remoção de acentos  
- uniformização de nomes  
- eliminação de duplicidades  
- normalização de maiúsculas/minúsculas  
- correções estruturais em pastas  

---

## 3. Verificação das Classes
Após a padronização, o notebook:

- revalida o diretório de treino  
- verifica quantidade de imagens por classe  
- confirma se não há classes faltando ou duplicadas  
- imprime estatísticas do dataset  

---

## 4. Construção do Modelo CNN
A CNN contém:

- múltiplas camadas Convolution2D  
- pooling para redução de dimensionalidade  
- batch normalization  
- dropout (reduz overfitting)  
- camada densa final com Softmax  

---

## 5. Treinamento Inicial
Primeiro treino (sem augmentation físico no disco):

- **Acurácia máxima de validação: 47.9%**

---

## 6. Data Augmentation Offline
Para aumentar o dataset, foram geradas **novas imagens fisicamente no disco**, aumentando a diversidade real.

Técnicas aplicadas:
- rotação  
- zoom  
- deslocamento  
- flip horizontal  
- efeitos de transformação moderados  

---

## 7. Novo Treinamento (Após Augmentation)
Com o dataset expandido:

- o modelo foi treinado novamente  
- as métricas melhoraram significativamente  
- **Acurácia de validação: ~93%**  
- gráficos de loss e accuracy foram gerados  

---

## 8. Visualização das Métricas
Inclui:

- gráfico de acurácia (treino × validação)  
- gráfico de loss (treino × validação)  
- análise visual da evolução durante as épocas  

---

# 🛠️ Tecnologias Utilizadas

## 🔹 Treinamento
- Python 3  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Kaggle API  
- PIL  
- OpenCV  
- pathlib / os / shutil  

## 🔹 API
- FastAPI  
- TensorFlow (inference)  
- NumPy  
- Pillow  

---

# 🌐 API — Classificação de Frutas (FastAPI)

O arquivo **`main.py`** implementa uma API pronta para uso em produção.

## 📦 Funcionamento

### 1. Carrega o modelo treinado
```python
model = tf.keras.models.load_model("modelo_frutas.h5")
