# Classificador de Frutas com CNN 🍎🍌🍊  
Este projeto implementa um classificador de imagens de frutas utilizando **Redes Neurais Convolucionais (CNN)**.  
O notebook `Classify product.ipynb` conduz todo o processo: preparação dos dados, padronização das classes, aumento de dados (data augmentation), treinamento do modelo e exportação final (`modelo_frutas.h5`).

Além disso, o repositório inclui uma **API FastAPI** capaz de receber imagens, processá-las e retornar a fruta prevista pela CNN.

---

## 📌 Objetivo
Treinar e servir um modelo de deep learning capaz de classificar imagens de frutas usando um dataset público do Kaggle.

---

## 🧠 O que o Notebook Faz

### 1. Download Automático do Dataset (Kaggle)
O notebook baixa o dataset direto da plataforma Kaggle usando a API oficial e organiza os arquivos automaticamente.

### 2. Padronização dos Nomes das Classes
Correções aplicadas:
- remoção de acentos  
- nomes uniformes  
- eliminação de duplicidades  
- ajuste de maiúsculas/minúsculas  

### 3. Verificação das Classes
Após a padronização o notebook:
- revalida o diretório de treino  
- confirma quantidade de imagens por classe  
- garante consistência do dataset  

### 4. Construção do Modelo CNN
A CNN implementada possui:
- múltiplas camadas convolucionais  
- camadas de pooling  
- *dropout* para reduzir overfitting  
- uma *dense layer* final softmax para classificação  

### 5. Treinamento Inicial
O primeiro treinamento registrou:
- **Acurácia de validação máxima: ~47.9%**

### 6. Data Augmentation Offline
Foram criadas imagens extras fisicamente no disco, aumentando a diversidade real do dataset:

Técnicas usadas:
- rotação  
- zoom  
- deslocamento  
- flip horizontal  

### 7. Novo Treinamento
Após o aumento de dados:
- o modelo foi treinado novamente  
- as métricas melhoraram  
- foram gerados gráficos de acurácia e loss  

### 8. Visualização de Métricas
Inclui:
- gráfico da evolução da acurácia  
- gráfico do loss  
- comparação entre treino e validação  

---
project/
├── Classify product.ipynb
├── modelo_frutas.h5
├── train_variacoes/ # pastas das frutas após augmentation offline
├── main.py # API FastAPI
├── index.html # interface web (upload + preview)
└── README.md

---

## 🛠️ Tecnologias Utilizadas
### Treinamento
- Python 3  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Kaggle API  
- PIL  
- OpenCV  
- pathlib / os / shutil  

### API
- FastAPI  
- Uvicorn  
- TensorFlow (inference)  
- NumPy  
- Pillow  

---

# 🌐 API — Classificação de Frutas

O arquivo **`main.py`** implementa uma API completa para servir o modelo treinado.

### 📦 Funcionamento

1. Carrega o modelo:
```python
model = tf.keras.models.load_model("modelo_frutas.h5")

2. Lê automaticamente as classes a partir do diretório:
## 🧱 Estrutura Recomendada do Projeto
train_dir = "train_variacoes"
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

3. Possui um pré-processador para imagens:
def preprocess_image(image_bytes, target_size=(128,128)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)/255.0
    return np.expand_dims(image_array, axis=0)
