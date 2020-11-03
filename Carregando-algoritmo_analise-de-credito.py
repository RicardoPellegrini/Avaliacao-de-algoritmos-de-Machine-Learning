
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Importando database
base = pd.read_csv('credit_data.csv')


# In[3]:


# Verificando dados com idade negativa
base.loc[base['age'] < 0]


# In[4]:


### Maneiras de contornar o problema das idades menores que zero

## 1) Apagar a coluna por inteiro (não recomendada, neste caso)
# base.drop('age', 1, inplace=True)

## 2) Apagar apenas os registros, por completo, que possuem essa incoerência
# base.drop(base[base.age < 0].index, inplace=True)

## 3) Preencher os valores com a média da coluna, apenas dos valores maiores que zero
media = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = media


# In[5]:


# Verificando valores nulos
base.loc[pd.isnull(base['age'])]


# In[6]:


# Divisão do dataset entre variáveis preditoras e target
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# In[7]:


# Substituindo os valores missing pela média de cada coluna
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(previsores[:, 0:3])

previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


# In[8]:


## Fazendo o escalonamento (normalização) dos atributos
from sklearn.preprocessing import StandardScaler

# Padronização
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Normalização
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# previsores = scaler.fit_transform(previsores)


# In[9]:


# Carregando modelos previamente treinados
import pickle


# In[10]:


svm = pickle.load(open('svm_finalizado.sav', 'rb'))


# In[11]:


random_forest = pickle.load(open('random_forest_finalizado.sav', 'rb'))


# In[12]:


mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))


# In[13]:


# Resultados para essa base de dados aplicando os modelos carregados previamente
resultado_svm = svm.score(previsores, classe)
resultado_svm


# In[14]:


resultado_random_forest = random_forest.score(previsores, classe)
resultado_random_forest


# In[16]:


resultado_mlp = mlp.score(previsores, classe)
resultado_mlp


# In[17]:


# Fazendo uma previsão para um dado específico
# uma pessoa que tenha renda de R$ 50000, tenha 40 anos e tenha 5000
novo_registro = [[50000, 40, 5000]]


# In[18]:


# Transformando em array, efetivamente
import numpy as np


# In[19]:


novo_registro = np.asarray(novo_registro)


# In[20]:


# Fazendo reshape para que seja possível fazer o escalonamento
novo_registro = novo_registro.reshape(-1,1)
novo_registro = scaler.fit_transform(novo_registro)


# In[21]:


# Voltando ao shape original
novo_registro = novo_registro.reshape(-1,3)


# In[22]:


# Calculando com o svm
resposta_svm = svm.predict(novo_registro)
resposta_svm


# In[23]:


# Calculando com o random forest
resposta_random_forest = random_forest.predict(novo_registro)
resposta_random_forest


# In[24]:


# Calculando com as redes neurais
resposta_mlp = mlp.predict(novo_registro)
resposta_mlp

