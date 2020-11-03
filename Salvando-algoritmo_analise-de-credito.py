
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Importando database
base = pd.read_csv('credit_data.csv')


# In[3]:


# Estatísticas do database
base.describe()


# In[4]:


# Amostra dos dados
base.head()


# In[5]:


# Verificando dados com idade negativa
base.loc[base['age'] < 0]


# In[6]:


### Maneiras de contornar o problema das idades menores que zero

## 1) Apagar a coluna por inteiro (não recomendada, neste caso)
# base.drop('age', 1, inplace=True)

## 2) Apagar apenas os registros, por completo, que possuem essa incoerência
# base.drop(base[base.age < 0].index, inplace=True)

## 3) Preencher os valores com a média da coluna, apenas dos valores maiores que zero
media = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = media


# In[7]:


# Verificando valores nulos
base.loc[pd.isnull(base['age'])]


# In[8]:


# Divisão do dataset entre variáveis preditoras e target
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# In[9]:


# Substituindo os valores missing pela média de cada coluna
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(previsores[:, 0:3])

previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


# In[10]:


## Fazendo o escalonamento (normalização) dos atributos
from sklearn.preprocessing import StandardScaler

# Padronização
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Normalização
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# previsores = scaler.fit_transform(previsores)


# In[11]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# In[12]:


# modelo SVM
classificadorSVM = SVC(kernel='rbf', C=2.0)
classificadorSVM.fit(previsores, classe)


# In[13]:


# modelo Random Forest
classificadorRandomForest = RandomForestClassifier(n_estimators=40, criterion='entropy')
classificadorRandomForest.fit(previsores, classe)


# In[14]:


# modelo Redes Neurais
classificadorMLP = MLPClassifier(verbose=True, max_iter=500, tol=0.00001, solver='adam', hidden_layer_sizes=(100),
                                activation='relu', batch_size=200, learning_rate_init=0.001)
classificadorMLP.fit(previsores, classe)


# In[15]:


# Salvando modelos já treinados para aplicações comerciais
import pickle


# In[16]:


pickle.dump(classificadorSVM, open('svm_finalizado.sav', 'wb'))


# In[17]:


pickle.dump(classificadorRandomForest, open('random_forest_finalizado.sav', 'wb'))


# In[18]:


pickle.dump(classificadorMLP, open('mlp_finalizado.sav', 'wb'))

