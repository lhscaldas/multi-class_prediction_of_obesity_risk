import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Importação de dados
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

# Codificação de dados categóricos
label_encoder = LabelEncoder()
train_dataset_encoded = train_dataset.copy()  # Faz uma cópia do dataset de treino para evitar alterações no original
mapeamento = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
}
train_dataset_encoded["NObeyesdad"] = train_dataset["NObeyesdad"].map(mapeamento) # mapeamento manual dos labels para manter a ordem
for col in train_dataset.columns.difference(['NObeyesdad']):
    if train_dataset[col].dtype == 'object':  # Verifica se a coluna é categórica
        train_dataset_encoded[col] = label_encoder.fit_transform(train_dataset[col])

test_dataset_encoded = test_dataset.copy() # Faz uma cópia do dataset de teste para evitar alterações no original
for col in test_dataset.columns:
    if test_dataset[col].dtype == 'object':  # Verifica se a coluna é categórica
        test_dataset_encoded[col] = label_encoder.fit_transform(test_dataset[col])

# Forma 1
# scaler = StandardScaler()
# cols_to_normalize = train_dataset_encoded.columns.difference(['id','NObeyesdad']) # deixa de fora a id e os labels
# train_dataset_encoded[cols_to_normalize] = scaler.fit_transform(train_dataset_encoded[cols_to_normalize])
# cols_to_normalize = test_dataset_encoded.columns.difference(['id']) # deixa de fora a id
# test_dataset_encoded[cols_to_normalize] = scaler.fit_transform(test_dataset_encoded[cols_to_normalize])
# features = train_dataset_encoded[train_dataset_encoded.columns.difference(['id','NObeyesdad'])].values # features do dataset de treino removendo o id
# labels = train_dataset_encoded['NObeyesdad'].values # labels
# test = test_dataset_encoded.drop(columns=['id']).values # removendo o id do dataset de teste

# Forma 1 mod
scaler = StandardScaler()
train_dataset_encoded.iloc[:,1:17] = scaler.fit_transform(train_dataset_encoded.iloc[:,1:17].values).astype('int32')
test_dataset_encoded.iloc[:,1:17] = scaler.fit_transform(test_dataset_encoded.iloc[:,1:17].values).astype('int32')
features = train_dataset_encoded.iloc[:,1:17].values # features do dataset de treino removendo o id
labels = train_dataset_encoded.iloc[:,17].values # labels
test = test_dataset_encoded.iloc[:,1:].values # removendo o id do dataset de teste

# # Forma 2 (correta)
# features2 = train_dataset_encoded.iloc[:,1:17].values # features do dataset de treino removendo o id
# labels2 = train_dataset_encoded.iloc[:,17].values # labels
# test2 = test_dataset_encoded.drop(columns=['id']).values # removendo o id do dataset de teste
# scaler = StandardScaler()
# features2 = scaler.fit_transform(features2)
# test2 = scaler.fit_transform(test2)

# # Forma 3
# features3 = train_dataset_encoded[train_dataset_encoded.columns.difference(['id','NObeyesdad'])].values # features do dataset de treino removendo o id
# labels3 = train_dataset_encoded['NObeyesdad'].values # labels
# test3 = test_dataset_encoded.drop(columns=['id']).values # removendo o id do dataset de teste
# scaler = StandardScaler()
# features3 = scaler.fit_transform(features3)
# test3 = scaler.fit_transform(test3)


# print(np.mean(features2-features3))
# print(np.array_equal(features2,features3))
# print(np.array_equal(test2,test3))




# Treinamento do modelo e predição
#  {'max_depth': 40, 'max_features': 5, 'min_samples_leaf': 2, 'n_estimators': 900}
random_forest = RandomForestClassifier(max_depth=40, max_features=5, min_samples_leaf=2, n_estimators=900, n_jobs=-1)
random_forest.fit(features,labels) # treinamento do modelo de Random Forest com o dataset de treinamento completo
y_predict_forest = random_forest.predict(test) # classificação com o modelo Random Forest

# Comparação com o resultado bom
print(f"Resultado agora {np.unique(y_predict_forest, return_counts=True)[1]}")
best = pd.read_csv('resuldado_best.csv')
print(f"Resultado anterior {np.unique(best.iloc[:,1].map(mapeamento).values, return_counts=True)[1]}")