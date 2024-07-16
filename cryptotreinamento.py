# -*- coding: utf-8 -*-
"""cryptotreinamento.ipynb

"""

import pandas as pd
import pyodbc
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, precision_score, accuracy_score
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
import os




warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning, module='feature_engine')
import logging

# Defina o nível de log para suprimir mensagens informativas
logging.getLogger('lightgbm').setLevel(logging.WARNING)

# Configurações de conexão com o servidor SQL
server = 'xxxxxx'  # Seu servidor SQL
database = 'xxx'  # Seu banco de dados
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
# Conectando ao servidor SQL
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

sql_query = '''
SELECT chain_id, dex_id, pair_address, token_address, token_name, token_symbol, token_m5_buys, token_m5_sells, token_h1_buys, token_h1_sells, token_h1_to_m5_buys, token_liquidity, token_market_cap, token_created_since_horas, token_created_since_min, token_eti, token_header, token_website, token_twitter, token_links, token_img_key, token_price_usd,  token_price_change_h24, token_price_change_h6, token_price_change_h1, token_price_change_m5, token_created_dia, token_created_data, token_created_hora

FROM DEX;
'''

df = pd.read_sql(sql_query, conn)

# Substituir valores diferentes de '0' por '1'
df['token_img_key'] = df['token_img_key'].apply(lambda x: '1' if x != '0' else x)
print(df['token_img_key'].value_counts())

# Primeira alteração: substituir valores na coluna 'dex_id' com mais de 20 caracteres por "outros"
df.loc[df['dex_id'].str.len() > 20, 'dex_id'] = 'outros'


# Terceira alteração: substituir valores NULL na coluna 'token_h1_to_m5_buys' por 0
df['token_h1_to_m5_buys'].fillna(0, inplace=True)

"""data ot

"""

data_limite = pd.to_datetime('2024-04-11')

df['token_created_data'] = pd.to_datetime(df['token_created_data'], format='%Y%m%d')
# Defina a data limite para a segmentação

"""horaaa"""

hora_seg1 = 15
hora_seg2 = hora_seg1 + 15
hora_seg3 = hora_seg1 + 90
hora_seg4 = hora_seg2  + 90
hora_limite = hora_seg4 +  10

"""Segmentando a base"""

df_1hora = df[(df['token_created_since_min'] >= hora_seg1) & (df['token_created_since_min'] <= hora_seg2)]

# Agrupar os dados pela coluna 'token_address' e manter apenas a linha com o maior valor de 'token_created_since_horas' em cada grupo
df_1hora = df_1hora.loc[df_1hora.groupby('token_address')['token_created_since_min'].idxmax()]

df_limite = df[(df['token_created_since_min'] >= hora_limite) & (df['token_created_since_min'] <= (hora_limite + 90))]

# Agrupar os dados pela coluna 'token_address' e manter apenas a linha com o maior valor de 'token_created_since_horas' em cada grupo
df_limite = df_limite.loc[df_limite.groupby('token_address')['token_created_since_min'].idxmax()]

print(df_1hora['token_address'].nunique())

"""segmentando nova hora"""

df_2hora = df[(df['token_created_since_min'] >= hora_seg3) & (df['token_created_since_min'] <=  hora_seg4)]

# Agrupar os dados pela coluna 'token_address' e manter apenas a linha com o maior valor de 'token_created_since_horas' em cada grupo
df_2hora = df_2hora.loc[df_2hora.groupby('token_address')['token_created_since_min'].idxmax()]

print(df_2hora['pair_address'].nunique())

"""merged"""

df_novomerged = pd.merge(df_1hora, df_limite[['token_address', 'token_price_usd', 'token_liquidity', 'token_created_since_min']], on='token_address', how='left')

# Preencha os valores faltantes com 0
df_novomerged['token_price_usd_y'].fillna(0, inplace=True)
df_novomerged['token_liquidity_y'].fillna(0, inplace=True)
df_novomerged['token_created_since_min_y'].fillna(0, inplace=True)
df_novomerged.rename(columns={'token_price_usd_y': 'token_price_usd_fut', 'token_liquidity_y': 'token_liquidity_fut','token_created_since_min_y':'token_created_since_min_fut',
 'token_price_usd_x': 'token_price_usd', 'token_liquidity_x': 'token_liquidity','token_created_since_min_x':'token_created_since_min'                            }, inplace=True)

df_merged = pd.merge(df_novomerged, df_2hora[['token_address', 'token_price_usd', 'token_liquidity', 'token_created_since_min' ]], on='token_address', how='left')
# Preencha os valores faltantes com 0
df_merged['token_price_usd_y'].fillna(0, inplace=True)
df_merged['token_liquidity_y'].fillna(0, inplace=True)
df_merged['token_created_since_min_y'].fillna(0, inplace=True)

print(df_merged.columns.tolist())

print(df_merged['pair_address'].nunique())

df_merged['crescimento_price'] = (df_merged['token_price_usd_y'] / df_merged['token_price_usd_x']) - 1
df_merged['crescimento_liquidity'] = (df_merged['token_liquidity_y'] / df_merged['token_liquidity_x']) - 1

condicao = (df_merged['crescimento_price'] > 0.046) & (df_merged['crescimento_liquidity'] > 0) & (df_merged['token_price_usd_fut'] > df_merged['token_price_usd_x']) & (df_merged['token_liquidity_fut'] > 100)
df_merged['crescimento_flag'] = np.where(condicao, 1, 0)

df_diferenca = df_merged[df_merged['crescimento_flag'] == 1]
summary = df_diferenca['crescimento_price'].describe()
print(summary)

removerbase = (df_merged['crescimento_price'] == -1) & (df_merged['token_liquidity_fut'] > 100)

# Removendo linhas que satisfazem a condição
df_merged = df_merged.drop(df_merged[removerbase].index)

print(df_merged['pair_address'].nunique())

# Segmentar o DataFrame com base na data limite
df_ot = df_merged[df_merged['token_created_data'] >= data_limite]
df_otnew = df_merged[(df_merged['token_created_data'] >= data_limite) & (df_merged['token_h1_buys'] >20)]
df_merged = df_merged[df_merged['token_created_data'] < data_limite]

print(df_merged['pair_address'].nunique())

"""categorizando variaveis"""

cat_vars =['chain_id', 'dex_id','token_created_dia', 'token_website', 'token_twitter', 'token_eti', 'token_header', 'token_img_key', 'token_created_hora' ]
target = ['crescimento_flag']
num_vars = ['token_m5_buys','token_m5_sells', 'token_h1_buys', 'token_h1_sells', 'token_liquidity_x', 'token_market_cap', 'token_links', 'token_price_change_h1', 'token_price_change_m5', 'token_h1_to_m5_buys']

features = num_vars+ cat_vars
df[cat_vars] = df[cat_vars].astype('category')





X = df_merged[cat_vars + num_vars]
y= df_merged[target]
X_ot = df_ot[cat_vars + num_vars]
y_ot= df_ot[target]
X_otnew = df_otnew[cat_vars + num_vars]
y_otnew= df_otnew[target]
X[cat_vars] = X[cat_vars].astype('category')



num_linhas = len(df_ot)
print("Número de linhas:", num_linhas)

"""treino e teste

RF
"""

rf = Pipeline(steps=[
    ("numeric_imputer", ArbitraryNumberImputer(variables=num_vars, arbitrary_number=-999)),
    ("categoric_imputer", CategoricalImputer(variables=cat_vars, fill_value="missing")),
    ("one", OneHotEncoder(variables=cat_vars)),
    ("rf", RandomForestClassifier(random_state=42))
])

# Ajuste o pipeline aos seus dados de treinamento (X_train são suas features e y_train são seus rótulos)
rf.fit(X, y)



"""salvando o modelo"""

import joblib

joblib.dump(rf, 'modelo90')

# Carregando o modelo de um arquivo
knn_from_joblib = joblib.load('modelo90')

# Utilizando o modelo para fazer predições
knn_from_joblib.predict(X_otnew)

probas_pred = knn_from_joblib.predict_proba(X_otnew)[:, 1]

# Ajuste do limiar de decisão (por exemplo, 0.5)
limiar = 0.55

# Classificação com base no limiar
previsoes_rf = (probas_pred >= limiar).astype(int)

# Calcular a precisão com o novo limiar
precision = precision_score(y_otnew, previsoes_rf)
print("precision",precision)

accuracy = accuracy_score(y_otnew, previsoes_rf)
print("Acurácia:", accuracy)

roc_auc_teste = roc_auc_score(y_otnew, probas_pred)
print("ROC AUC:", roc_auc_teste)

quantidade_rf2 = np.sum(previsoes_rf == 1)
print("Quantidade classificada como 1:", quantidade_rf2)

caminho_completo = os.path.abspath('modelo120')
print(caminho_completo)