import pandas as pd
import random
from sklearn.linear_model import LinearRegression
import numpy as npy

df = pd.read_csv('TabelaCarros.csv')

var_indep = ['Qtd_Pessoas', 'Porta_Malas', 'Ar_Condicionado', 'Cambio', 'Categoria']
var_dep = ['Preco']

dados_x = df[var_indep]
dados_y = df[var_dep]

modelo = LinearRegression().fit(dados_x,dados_y)


QP = random.choice([2,5,7])
PM = random.randint(1, 3)
AC = 1
Cam = random.randint(1, 2)
Cat = random.randint(1, 10)

valores_teste = npy.array([[QP, PM, AC, Cam, Cat]])

predicao = modelo.predict(valores_teste)

print("Preço Estimado: " + str(int(predicao)))
