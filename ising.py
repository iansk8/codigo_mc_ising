'''
=============================================================================================
PROGRAMA: MODELO DE ISING EM 2D, COM CAMPO EXTERNO NULO. SIMULALCAO MONTE CARLO

USANDO ALGORITMO DE METROPOLIS

AUTORES: IAN DE JESUS FONTENELE LOPES/PROF. Dr. CICERO THIAGO GOMES DOS SANTOS, IFSERTAO-PE
=============================================================================================
'''
#============================================================================================
#                               IMPORTANDO BIBLIOTECAS
import numpy as np
from numba import njit
from math import exp
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.linear_model import LinearRegression
#============================================================================================

#============================================================================================
#                                DEFININDO PARAMETROS
l = 8 #Dimensao da rede
num_dir = 4 #Numero de vizinhos
t = 2.2 #Temperatura inicial
mcs = 10**6 #passo monte carlo
range_temp = 20 #Faixa de temperatura
passo_temp = 0.01 #Passo de temperatura
#============================================================================================

def inicialize_rede(L): #Cria rede de spins 
    matriz = []

    for _ in range(L):
        matriz.append([1 for i in range(L)]) 
        #Para rede aleatoria trocar 1 por: np.random.choice((1, -1))

    return matriz


@njit()
def metropolis(mcs, matriz, T, L, energia_total, magnetizacao): 
    #Executa o Algoritmo de Metropolis
    energia_autal = energia_total
    sum_energia = 0
    sum_energia_quadrada = 0
    magnetizacao_atual = magnetizacao
    sum_magnetizacao = 0
    sum_magnetizacao_quadrada = 0
    for i in range(mcs): #Loop sobre o Passo Monte Carlo
        for j in range(L * L): #Loop sobre os sites
            y = np.random.choice(L) #Site aleatorio
            x = np.random.choice(L) #Site aleatorio
            delta_E = calcular_variacao_energia(matriz, L, y, x)
            delta_M = calcular_variacao_magnetizacao(matriz, y, x)
            if delta_E > 0:
                q = np.random.random() #Número aleatorio entre 0 e 1
                prob = np.exp(-delta_E/ T) #Calcula a probabilidade de Boltzmann
                if q < prob:
                    matriz[y][x] = (matriz[y][x]) * -1 #Inverte o spin
                    energia_autal += delta_E
                    magnetizacao_atual += delta_M
            else:
                matriz[y][x] = (matriz[y][x]) * -1 #Inverte o spin
                energia_autal += delta_E
                magnetizacao_atual += delta_M
            sum_energia += energia_autal
            sum_energia_quadrada += energia_autal * energia_autal
            sum_magnetizacao += magnetizacao_atual
            sum_magnetizacao_quadrada += magnetizacao_atual * magnetizacao_atual
    e_m = sum_energia/(mcs * (L**2))
    e_m_q = sum_energia_quadrada/(mcs * (L**2))
    m_m = sum_magnetizacao/(mcs * (L**2))
    m_m_q = sum_magnetizacao_quadrada/(mcs * (L**2))
    t_ = round(T + passo_temp, 5)
    return matriz, e_m, energia_autal, e_m_q, m_m, sum_energia, magnetizacao_atual, m_m_q, t_


@njit()
def calcular_variacao_energia(matriz, L, y, x): #Calcula a variacao da energia interna
    sum_vizinhos, _ = vizinho(matriz, L, y, x)
    variacao_energia = 2 * matriz[y][x] * sum_vizinhos
    return variacao_energia


@njit()
def calcular_variacao_magnetizacao(matriz, y, x): #Calcula a variacao da magnetizacao
    variacao_magnetizacao = -2 * matriz[y][x]
    return variacao_magnetizacao


@njit()
def vizinho(matriz, L, y, x): #Condicoes de contorno periodica
    vizinhos = []

    vizinhos.append(matriz[y][(x+1) % L]) #primeiro vizinho da direita
    vizinhos.append(matriz[y-1][x]) # primeiro vizinho de cima
    vizinhos.append(matriz[y][x-1]) # primeiro vizinho da esquerda
    vizinhos.append(matriz[(y+1) % L][x]) #primeiro vizinho de baixo
    sum_vizinhos = 0
    for _ in range(4): #Soma sobre os vizinhos
        sum_vizinhos += vizinhos[_]
    return sum_vizinhos, vizinhos


def energia_tot(matriz, L, num_dir):
    energia_total = 0
    for y in range(L):
        for x in range(L):
            for _ in range(num_dir):
                i , vizinhos = vizinho(matriz, L, y, x)
                energia_total += (matriz[y][x] * vizinhos[_])
    energia_total = -energia_total / 2 
    #Dividido por 2 dois porque cada sitio e contado duas vezes
    return energia_total


def magnetizcao_total(matriz, L):
    magnetizcao = 0
    for i in range(L):
        for j in range(L):
            magnetizcao += matriz[i][j]
    return magnetizcao


#============================================================================================
#                                          PROGRAMA PRINCIPAL
matriz = np.array(inicialize_rede(l))
energia_total = energia_tot(matriz, l, num_dir)
magnetizcao = magnetizcao_total(matriz, l)
matriz, _, energia_total, _, _, _, magnetizcao, _, _ = metropolis(mcs, matriz, t, l, energia_total, magnetizcao) 
#Executa o Algoritmo de Metropolis para a equilibracao do sistema
matriz = np.array(matriz)
energia_media_ps_8 = []
m_m_ps_8 = []
calor_esp_64 = []
suscep_mag_64 = []
t_p = []
for k in range(range_temp): #Loop sobre as temperaturas
    matriz, e_m, energia_total, e_m_q, m_m, sum_e, magnetizcao, m_m_q, t = metropolis(mcs, matriz, t, l, energia_total, magnetizcao) 
    #Executa o algoritmo para calcular as propriedades
    t = round(t + passo_temp, 4)
    t_p.append(t)
    e = sum_e/(mcs * (l**2))
    energia_media_ps_8.append(e/(l * l))
    calor = ((e_m_q - (e_m * e_m))/( l * l * t * t))
    calor_esp_64.append(calor) #Calor especifico por sitio
    m_m_ps_8.append((m_m)/(l * l))
    suscep_mag_64.append((m_m_q -(m_m * m_m))/(l * l * t))