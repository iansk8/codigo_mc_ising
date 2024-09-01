'''
=============================================================================================
PROGRAMA: MODELO DE ISING EM 2D, COM CAMPO EXTERNO NULO. SIMULALÇÃO MONTE CARLO 

USANDO ALGORITMO DE METROPOLIS

AUTORES: IAN DE JESUS FONTENELE LOPES/PROF. Dr. CÍCERO THIAGO GOMES DOS SANTOS, IFSERTAO-PE
=============================================================================================
'''
#============================================================================================
#                               IMPORTANDO BIBLIOTECAS
import numpy as np
from numba import njit
from math import exp
import matplotlib.pyplot as plt
import time
#============================================================================================

#============================================================================================
#                                DEFININDO PARÂMETROS
l = 8 #dimensão da rede 
num_dir = 4 #número de vizinhos
t = 1.0 #temperatura inicial
mcs = 10**6 #passo monte carlo
range_temp = 21 #faixa de temperatura
passo_temp = 0.2 #passo de temperatura
#============================================================================================


def inicialize_rede(L): #Cria rede de spins aleatórios
    matriz = []

    for _ in range(L):
        matriz.append([1 for i in range(L)]) # Para rede aleatória trocar 1 por: np.random.choice((1, -1))
    
    return matriz


@njit()
def metropolis(mcs, matriz, T, L, energia_total, magnetizacao): #Executa o Algoritmo de Metropolis
    energia_autal = energia_total
    sum_energia = 0
    sum_energia_quadrada = 0
    magnetizaçao_atual = magnetizacao
    sum_magnetizaçao = 0
    sum_magnetizaçao_quadrada = 0
    for i in range(mcs): #Loop sobre o Passo Monte Carlo
        for j in range(L * L): #Loop sobre os sites
            y = np.random.choice(L) #Site aleatório
            x = np.random.choice(L) #Site aleatório
            delta_E = calcular_variação_energia(matriz, L, y, x)
            delta_M = calcular_variação_magnetizaçao(matriz, y, x)
            if delta_E > 0:
                q = np.random.random() #Número aleatório entre 0 e 1 
                prob = np.exp(-delta_E/ T) #Calcula a probabilidade de Boltzmann
                if q < prob:
                    matriz[y][x] = (matriz[y][x]) * -1 #Inverto o spin
                    energia_autal += delta_E #Atualiza a energia interna
                    magnetizaçao_atual += delta_M #Atualiza a magnetização
            else:
                matriz[y][x] = (matriz[y][x]) * -1 #Inverto o spin
                energia_autal += delta_E #Atualiza a energia interna
                magnetizaçao_atual += delta_M #Atualiza a magnetização
            sum_energia += energia_autal #Atualiza a soma da energia interna
            sum_energia_quadrada += energia_autal * energia_autal #Atualiza a soma da energia interna quadrática
            sum_magnetizaçao += magnetizaçao_atual #Atualiza a soma da magnetização
            sum_magnetizaçao_quadrada += magnetizaçao_atual * magnetizaçao_atual #Atualiza a soma da magnetização quadrática
    e_m = sum_energia/(mcs * (L**2)) #Calcula a energia interna média
    e_m_q = sum_energia_quadrada/(mcs * (L**2)) #Calcula a energia interna mádia quadrática
    m_m = sum_magnetizaçao/(mcs * (L**2)) #Calcula a magnetização média
    m_m_q = sum_magnetizaçao_quadrada/(mcs * (L**2)) #Calcula a magnetização média quadrática
    return matriz, e_m, energia_autal, e_m_q, m_m, sum_energia, magnetizaçao_atual, m_m_q


@njit()
def calcular_variação_energia(matriz, L, y, x): #Calcula a variação da energia cada vez que um novo sítio aleatório é escolhido
    sum_vizinhos, _ = vizinho(matriz, L, y, x)
    variacao_energia = 2 * matriz[y][x] * sum_vizinhos
    return variacao_energia


@njit()
def calcular_variação_magnetizaçao(matriz, y, x): #Calcula a variação da magnetização cada vez que um novo sítio aleatório é escolhido
    variação_magnetizaçao = -2 * matriz[y][x]
    return variação_magnetizaçao
    

@njit()
def vizinho(matriz, L, y, x): #Condições de contorno periódica
    vizinhos = []

    vizinhos.append(matriz[y][(x+1) % L]) #primeiro vizinho da direita
    vizinhos.append(matriz[y-1][x]) # primeiro vizinho de cima
    vizinhos.append(matriz[y][x-1]) # primeiro vizinho da esquerda
    vizinhos.append(matriz[(y+1) % L][x]) #primeiro vizinho de baixo
    sum_vizinhos = 0
    for _ in range(4): #Soma sobre os vizinhos
        sum_vizinhos += vizinhos[_] 
    return sum_vizinhos, vizinhos


def energia_tot(matriz, L, num_dir): #Calcula a energia interna total
    energia_total = 0
    for y in range(L):
        for x in range(L):
            for _ in range(num_dir):
                i , vizinhos = vizinho(matriz, L, y, x)
                energia_total += (matriz[y][x] * vizinhos[_])
    energia_total = -energia_total / 2 #Dividido por 2 dois porque cada sítio é contado duas vezes
    return energia_total


def magnetizçao_total(matriz, L): #Calcula a manetização total
    magnetizçao = 0
    for i in range(L):
        for j in range(L):
            magnetizçao += matriz[i][j]
    return magnetizçao


#===================================================================================================================
#                                          PROGRAMA PRINCIPAL
matriz = np.array(inicialize_rede(l))
energia_total = energia_tot(matriz, l, num_dir)
magnetizcao = magnetizçao_total(matriz, l)
matriz, _, energia_total, _, _, _, magnetizcao, _ = metropolis(mcs, matriz, t, l, energia_total, magnetizcao) #Executa o Algoritmo de Metropolis para a equilibração do sistema
matriz = np.array(matriz)
energia_media_ps = []
m_m_ps = []
calor_esp = []
susceptibilidade_magnetica = []
t_p = []
inicio = time.time()
for k in range(range_temp): #Loop sobre as temperaturas
    t_p.append(t)
    matriz, e_m, energia_total, e_m_q, m_m, sum_e, magnetizcao, m_m_q = metropolis(mcs, matriz, t, l, energia_total, magnetizcao) #Executa o algoritmo para calcular as propriedades
    matriz = np.array(matriz)
    t = t + passo_temp #Incrementa a temperatura
    energia_media_ps.append(e_m/(l * l)) #Energia média por sítio
    calor_esp.append((e_m_q - (e_m * e_m))/( l * l * t * t)) #Calor específico por sítio
    m_m_ps.append((m_m)/(l * l)) #Magnetização média por sítio
    susceptibilidade_magnetica.append((m_m_q -(m_m * m_m))/(l * l * t)) #Susceptibilidade magnética por sítio
plt.title('magnetização média por sítio em funcão da tempertura')
plt.xlabel('T')
plt.ylabel('m/N')
plt.scatter(t_p, energia_media_ps)
plt.scatter(t_p, calor_esp)
plt.scatter(t_p,m_m_ps)
plt.scatter(t_p,susceptibilidade_magnetica)
plt.show()
#=====================================================================================================================