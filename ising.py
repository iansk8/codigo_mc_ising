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

inicio = time.time()
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
def metropolis(mcs, matriz, T, L):
    #cont_aceite = 0
    #cont_total = 0
    energia_total = 0
    for y in range(L):
        for x in range(L):
            for _ in range(num_dir):
                i , vizinhos = vizinho(matriz, L, y, x)
                energia_total += (matriz[y][x] * vizinhos[_])
    energia_total = -energia_total / 2
    energia_autal = energia_total
    sum_energia = 0
    sum_energia_quadrada = 0
    cont = 0
    magnetizçao = 0
    for i in range(L):
        for j in range(L):
            magnetizçao += matriz[i][j]
    magnetizaçao_atual = magnetizçao
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
                prob = np.exp(-delta_E/ T)
                if q < prob:
                    matriz[y][x] = (matriz[y][x]) * -1 
                    #cont_aceite += 1
                    energia_autal += delta_E
                    magnetizaçao_atual += delta_M
                #print(f'q = {q} e prob = {prob}')
            else:
                matriz[y][x] = (matriz[y][x]) * -1 
                #cont_aceite += 1
                energia_autal += delta_E
                magnetizaçao_atual += delta_M
            #cont_total += 1
            cont += 1
            sum_energia += energia_autal
            sum_energia_quadrada += energia_autal * energia_autal
            sum_magnetizaçao += magnetizaçao_atual
       # if (i % 500 == 0):
    e_m = sum_energia/cont
    e_m_q = sum_energia_quadrada/cont
    m_m = sum_magnetizaçao/cont
    """sum_energia_quadrada += energia_autal * energia_autal
    sum_magnetizaçao += magnetizaçao_atual
    sum_magnetizaçao_quadrada += magnetizaçao_atual * magnetizaçao_atual"""
    #energia_total = energia_autal
    return matriz, e_m, energia_autal, e_m_q, m_m, cont, sum_energia


@njit()
def calcular_variação_energia(matriz, L, y, x): 
    sum_vizinhos, _ = vizinho(matriz, L, y, x)
    variacao_energia = 2 * matriz[y][x] * sum_vizinhos
    return variacao_energia


@njit()
def calcular_variação_magnetizaçao(matriz, y, x):
    variação_magnetizaçao = -2 * matriz[y][x]
    return variação_magnetizaçao
    

@njit()
def vizinho(matriz, L, y, x): #Condições de contorno periódica
    vizinhos = []

    vizinhos.append(matriz[y][(x+1) % L]) #primeiro vizinho da direita
    vizinhos.append(matriz[y-1][x]) # primeiro vizinho de cima
    vizinhos.append(matriz[y][x-1]) # primeiro vizinho da esquerda
    vizinhos.append(matriz[(y+1) % L][x]) #primeiro vizinho de baixo
    #vizinhos = np.array(r)
    #print(vizinhos)
    sum_vizinhos = 0
    for _ in range(4): #Soma sobre os vizinhos
        sum_vizinhos += vizinhos[_] 
        #print(n_vizinhos)
    return sum_vizinhos, vizinhos


"""def energia_tot(matriz, L, num_dir):
    energia_total = 0
    for y in range(L):
        for x in range(L):
            for _ in range(num_dir):
                i , vizinhos = vizinho(matriz, L, y, x)
                energia_total += (matriz[y][x] * vizinhos[_])
    energia_total = -energia_total / 2 #Dividido por 2 dois porque cada sítio é contado duas vezes
    return energia_total"""


"""def magnetizçao_total(matriz, L):
    magnetizçao = 0
    for i in range(L):
        for j in range(L):
            magnetizçao += matriz[i][j]
    return magnetizçao
"""

"""#@njit
def resultados( cont, sum_energia, sum_energia_quadrada, L, T, sum_magnetizaçao, sum_magnetizaçao_quadrada):
    energia_media = sum_energia/cont
    energia_media_quadrada = sum_energia_quadrada/cont
    magnetizçao_media = sum_magnetizaçao/cont
    magnetizçao_media_quadrada = sum_magnetizaçao_quadrada/cont
    calor_especifico = (energia_media_quadrada - (energia_media * energia_media))/(L * L * T *T) #Calor específico por sítio
    susceptibilidade_magnetica = (magnetizçao_media_quadrada -(magnetizçao_media * magnetizçao_media))/(L * L * T) #Susceptibilidade magnética por sítio"""

#===================================================================================================================
#                                          PROGRAMA PRINCIPAL
inicio = time.time()
matriz = np.array(inicialize_rede(l))
#print(f"initialize demorou {time.time() - inicio:.2f} segundos")
#print(f'Matriz inicial: {matriz}')
inicio = time.time()
#energia_total = energia_tot(matriz, l, num_dir)
#print(f"energia_tot demorou {time.time() - inicio:.2f} segundos")
inicio = time.time()
#magnetizçao_t = magnetizçao_total(matriz, l) #tirar função desnecessária
#print(f"magnetizacao demorou {time.time() - inicio:.2f} segundos")
inicio = time.time()
matriz, _, _, _, _, _, _ = metropolis(mcs, matriz, t, l) #Executa o Algoritmo de Metropolis para a equilibração do sistema
print(f"metropolis demorou {time.time() - inicio:.2f} segundos")
matriz = np.array(matriz)
energia_media_ps = []
m_m_ps = []
calor_esp = []
t_p = []
inicio = time.time()
for k in range(range_temp): #Loop sobre as temperaturas
    t_p.append(t)
    matriz, e_m, energia_totall, e_m_q, m_m, cont, sum_e = metropolis(mcs, matriz, t, l) #Executa o algoritmo para calcular as propriedades
    matriz = np.array(matriz)
    e = sum_e/cont
    energia_total = energia_totall
    calor = ((e_m_q - (e_m * e_m))/( l * l * t * t))
    calor_esp.append(calor) #Calor específico por sítio
    t = t + passo_temp
    energia_media_ps.append(e/(l * l))
    m_m_ps.append((m_m)/(l * l))
print(energia_total)
fim = time.time()
print(fim - inicio)
plt.title('magnetização média por sítio em funcão da tempertura')
plt.xlabel('T')
plt.ylabel('m/N')
#plt.scatter(t_p, energia_media_ps)
#plt.scatter(t_p, calor_esp)
plt.scatter(t_p,m_m_ps)
plt.show()
#=====================================================================================================================