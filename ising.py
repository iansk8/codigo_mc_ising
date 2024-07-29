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
from math import exp
#============================================================================================

#============================================================================================
#                                DEFININDO PARÂMETROS

L = 3 #dimensão da rede 
num_dir = 4
T = 2.7
mcs = 2 #passo monte carlo
range_temp = 20 #faixa de temperatura
passo_temp = - 0.1
#============================================================================================

def inicialize_rede(L): #Cria rede de spins aleatórios
    matriz = []

    for _ in range(L):
        matriz.append([1 for i in range(L)]) # Para rede aleatória trocar 1 por: np.random.choice((1, -1)) for i in range(L)
    
    return matriz


def metropolis(mcs, matriz, T, L):
    for i in range(mcs): #Loop sobre o Passo Monte Carlo
        for j in range(L * L): #Loop sobre os sites
            delta_E, y, x = calcular_variação_energia(matriz)
            if delta_E > 0:
                q = np.random.random() #Número aleatório entre 0 e 1
                prob = np.exp(-delta_E/ T)
                if q < prob:
                    spin = matriz[y][x] 
                    spin_flip = spin * -1
                    matriz[y][x] = spin_flip
                print(f'q = {q} e prob = {prob}')
            else:
                spin = matriz[y][x] 
                spin_flip = spin * -1
                matriz[y][x] = spin_flip
            print(j)
            print(f'Delta_E = {delta_E}')
            print(f'site = ({y}, {x})')
            print(matriz)
    #return matriz


def calcular_variação_energia(matriz): 
    sum_vizinhos, y, x, _ = vizinho(matriz, L)
    variacao_energia = 2 * matriz[y][x] * sum_vizinhos
    return variacao_energia, y, x
    

def vizinho(matriz, L, y, x): #Condições de contorno periódica
    vizinhos = []
    y = np.random.choice(L) #Site aleatório
    x = np.random.choice(L) #Site aleatório

    tamanho = len(matriz)

    vizinhos.append(matriz[y][(x+1) % tamanho]) #primeiro vizinho da direita
    vizinhos.append(matriz[y-1][x]) # primeiro vizinho decima
    vizinhos.append(matriz[y][x-1]) # primeiro vizinho da esquerda
    vizinhos.append(matriz[(y+1) % tamanho][x]) #primeiro vizinho debaixo
    #print(vizinhos)
    sum_vizinhos = 0
    for _ in range(num_dir): #Soma sobre os vizinhos
        sum_vizinhos += vizinhos[_] 
        #print(n_vizinhos)
    return sum_vizinhos, y, x, vizinhos


def energia_tot(matriz, L):
    energia_total = 0
    for y in range(L):
        for x in range(L):
            for _ in range(num_dir):
                _, _, _, primeiro_vizinho = vizinho(matriz, L, y, x)
                energia_total += (matriz[y][x] * primeiro_vizinho[_])
    return energia_total


#===================================================================================================================
#                                          PROGRAMA PRINCIPAL
matriz = np.array(inicialize_rede(L))
print('Matriz inicial: ', matriz)
energia_total = energia_tot(matriz, L)
metropolis(mcs, matriz, T, L) #Executa o Algoritmo de Metropolis para a equilibração do sistema
for k in range(range_temp): #Loop sobre as temperaturas
    metropolis(mcs, matriz, T, L) #Executa o algoritmo para calcular as propriedades
    if (T <= 2.4) and (T > 2.25):
        T -= 0.01
    else:
        T += passo_temp
#=====================================================================================================================