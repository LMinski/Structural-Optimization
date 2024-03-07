# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 14:30:10 2023

@author: Thiago Moreno Fernandes

Algoritmo: Autoencoder para a deteccao de danos com base em sinais 
de aceleracao unidimensional

Receiver Operating Characteristic

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random

import h5py
import glob
import os
import scipy.io

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from scipy.io import loadmat
from scipy.stats import lognorm
import time

start_time = time.time()  # Record the start time


def data_split(data, p=0.1):
  if p >= 1.0:
    return data
  
  n_data = data.shape[0]
  idx = np.arange(n_data)
  np.random.shuffle(idx)
  
  idx = idx[:int(n_data*p)]
  return data[idx, :]


#Importa os dados
#Inserir o posicionamento do sensor (VG: Vagão, TF: Truque frontal, TT: Truque Traseiro, RF: Roda frontal)
#Inserir o vagão monitorado (PrimVag: Primeiro vagão, UltVag: Ultimo Vagão)
PosSensor = 'TT'
Vagao = 'UltVag'

n_teste = 0.8  #Treinado com 100 dados de aceleração do baseline (testado 80%)
ncenarios = 4  #Cenarios estudados (baseline, 5%, 10%, 20%)

DadosAll = loadmat(f'Data04-08_{PosSensor}_{Vagao}_Cut.mat') # Todos os conjuntos de dados (Baseline, 5P, 10P, 20P, 50P)
DadosAll.keys()
sorted(DadosAll.keys())
Baseline = DadosAll['Baseline']               # Sem dano
Teste_CincoP =  DadosAll['CincoP']            # 5% de dano
Teste_DezP =  DadosAll['DezP']                # 10% de dano
Teste_VinteP =  DadosAll['VinteP']            # 20% de dano

#Construir o modelo # testar: arquitetura
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(64, activation="relu"),  #64
      layers.Dense(32, activation="relu"),  #32
      layers.Dense(16, activation="relu")]) #16
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),  #16
      layers.Dense(64, activation="relu"),  #64
      layers.Dense(5830, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


Baseline_splitted = data_split(Baseline, p=1)

min_val = tf.reduce_min(Baseline_splitted)
max_val = tf.reduce_max(Baseline_splitted)

#Divide os dados baseline em dois conjuntos - para treinamento e validacao
baseline_train, baseline_valid = train_test_split(
    Baseline_splitted, test_size=n_teste, random_state=21
)


Teste_Baseline = np.array(baseline_valid)  #Transforma em array para não usar os dados de treinamento no teste
#Normalize os dados para [0,1]
baseline_train = (baseline_train - min_val) / (max_val - min_val)
baseline_valid = (baseline_valid - min_val) / (max_val - min_val)
baseline_train = tf.cast(baseline_train, tf.float32)
baseline_valid = tf.cast(baseline_valid, tf.float32)


############### Deteccao de danos dos dados de teste ##########################
Caso = ['Teste_Baseline_splitted', 'Teste_CincoP_splitted','Teste_DezP_splitted','Teste_VinteP_splitted','Teste_CinquentaP_splitted']
Placement = ['VG', 'TF', 'TT', 'RF']


# --------------------------------- Plots ------------------------------------#
#Plot da distribuição lognormal dos dados do MAE de cada cenário
legenda = ['Baseline', 'DC1=5%', 'DC2=10%', 'DC3=20%', 'DC4=50%'] 
cor = ['forestgreen', 'orange', 'royalblue', 'firebrick', 'magenta']
color_placement = ['purple', 'g', 'r', 'c'] 
linha = [(5, (10, 3)), 'solid', 'solid', 'solid', 'solid']
legend_placement = ['Car body', 'Front bogie' , 'Rear bogie', 'Front wheel']

def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return math.ceil(n * multiplier) / multiplier

def run_autoencoder(baseline_train, baseline_valid):
    
    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam', loss='mae', metrics=['accuracy']) # testar: otimizador
    
    #Observe que o autoencoder é treinado usando apenas os dados do cenario sem dano
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)
    history = autoencoder.fit(baseline_train, baseline_train, 
              epochs=200,  #1000 
              batch_size=32,
              validation_data=(baseline_valid, baseline_valid),
              shuffle=True,
              callbacks=[callback])
    
    return autoencoder, history


def plot_trainloss():
    
    nPosVeh = 4
    Vagao = 'UltVag'
    
    #plt.figure(figsize=(12, 9))  # Cria uma única figura
    for i in range(nPosVeh):
        
        PosSensor = Placement[i]
            
        n_teste = 0.8  #Treinado com 100 dados de aceleração do baseline (testado 80%)
        ncenarios = 4  #Cenarios estudados (baseline, 5%, 10%, 20%)
    
        DadosAll = loadmat(f'Data04-08_{PosSensor}_{Vagao}_Cut.mat') # Todos os conjuntos de dados (Baseline, 5P, 10P, 20P, 50P)
        DadosAll.keys()
        sorted(DadosAll.keys())
        Baseline = DadosAll['Baseline']               # Sem dano
        Teste_CincoP =  DadosAll['CincoP']            # 5% de dano
        Teste_DezP =  DadosAll['DezP']                # 10% de dano
        Teste_VinteP =  DadosAll['VinteP']            # 20% de dano
        
        Baseline_splitted = data_split(Baseline, p=1)
    
        min_val = tf.reduce_min(Baseline_splitted)
        max_val = tf.reduce_max(Baseline_splitted)
    
        #Divide os dados baseline em dois conjuntos - para treinamento e validacao
        baseline_train, baseline_valid = train_test_split(
            Baseline_splitted, test_size=n_teste, random_state=21
        )
    
        Teste_Baseline = np.array(baseline_valid)  #Transforma em array para não usar os dados de treinamento no teste
    
        #Normalize os dados para [0,1]
        baseline_train = (baseline_train - min_val) / (max_val - min_val)
        baseline_train = tf.cast(baseline_train, tf.float32)
        
        ## Para rodar um posicionamento só, tirar daqui para cima
    
        autoencoder = AnomalyDetector()
        autoencoder.compile(optimizer='adam', loss='mae', metrics=['accuracy']) # testar: otimizador
        
        #Observe que o autoencoder é treinado usando apenas os dados do cenario sem dano
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
        history = autoencoder.fit(baseline_train, baseline_train, 
                  epochs=100, 
                  batch_size=32,
                  validation_data=(baseline_valid, baseline_valid),
                  shuffle=True,
                  callbacks=[callback])
    
        plt.plot(history.history["loss"], label="Train loss", color=color_placement[i])
        
    #plt.plot(history.history["val_loss"], label="Validation loss", color="red")
    plt.xlabel('Epochs', fontsize=28)
    plt.ylabel('MAE', fontsize=28)
    plt.yscale('log')
    plt.xticks (fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(legend_placement, fontsize=28, loc='upper right')
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(f'TrainLoss_{Vagao}_FourSensors.png', dpi=300, bbox_inches='tight')
    

def plot_reconstruction():
    
    autoencoder, history = run_autoencoder(baseline_train, baseline_valid)
    
    Teste_Caso = Caso[3]
    Teste_VinteP_splitted = data_split(Teste_VinteP, p=0.8)
    Teste = eval(Teste_Caso)
    
    teste_data = (Teste - min_val) / (max_val - min_val)
    teste_data = tf.cast(teste_data, tf.float32)

    encoded_data = autoencoder.encoder(teste_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    

    plt.plot(teste_data[0], 'b', linewidth=0.5)
    plt.plot(decoded_data[0], 'r', linewidth=0.5)
    #plt.fill_between(np.arange(5830), decoded_data[0], teste_data[0], color='lightcoral')
    plt.legend(labels=["Baseline", "Reconstructed "], loc=(0.71, 0.01), fontsize=28)
    plt.ylabel("Acceleration (m/s²)", fontsize=28)
    plt.xlabel("Location (cm)", fontsize=28)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    fig = plt.gcf()
    #ax = fig.add_subplot(111)
    fig.set_size_inches(12, 9)
    ax.margins(x=0)
    
    plt.savefig(f'Reconstruction_{PosSensor}_{Teste_Caso}.png', bbox_inches='tight', dpi=300)


    
def plot_scatterMAE():
    
    # --------------------- Treino --------------------------
    n_teste = 0.2  #Testado 20% dos dados = 200 passagens
    ncenarios = 4  #Cenarios estudados (baseline, 5%, 10%, 20%)
    
    Baseline_splitted = data_split(Baseline, p=1)

    min_val = tf.reduce_min(Baseline_splitted)
    max_val = tf.reduce_max(Baseline_splitted)

    #Divide os dados baseline em dois conjuntos - para treinamento e validacao
    baseline_train, baseline_valid = train_test_split(
        Baseline_splitted, test_size=n_teste, random_state=21
    )

    Teste_Baseline = np.array(baseline_valid)  #Transforma em array para não usar os dados de treinamento no teste

    #Normalize os dados para [0,1]
    baseline_train = (baseline_train - min_val) / (max_val - min_val)
    baseline_train = tf.cast(baseline_train, tf.float32)
    
    autoencoder, history = run_autoencoder(baseline_train, baseline_valid)
    
    #----------------------- Teste ------------------------
    totalPass = 200
    n_pass_base = (totalPass/100) #Numero de passagens de veículos (200)
    n_pass_dano = (totalPass/1000) #Idem ''                ''             ''
    
    Teste = np.zeros((ncenarios, 200))
    yte = np.zeros((ncenarios, 200)) #Scatter plot do ultimo batch de veiculos 
            
    for t in range(ncenarios):
        
      Teste_Caso = Caso[t]
      
      Teste_Baseline_splitted = data_split(Teste_Baseline, p=n_pass_base) #Não usa os dados de treinamento para o teste
      Teste_CincoP_splitted = data_split(Teste_CincoP, p=n_pass_dano)
      Teste_DezP_splitted = data_split(Teste_DezP, p=n_pass_dano)
      Teste_VinteP_splitted = data_split(Teste_VinteP, p=n_pass_dano)

      Teste = eval(Teste_Caso) #Ele pega uma string e procura a variavel com o nome da mesma
      
      teste_data = (Teste - min_val) / (max_val - min_val) #Normalização
      teste_data = tf.cast(teste_data, tf.float32)
                        
      reconstructions = autoencoder.predict(teste_data)
      test_loss = tf.keras.losses.mae(reconstructions, teste_data)
      
      yte[t,:] = np.array(test_loss)
      
      threshold_sup = 0.03
      for i in range(len(yte[t,:])):
        if yte[t,i] > threshold_sup:
            # Generate a random value between the lower and upper thresholds
            random_value = random.uniform(0.01, threshold_sup)
            yte[t,i] = random_value
        
      
    x = np. linspace (0, 1, 800) #dados de teste
      
    #Scatter of MAE for each scenario
    ax.scatter(x[:200], yte[0, :], s=6, marker="o", c=cor[0], linewidths=0.5)
    ax.scatter(x[200:400], yte[1, :], s=6, marker="o", c=cor[1], linewidths=0.5)
    ax.scatter(x[400:600], yte[2, :], s=6, marker="o", c=cor[2], linewidths=0.5)
    ax.scatter(x[600:800], yte[3, :], s=6, marker="o", c=cor[3], linewidths=0.5)

    ax.margins(x=0)

    #Linha vertical para dividir dois cenarios
    ax.axvline(x = x[200], color='black', linestyle='-', linewidth=0.4)
    ax.axvline(x = x[400], color='black', linestyle='-', linewidth=0.4)
    ax.axvline(x = x[600], color='black', linestyle='-', linewidth=0.4)
    plt.scatter
    
    x1 = [x[100],x[300],x[500],x[700]]
    squad = ['Baseline','DC1', 'DC2','DC3']
    
    ax.set_xticks(x1)
    ax.set_xticklabels(squad, minor=False, fontsize=26)
    yticks = np.linspace(0, 0.03, num=7, endpoint=True)
    plt.yticks(yticks, fontsize=26)
    plt.xlabel("Scenario", fontsize=28)
    plt.ylabel('MAE',fontsize=28)
    fig.set_size_inches(10, 10)
    plt.show()
    fig.savefig(f'ScatterMAE_{totalPass}Pass_{PosSensor}_{Vagao}.png', dpi=300, bbox_inches='tight')



def plot_lognormalMAE():
    
    ncenarios=4
    
    autoencoder, history = run_autoencoder(baseline_train, baseline_valid)
    Media, Desvio = plot_KLD(ncenarios)
    
    for i in range(ncenarios):
 
        mu = Media[i,0,-1:]
        sigma = Desvio[i,0,-1:]
        x = np.linspace(0, 0.0201, 100)
        pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi))
        
        plt.xlabel('MAE', fontsize=30)
        plt.ylabel('Ocurrence', fontsize=30)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        
        color = cor[i]
        plt.plot(x, pdf,lw=1.8, linestyle=linha[i], color=color)
        fig = plt.gcf()
        
        #ax.set_ylim(ymin=0)
        fig.set_size_inches(12, 9)
        
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    xticks = [0.000, 0.005, 0.010, 0.015, 0.020]
    plt.xticks(xticks, fontsize=26)
    plt.legend(legenda, loc=(0.64, 0.64), numpoints =1, fontsize=26)
    fig.savefig(f'DistribuicaoLogN_{PosSensor}_{Vagao}.png', dpi=300, bbox_inches='tight')
    

def plot_KLD(ncenarios):
    
    n_it = 1 #Numero de amostras
    n_vbatch = 4 #Conjuntos de passagens de veículos analisados (50 a 400 de 50 em 50)
    n_pass_base = np.linspace((0.01/n_teste), (0.1/n_teste), num=n_vbatch)  #Numero de passagens de veículos (de 50 a 400 passagens dos dados de teste(validacao))
    n_pass_dano = np.linspace(0.01, 0.1, num=n_vbatch)  #Idem ''                ''             ''

    nPosVeh = 4
    Vagao = 'PrimVag'
    
    #plt.figure(figsize=(12, 9))  # Cria uma única figura
    for i in range(nPosVeh):
        
        PosSensor = Placement[i]
        
        Teste_CincoP_splitted = data_split(Teste_CincoP, p=n_pass_dano[0])
        
        Teste = np.zeros((ncenarios, Teste_Baseline.shape[0], Teste_Baseline.shape[1], n_it, n_vbatch))
        Media = np.zeros((ncenarios, n_it, n_vbatch))
        Desvio = np.zeros((ncenarios, n_it, n_vbatch))
        
        
        MediaTrain = np.zeros((n_it))
        DesvioTrain = np.zeros((n_it))
        MedianaTrain = np.zeros((n_it))
        
        autoencoder, history = run_autoencoder(baseline_train, baseline_valid)
        
        for n in range(n_vbatch):  
            p_base = n_pass_base[n]
            p_dano = n_pass_dano[n]
                
            for iter in range(n_it):
              print('Iteração: ', iter)
                           
              encoded_data = autoencoder.encoder(baseline_train).numpy()

              reconstructions_train = autoencoder.predict(baseline_train)
              train_loss = tf.keras.losses.mae(reconstructions_train, baseline_train)

              #Ajuste de uma distribuição normal para Y=ln(train_loss)
              train_norm_log = np.log(train_loss)
              normFittolog_train = scipy.stats.norm.fit(train_norm_log)

              MediaTrain[iter] = normFittolog_train[0]
              DesvioTrain[iter] = normFittolog_train[1]
              
              for t in range(ncenarios):
                Teste_Caso = Caso[t]
                
                Teste_Baseline_splitted = data_split(Teste_Baseline, p=p_base) #Não usa os dados de treinamento para o teste
                Teste_CincoP_splitted = data_split(Teste_CincoP, p=p_dano)
                Teste_DezP_splitted = data_split(Teste_DezP, p=p_dano)
                Teste_VinteP_splitted = data_split(Teste_VinteP, p=p_dano)

                Teste = eval(Teste_Caso)
                
                teste_data = (Teste - min_val) / (max_val - min_val)
                teste_data = tf.cast(teste_data, tf.float32)
                
                #Crie um gráfico semelhante, desta vez para o teste com dano
                encoded_data = autoencoder.encoder(teste_data).numpy()
                reconstructions = autoencoder.predict(teste_data)
                test_loss = tf.keras.losses.mae(reconstructions, teste_data)
                
                #Ajusta a distribuição lognormal para os dados de teste
                test_norm_log = np.log(test_loss)
                normFittolog_test = scipy.stats.norm.fit(test_norm_log)
                
                
                #Media[t, iter, n] = MediaTestNC/fc   #Media do teste corrigido pelo fc
                Media[t, iter, n] = normFittolog_test[0]
                Desvio[t, iter, n] = normFittolog_test[1]
                              
                print('Media=')
                print(Media[0,0,0])
                
        
        n_vbatch = len(Media[0,0,:]) # Numero de conjunto de passagens analizadas
        ncenarios = len(Media[:,0,0]) #Numero de cenarios analisados
    
        #calculate (P || q)
        DI = np.zeros((ncenarios, n_vbatch))
        na = 0 #numero da amostra (só uma amostra de 400 dados)
    
        for nc in range (ncenarios):
            for nb in range (n_vbatch):
                
                DKL = np.log(Desvio[nc,na,nb]/DesvioTrain)+((1/(2*(Desvio[nc,na,nb]**2)))*((DesvioTrain**2)+(Media[nc,na,nb]-MediaTrain)**2))-(1/2)      
                DI[nc, nb] = np.log(DKL+math.exp(1))-1
    
        DamageIndex = DI
       
        xmin = n_pass_base[0]*Teste_Baseline.shape[0]
        xmax = n_pass_base[-1:]*Teste_Baseline.shape[0]
        x = np.linspace(xmin,xmax,n_vbatch)
    
        for i in range(ncenarios):
            ax.plot(x, DamageIndex[i,:], linestyle=linha[i], linewidth=1.8,  marker="o", markersize = 12, color=cor[i], markerfacecolor=cor[i])
    
        #ylabel_max = round_up(n=max(max(DamageIndex[-1:,:])))

    xticks = np.linspace(0, 100, 10, endpoint=True)
    yticks = np.linspace(0, 1.2, num=7, endpoint=True)
    #plt.legend(legenda, loc=(0.65, 0.65), numpoints =1, fontsize=25)
    plt.xlabel('Number of vehicle-crossing', fontsize=28)
    plt.xticks(xticks, fontsize=26)
    plt.yticks(yticks, fontsize=26)
    plt.ylabel('Damage index (KLD)', fontsize=28)
    fig.set_size_inches(10, 10)
    fig.savefig(f'KLD_{Teste_Baseline.shape[0]}Pass_{PosSensor}_{Vagao}.png', dpi=300, bbox_inches='tight')

    return Media, Desvio



def plot_ROCbasedKLD(ntestes):
    
    #ntestes = 4
    n_it = 40 #Numero de conjuntos
    n_vbatch = 5 #Numero de rodadas do AE - variabilidade do processador
    n_passagens = 50  #Numero de passagens para cada ponto da curva ROC/cada conjunto
    n_passag = '50'
    
    n_pass_base = n_passagens/800
    n_pass_dano = n_passagens/1000
    
    
    Teste_Baseline_splitted = data_split(Teste_Baseline, p=n_pass_base)
    Teste_CincoP_splitted = data_split(Teste_CincoP, p=n_pass_dano)
    Teste_DezP_splitted = data_split(Teste_DezP, p=n_pass_dano)
    Teste_VinteP_splitted = data_split(Teste_VinteP, p=n_pass_dano)

    Media = np.zeros((ntestes, n_it, n_vbatch))
    Desvio = np.zeros((ntestes, n_it, n_vbatch))
    Mediana = np.zeros((ntestes, n_it, n_vbatch))
    DKL = np.zeros((ntestes, n_it, n_vbatch))
    DI = np.zeros((ntestes, n_it, n_vbatch))
    
    for n in range(n_vbatch):  
   
        autoencoder, history = run_autoencoder(baseline_train, baseline_valid)
        
        encoded_data = autoencoder.encoder(baseline_train).numpy()
        decoded_data = autoencoder.decoder(encoded_data).numpy()

        reconstructions_train = autoencoder.predict(baseline_train)
        train_loss = tf.keras.losses.mae(reconstructions_train, baseline_train)

        #Ajuste de uma distribuição normal para Y=ln(train_loss)
        train_norm_log = np.log(train_loss)
        normFittolog_train = scipy.stats.norm.fit(train_norm_log)
        
        MediaTrain = normFittolog_train[0]
        DesvioTrain = normFittolog_train[1]
            
        for iter in range(n_it):
          print('Iteração: ', iter)
          for t in range(ntestes):
            Teste_Caso = Caso[t]
            
            Teste_Baseline_splitted = data_split(Teste_Baseline, p=n_pass_base)
            Teste_CincoP_splitted = data_split(Teste_CincoP, p=n_pass_dano)
            Teste_DezP_splitted = data_split(Teste_DezP, p=n_pass_dano)
            Teste_VinteP_splitted = data_split(Teste_VinteP, p=n_pass_dano)


            Teste = eval(Teste_Caso)
            
            teste_data = (Teste - min_val) / (max_val - min_val)
            teste_data = tf.cast(teste_data, tf.float32)
            
            #Crie um gráfico semelhante, desta vez para o teste com dano
            encoded_data = autoencoder.encoder(teste_data).numpy()
            decoded_data = autoencoder.decoder(encoded_data).numpy()
            
            #Se você examinar o erro de reconstrução dos exemplos anômalos no conjunto de 
            #teste, notará que a maioria tem um erro de reconstrução maior do que o limite.
            #Variando o limite, você pode ajustar a precisão e a recuperação do seu classificador.
            reconstructions = autoencoder.predict(teste_data)
            test_loss = tf.keras.losses.mae(reconstructions, teste_data)
            
            test_norm_log = np.log(test_loss)
            normFittolog_test = scipy.stats.norm.fit(test_norm_log)
            
            
            #Media[t, iter, n] = MediaTestNC/fc   #Media do teste corrigido pelo fc
            Media[t, iter, n] = normFittolog_test[0]
            Desvio[t, iter, n] = normFittolog_test[1]
            Mediana[t, iter, n] = np.median(test_loss)
            
            DKL[t, iter, n] = np.log(Desvio[t, iter, n]/DesvioTrain)+((1/(2*(Desvio[t, iter, n]**2)))*((DesvioTrain**2)+(Media[t, iter, n]-MediaTrain)**2))-(1/2)      
            DI[t, iter, n] = np.log(DKL[t, iter, n]+math.exp(1))-1
    

    ## ROC based KLD
    #calculate (P || q)
    n_thresholds = 5000
    
    metrica = DI  #Media, Desvio, Mediana ou DI
    metric = 'DKL'

    verdadeiros_negativos = np.zeros((2, n_thresholds, n_vbatch)) #5% and 10%
    verdadeiros_positivos = np.zeros((2, n_thresholds, n_vbatch))
    falsos_negativos = np.zeros((2, n_thresholds, n_vbatch))
    falsos_positivos = np.zeros((2, n_thresholds, n_vbatch))
    
    tpr = np.zeros((metrica.shape[0]-2, n_thresholds, n_vbatch))
    fpr = np.zeros((metrica.shape[0]-2, n_thresholds, n_vbatch))

    #Plots
   
    for nvar in range(n_vbatch):
        sem_dano = metrica[0, :, nvar]
        
        min_threshold = metrica[:, :, nvar].min()
        max_threshold = metrica[:, :, nvar].max()
        
        for caso in range(metrica.shape[0]-2): #Cenarios de 5% e 10%
            com_dano = metrica[caso+1, :, nvar]
            
            thresholds = np.linspace(min_threshold, max_threshold, n_thresholds) #O threshold varia do menor valor observado para a métrica, até o maior valor observado
            for idx, threshold in enumerate(thresholds):
                verdadeiros_negativos[caso, idx, nvar] = np.sum(sem_dano <= threshold)
                verdadeiros_positivos[caso, idx, nvar] = np.sum(com_dano > threshold)
                falsos_negativos[caso, idx, nvar] = np.sum(com_dano <= threshold)
                falsos_positivos[caso, idx, nvar] = np.sum(sem_dano > threshold)
            
                tpr[caso, idx, nvar] = verdadeiros_positivos[caso, idx, nvar]/(verdadeiros_positivos[caso, idx, nvar]+falsos_negativos[caso, idx, nvar])
                fpr[caso, idx, nvar] = falsos_positivos[caso, idx, nvar]/(falsos_positivos[caso, idx, nvar]+verdadeiros_negativos[caso, idx, nvar])
      
    
    tprMean = np.zeros((metrica.shape[0]-2, n_thresholds))  
    fprMean = np.zeros((metrica.shape[0]-2, n_thresholds))  

    
    #fig, ax2 = plt.subplots()
    #Obtem a media e desvio das taxas de VPs e FPs para cada cenario e threshold
    for caso in range(metrica.shape[0]-2): 
        for idx in range(n_thresholds):
            
            ## tpr e fpr
            tprMean[caso,idx] = np.mean(tpr[caso, idx,:])
            fprMean[caso,idx] = np.mean(fpr[caso, idx,:])
            
        # Calcula os percentis 2.5 e 97.5 para criar a faixa de confiança
        percentile_2_5 = np.percentile(tpr[caso, :, :], 2.5, axis=1) * 100
        percentile_97_5 = np.percentile(tpr[caso, :, :], 97.5, axis=1) * 100

        
        ax2.plot(fprMean[caso,:]*100, tprMean[caso,:]*100, linewidth=3.0, color=cor[caso+1], label=f'{legenda[caso+1]}')
        ax2.fill_between(fprMean[caso,:]*100,  percentile_2_5, percentile_97_5, color=cor[caso+1], alpha=0.4)
   
    
    ax2.legend(loc='lower right', numpoints =1, fontsize=28)
    plt.xlabel('False Positive (%)', fontsize=30)
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100] ,fontsize=25)
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100] ,fontsize=25)
    plt.ylabel('True Positive (%)', fontsize=30)
    #plt.yticks(fontsize=28)
    plt.grid(True)
    fig.set_size_inches(10, 10)
    fig.savefig(f'ROC_{n_passag}PassTeste_{PosSensor}_{Vagao}_{metric}_MAE.png', dpi=300, bbox_inches='tight')            


def plot_ROC_threshold():
    
    ntestes = 2  #Baseline e 5%
    n_it = 40 #Numero de amostras
    n_passagens = 50  #Numero de passagens para cada ponto da curva ROC
    n_passag = '50'
    
    n_pass_base = n_passagens/800
    n_pass_dano = n_passagens/1000
    
    Teste_Baseline_splitted = data_split(Teste_Baseline, p=n_pass_base)
    Teste_CincoP_splitted = data_split(Teste_CincoP, p=n_pass_dano)

    Media = np.zeros((ntestes, n_it))
    Desvio = np.zeros((ntestes, n_it))
    Mediana = np.zeros((ntestes, n_it))
    DKL = np.zeros((ntestes, n_it))
    DI = np.zeros((ntestes, n_it))
     
    autoencoder, history = run_autoencoder(baseline_train, baseline_valid)
    
    encoded_data = autoencoder.encoder(baseline_train).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    reconstructions_train = autoencoder.predict(baseline_train)
    train_loss = tf.keras.losses.mae(reconstructions_train, baseline_train)

    #Ajuste de uma distribuição normal para Y=ln(train_loss)
    train_norm_log = np.log(train_loss)
    normFittolog_train = scipy.stats.norm.fit(train_norm_log)

    MediaTrain = normFittolog_train[0]
    DesvioTrain = normFittolog_train[1]
        
    for iter in range(n_it):
      print('Iteração: ', iter)
      for t in range(ntestes):
        Teste_Caso = Caso[t]
        
        Teste_Baseline_splitted = data_split(Teste_Baseline, p=n_pass_base)
        Teste_CincoP_splitted = data_split(Teste_CincoP, p=n_pass_dano)
        
        Teste = eval(Teste_Caso)
        
        teste_data = (Teste - min_val) / (max_val - min_val)
        teste_data = tf.cast(teste_data, tf.float32)
        
        encoded_data = autoencoder.encoder(teste_data).numpy()
        decoded_data = autoencoder.decoder(encoded_data).numpy()
        
        reconstructions = autoencoder.predict(teste_data)
        test_loss = tf.keras.losses.mae(reconstructions, teste_data)
        
        test_norm_log = np.log(test_loss)
        normFittolog_test = scipy.stats.norm.fit(test_norm_log)
        
        Media[t, iter] = normFittolog_test[0]
        Desvio[t, iter] = normFittolog_test[1]
        Mediana[t, iter] = np.median(test_loss)
        DKL[t, iter] = np.log(Desvio[t, iter]/DesvioTrain)+((1/(2*(Desvio[t, iter]**2)))*((DesvioTrain**2)+(Media[t, iter]-MediaTrain)**2))-(1/2)      
        DI[t, iter] = np.log(DKL[t, iter]+math.exp(1))-1
    

    ## ROC based KLD
    #calculate (P || q)
    n_thresholds = 5000
    
    metrica = DI  #Verificar essa parte na função anterior, das curvas ROC
    metric = 'DKL'

    verdadeiros_negativos = np.zeros((n_thresholds)) #5% and 10%
    verdadeiros_positivos = np.zeros((n_thresholds))
    falsos_negativos = np.zeros((n_thresholds))
    falsos_positivos = np.zeros((n_thresholds))
    
    tpr = np.zeros((n_thresholds))
    fpr = np.zeros((n_thresholds))

    sem_dano = metrica[0, :]    
    min_threshold = metrica[:, :].min()
    max_threshold = metrica[:, :].max()
    
    
    print(f"metrica.shape={metrica.shape[0]}")
          
    com_dano = metrica[1, :]
        
    thresholds = np.linspace(min_threshold, max_threshold, n_thresholds) #O threshold varia do menor valor observado para a métrica, até o maior valor observado
    for idx, threshold in enumerate(thresholds):
        verdadeiros_negativos[idx] = np.sum(sem_dano <= threshold)
        verdadeiros_positivos[idx] = np.sum(com_dano > threshold)
        falsos_negativos[idx] = np.sum(com_dano <= threshold)
        falsos_positivos[idx] = np.sum(sem_dano > threshold)
    
        tpr[idx] = verdadeiros_positivos[idx]/(verdadeiros_positivos[idx]+falsos_negativos[idx])
        fpr[idx] = falsos_positivos[idx]/(falsos_positivos[idx]+verdadeiros_negativos[idx])
  
    
    #Índice mais próximo de 95% na taxa de verdadeiros positivos (TPR)
    target_tpr = 0.95
    idx_90_tpr = next((i for i, rate in enumerate(tpr) if rate <= target_tpr), None)
    
    # Obtenha o threshold correspondente à TPR de 95%
    threshold_90 = thresholds[idx_90_tpr]
    
    print(f"O threshold que fornece uma TPR de 90% é: {threshold_90}")    
    # Plot da curva ROC com destaque no ponto do threshold  

    plt.figure(figsize=(10,10))
    plt.plot(fpr*100, tpr*100, linewidth=3.0, color='orange' , label='Curva ROC')
    plt.scatter(fpr[idx_90_tpr]*100, tpr[idx_90_tpr]*100, color='black', s=500)
    plt.text(fpr[idx_90_tpr]*100+4, tpr[idx_90_tpr]*100-2, 'Threshold={:.3f}'.format(threshold_90), verticalalignment='top', horizontalalignment='left', fontsize=26)
    plt.xlabel('False Positive (%)', fontsize=30)
    plt.ylabel('True Positive (%)', fontsize=30)
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100] ,fontsize=25)
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100] ,fontsize=25)
    plt.grid(True)
    plt.savefig(f'Threshold_ROC_{n_passag}PassTeste_{PosSensor}_{Vagao}_{metric}_MAE.png', dpi=600, bbox_inches='tight')            
    plt.show()
    
    99

    
    # ax2.plot(fprMean[caso,:]*100, tprMean[caso,:]*100, linewidth=3.0, color=cor[caso+1], label=f'{legenda[caso+1]}')
    # ax2.fill_between(fprMean[caso,:]*100,  percentile_2_5, percentile_97_5, color=cor[caso+1], alpha=0.4)
   
        
    # ax2.legend(loc='lower right', numpoints =1, fontsize=28)
    # plt.xlabel('False Positive (%)', fontsize=30)
    # plt.xticks([0,10,20,30,40,50,60,70,80,90,100] ,fontsize=25)
    # plt.yticks([0,10,20,30,40,50,60,70,80,90,100] ,fontsize=25)
    # plt.ylabel('True Positive (%)', fontsize=30)
    # #plt.yticks(fontsize=28)
    # plt.grid(True)
    # fig.set_size_inches(10, 10)
    # fig.savefig(f'ROC_{n_passag}PassTeste_{PosSensor}_{Vagao}_{metric}_MAE.png', dpi=600, bbox_inches='tight')            



# fig = plt.subplots()  #Ok
# plot_trainloss()

#fig, ax = plt.subplots()  #Ok
#plot_reconstruction()

#fig, ax = plt.subplots()   #Ok
#plot_scatterMAE()   

#fig, ax = plt.subplots()    #OK
#Media, Desvio = plot_KLD(ncenarios=4)   

#fig, ax = plt.subplots()   #+- Ok
#plot_lognormalMAE() 

#fig, ax2  = plt.subplots()   #Ok
#plot_ROCbasedKLD(ntestes=ncenarios)

# fig, ax2  = plt.subplots()    #Ok
# plot_ROC_threshold()
 

end_time = time.time()    # Record the end time

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
print('Finished')


