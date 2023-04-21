import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from sklearn import preprocessing
import pandas as pd
import random
import time
from datetime import timedelta

from self_weight import self_weight
from FEM import FEM
from set_up_datatable import set_up_datatable
from utils import group_results, vary_groups

def trelica():

    """
    SET PARAMETERS
    """
    # Parâmetros
    var_sec = True
    var_mat = False
    var_Fx = False
    var_Fy = False
    n = 100 # number of times to run

    # Montagem da estrutura
    # Nós
    x = np.array([0, 0, 1, 1, 2], dtype='float64')
    y = np.array([0, 1, 0, 1, 0], dtype='float64')

    # Escala da deformação (só para visualização do plot)
    scale = 5

    # Deslocamento admissível
    desloc_adm_x = 0.004  # m
    desloc_adm_y = 0.004  # m
    no_critico = 5 # número do nó onde o desloc adm será observado

    # Materiais
    Young_modulus = [  # módulo de elasticidade
        200e9,
    ]

    fyk = [  # tensão de escoamento
        344.7e6,  # ASTM A572 G50
        413.6e6,  # ASTM A572 G60
    ]

    gama_d = 1.0 # fator de redução de fyk

    density = [
        7836.41
    ]

    # Multiplicador do peso próprio
    pp = 0  # 0 para não considerar o peso próprio da estrutura
    gravidade = 9.80665  # m/s**2

    material = np.array([
        [Young_modulus[0], fyk[0], density[0]],
        [Young_modulus[0], fyk[1], density[0]],
    ], dtype='float64')

    # Seções [número da seção,  área]
    secoes = np.array([
    #    Área      b      t      Ix        Iy        rx        ry        rz mín    wdt    J         W         x         s4g
        [2.35E-04, 0.04,  0.003, 3.45E-08, 3.45E-08, 1.21E-02, 1.21E-02, 7.80E-03, 10.33, 6.93E-10, 1.24E-06, 1.11E-02, 0], # 1 L40x40x3
        [3.08E-04, 0.04,  0.004, 4.61E-08, 4.61E-08, 1.21E-02, 1.21E-02, 7.80E-03, 7.5,   1.56E-09, 1.55E-06, 1.15E-02, 0], # 2 L40x40x4
        [3.79E-04, 0.04,  0.005, 5.43E-08, 5.43E-08, 1.20E-02, 1.20E-02, 7.70E-03, 5.8,   2.97E-09, 1.97E-06, 1.18E-02, 0], # 3 L40x40x5
        [2.66E-04, 0.045, 0.003, 4.93E-08, 4.93E-08, 1.36E-02, 1.36E-02, 8.80E-03, 11.67, 7.83E-10, 1.58E-06, 1.23E-02, 0], # 4 L45x45x3
        [3.49E-04, 0.045, 0.004, 6.67E-08, 6.67E-08, 1.36E-02, 1.36E-02, 8.70E-03, 8.5,   1.77E-09, 2.07E-06, 1.28E-02, 0], # 5 L45x45x4
        [4.30E-04, 0.045, 0.005, 7.84E-08, 7.84E-08, 1.35E-02, 1.35E-02, 8.70E-03, 6.6,   3.54E-09, 2.43E-06, 1.40E-02, 0], # 6 L45x45x5
        [2.96E-04, 0.05,  0.003, 7.15E-08, 7.15E-08, 1.52E-02, 1.52E-02, 9.90E-03, 13.33, 8.53E-10, 1.96E-06, 1.35E-02, 0], # 7 L50x50x3
        [3.89E-04, 0.05,  0.004, 9.26E-08, 9.26E-08, 1.52E-02, 1.52E-02, 9.80E-03, 9.75,  1.99E-09, 2.57E-06, 1.40E-02, 0], # 8 L50x50x4
        [4.80E-04, 0.05,  0.005, 1.10E-07, 1.10E-07, 1.51E-02, 1.51E-02, 9.70E-03, 7.6,   3.96E-09, 3.05E-06, 1.42E-02, 0], # 9 L50x50x5
        [4.71E-04, 0.06,  0.004, 1.63E-07, 1.63E-07, 1.83E-02, 1.83E-02, 1.18E-02, 12,    2.41E-09, 3.75E-06, 1.65E-02, 0], # 10 L60x60x4
        [5.82E-04, 0.06,  0.005, 1.99E-07, 1.99E-07, 1.82E-02, 1.82E-02, 1.17E-02, 9.4,   4.64E-09, 4.45E-06, 1.64E-02, 0], # 11 L60x60x5
        [5.13E-04, 0.065, 0.004, 2.09E-07, 2.09E-07, 1.98E-02, 1.98E-02, 1.28E-02, 13,    2.63E-09, 4.42E-06, 1.77E-02, 3.84E-02], # 12 L65x65x4
        [6.31E-04, 0.065, 0.005, 2.47E-07, 2.47E-07, 1.98E-02, 1.98E-02, 1.28E-02, 10.2,  5.06E-09, 5.20E-06, 1.77E-02, 3.84E-02], # 13 L65x65x5
    ], dtype='float64')

    # Elementos (conectividades) [elemento,   grupo,   nó_1,    nó_2]
    conec = np.array([
        [1, 1, 1, 3],
        [2, 2, 3, 4],
        [3, 3, 2, 4],
        [4, 4, 1, 4],
        [5, 5, 2, 3],
        [6, 6, 3, 5],
        [7, 7, 4, 5],
    ], dtype='int32')

    # Seção e material do grupo
    prop_group =  np.array([ # [seção, material]
        [13, 1],
        [13, 1],
        [13, 1],
        [13, 1],
        [13, 1],
        [13, 1],
        [13, 1],
    ], dtype='int32')

    # Carregamentos [nó,   intensidade_x,  intensidade_y]
    forcas = np.array([
        [5, 0, -1e5]
    ])

    # Apoios
    GDL_rest = np.array([  # [nó, restringido_x, restringido_y] (1 para restringido, e 0 para livre)
        [1, 1, 1],
        [2, 1, 1],
    ], dtype='int32')



    """
    VARIA AS PROPRIEDADES/FORÇAS
    E REALIZA A ANÁLISE
    """
    n_el = conec.shape[0]
    # set group variable
    n_groups = np.unique(conec[:, 1]).shape[0]
    group = np.zeros((n_groups, n_el), dtype='int32')
    selected_mat = np.zeros((n_groups), dtype='int32')
    selected_sec = np.zeros((n_groups), dtype='int32')
    for el in range(n_el):
        idx = conec[el][1]-1
        g = group[idx]
        group[idx][np.count_nonzero(g)] = el+1
        
        selected_mat[idx] = prop_group[idx][1]-1
        selected_sec[idx] = prop_group[idx][0]-1


    # Set up data table
    data = set_up_datatable(group)

    ### Varia as propriedades, forças, etc
    for _ in range(n):
        start = time.time()
        if var_Fx:
            fx = random.randint(-10**6, 10**6)  # em Newtons
            forcas[0][1] = fx
        else:
            fx = forcas[0][1]
            
        if var_Fy:
            fy = random.randint(-10**6, 10**6)  # em Newtons
            forcas[0][2] = fy
        else:
            fy = forcas[0][2]

        if var_mat:
            E_group, E, selected_mat = vary_groups(group, n_el, material, 0, [])
            fy_k_group, fy_k, selected_mat = vary_groups(group, n_el, material, 1, np.array([]))
        else:
            E_group, E, selected_mat = vary_groups(group, n_el, material, 0, selected_mat)
            fy_k_group, fy_k, selected_mat = vary_groups(group, n_el, material, 1, selected_mat)
            
        if var_sec:
            area_group, area, selected_sec = vary_groups(group, n_el, secoes, 0, np.array([]))
        else:
            area_group, area, selected_sec = vary_groups(group, n_el, secoes, 0, selected_sec)
        
        # Realiza análise
        prop_group[:,0] = selected_sec+1
        prop_group[:,1] = selected_mat+1
        forcas = self_weight(x, y, conec, prop_group, secoes, material, pp, gravidade, forcas, GDL_rest)
        desloc, F, sigma, reacoes = FEM(x, y, conec, prop_group, secoes, material, forcas, GDL_rest)
        
        # Agrupa resultados (se houverem diferentes grupos)
        F_group = group_results(F, group)
        sigma_group = group_results(sigma, group)

        # Deslocamento do nó mais crítico
        dx = desloc[2*(no_critico)-2]
        dy = desloc[2*(no_critico)-1]

        # Guarda na tabela 'data'
        data.loc[len(data)] = np.concatenate((area_group, E_group, np.array([fx]), np.array([fy]), dx, dy,
                                            sigma_group))
        
        end = time.time()
        porcentagem = (_/n)*100
        if porcentagem % 20 == 0:
            duration = end - start
            total_time = duration * n
            elapsed_time = duration * _
            remaining_time = total_time - elapsed_time
            
            print(f'Elapsed time (hh:mm:ss) {timedelta(seconds=elapsed_time)} --- '  
                f'Reamining time (hh:mm:ss) {timedelta(seconds=remaining_time)} --- '
                f'Total time (hh:mm:ss) {timedelta(seconds=total_time)}')


    """
    SALVA OS DADOS
    """
    # filename = 'output_' + str(n) + '.xlsx'
    # with pd.ExcelWriter(filename) as writer:
    #     data.to_excel(writer, sheet_name="analysis")

    #     # Salva os materiais e seção na database
    #     df_sec = pd.DataFrame(secoes, index=list(range(1, secoes.shape[0]+1)), columns = ["A", "b", "t", "Ix", "Iy", "rx", "ry", "rz_min", "wdt", "J", "W", "x", "s4g"])
    #     df_sec.to_excel(writer, sheet_name="sections")

    #     df_mat = pd.DataFrame(material, index=list(range(1, material.shape[0]+1)), columns = ["E", "fy_k", "density"])
    #     df_mat.to_excel(writer, sheet_name="materials")

    #     # Salva os grupos
    #     df_gru = pd.DataFrame(group, index=list(range(1, group.shape[0]+1)), columns=np.zeros((group.shape[1])))
    #     df_gru.to_excel(writer, sheet_name="groups")
        
    #     # Salva as coordenadas e conectividades
    #     coord = np.array([x,y])
    #     df_coo = pd.DataFrame(coord, index=['x', 'y'], columns=list(range(1, x.shape[0]+1)))
    #     df_coo.to_excel(writer, sheet_name="coord")
        
    #     df_con = pd.DataFrame(conec[:,-2:], index=list(range(1, conec.shape[0]+1)), columns=['node 1', 'node 2'])
    #     df_con.to_excel(writer, sheet_name="elements")
        
    #     # Salva os GDL restringidos
    #     df_res = pd.DataFrame(GDL_rest, index=list(range(1, GDL_rest.shape[0]+1)), columns=['node', 'rest x', 'rest y'])
    #     df_res.to_excel(writer, sheet_name="GDL_rest")
        
    #     # Salva as forças
    #     df_for = pd.DataFrame(forcas, index=list(range(1, forcas.shape[0]+1)), columns=['node', 'Fx', 'Fy'])
    #     df_for.to_excel(writer, sheet_name="ext forces")
        
    #     # Salva informações
    #     df_inf = pd.DataFrame([pp, gravidade, gama_d, no_critico], index=['self weight coef', 'gravity', 'fy_k reduction', 'observed node'])
    #     df_inf.to_excel(writer, sheet_name="info")
        


    """
    PLOTA A ÚLTIMA ESTRUTURA ANALISADA
    """
    fn = F
    ten = sigma
    no = np.arange(1, len(x)+1)  # numeração dos nós
    n_nos = no[-1]  # número de nós
    n_forcas = forcas.shape[0]
    print(f'Vetor de esforços: {fn}')
    print(f'Vetor de tensões:  {ten}')
    print(f'Esforço axial no elemento 2: {fn[1]} \n')

    print('===========================================')
    print('NÓ        UX                             UY')
    print('-------------------------------------------')
    x_deformed = np.copy(x)
    y_deformed = np.copy(y)
    falha = False
    for i in range(n_nos):
        x_deformed[i] = x[i] + desloc[2*(i+1)-2]*scale
        y_deformed[i] = y[i] + desloc[2*(i+1)-1]*scale
        print(
            f'Nó {i+1:2.0f} --> Deslocamento x: {float(desloc[2*(i+1)-2]):10.5f} m;  Deslocamento y: {float(desloc[2*(i+1)-1]):10.5f} m')

        desloc_taxa = np.array(
            [desloc[2*(i+1)-2]/desloc_adm_x, desloc[2*(i+1)-1]/desloc_adm_y])
        if np.abs(desloc_taxa[0][0]) > 1:
            falha = True
            print(
                f'NÓ {i+1} ULTRAPASSOU O DESLOCAMENTO MÁXIMO ADMISSÍVEL EM {np.abs(desloc_taxa[0][0]-1)*100:4.1f}%')

        if np.abs(desloc_taxa[1][0]) > 1:
            falha = True
            print(
                f'NÓ {i+1} ULTRAPASSOU O DESLOCAMENTO MÁXIMO ADMISSÍVEL EM {np.abs(desloc_taxa[1][0]-1)*100:4.1f}%')

    print('\n')

    print('-----------------------------------------------------------')
    print('ELEMENTO        FORÇA AXIAL                   TENSÃO NORMAL')
    print('-----------------------------------------------------------')
    for i in range(n_el):
        print(
            f'Elemento {i+1:2.0f} --> Força axial: {float(fn[i]):15.5f}; Tensão normal: {float(ten[i]):10.5f}')
        m = prop_group[conec[el][1]-1][1]
        fy_taxa = np.abs(ten[i])/(material[m-1][1]/gama_d)
        if fy_taxa > 1:
            falha = True
            print(
                f'ELEMENTO {conec[i][0]} FALHOU. ULTRAPASSOU A TENSÃO DE ESCOAMENTO EM {(fy_taxa-1)*100:4.1f}%')
    print('===========================================================')
    print('\n')

    if falha:
        print('A ESTRUTURA NÃO ESTÁ SEGURA!')

    # Plots
    plt.style.use('_mpl-gallery')
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 20}

    plt.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [16, 1]})

    cmap = mpl.cm.cool # MEXER AQUI PRA COR
    norm = mpl.colors.Normalize(vmin=min(fn), vmax=max(fn))
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')

    xmin = min(x)
    xmax = max(x)

    ymin = min(y)
    ymax = max(y)

    # Plota treliça
    find_nearest = lambda array, value: (np.abs(array - value)).argmin()
    for i in range(n_el):
        x_el = np.array(x[conec[i][-2]-1], x[conec[i][-1]-1])
        y_el = np.array(y[conec[i][-2]-1], y[conec[i][-1]-1])

        x_el_deformed = np.array(
            [x_deformed[conec[i][-2]-1], x_deformed[conec[i][-1]-1]])
        y_el_deformed = np.array(
            [y_deformed[conec[i][-2]-1], y_deformed[conec[i][-1]-1]])

        idx = find_nearest(np.array(cb1._values), fn[i])

        ax1.plot(x_el_deformed, y_el_deformed, linewidth=3, c=cm.cool(idx))

    # Plota forças (representadas por linhas tracejadas pretas)
    normalized = preprocessing.normalize([forcas[:,1:].flatten()])

    no1 = conec[-1][-2]-1
    no2 = conec[-1][-1]-1
    L = np.sqrt((x[no2] - x[no1])**2 + (y[no2] - y[no1])**2)

    for i in range(n_forcas):
        fx_i = x_deformed[int(forcas[i][0]-1)]
        fy_i = y_deformed[int(forcas[i][0]-1)]

        fx_size = normalized[0][i*2]*L/7
        fy_size = normalized[0][i*2+1]*L/7

        ax1.plot(np.array([fx_i, fx_i+fx_size]),
                 np.array([fy_i, fy_i]),
                 linewidth=2, c='k', linestyle='--')
        
        ax1.plot(np.array([fx_i, fx_i]),
                 np.array([fy_i, fy_i+fy_size]),
                 linewidth=2, c='k', linestyle='--')

    # Ajusta algumas configurações dos plots
    plt.subplots_adjust(bottom=0.1, right=0.9, left=0.1, top=0.9)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax2.set_xlabel('Esforço axial (N)')

    plt.grid(color='0.5', linestyle=':', linewidth=1)
    ax1.set_title(f'Treliça na configuração deformada: Escala {scale}')

    plt.show()

if __name__ == "__main__":
    trelica()

