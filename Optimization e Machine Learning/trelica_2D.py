import numpy as np
import random
import time

from self_weight import self_weight
from FEM import FEM
from set_up_datatable import set_up_datatable
from utils import group_results, vary_groups, convert_time
from plot_truss import plot_truss
from save_truss_info import save_truss_info
from secoes import secoes

def trelica(secao_grupo, plotagem=False, salva_excel=False, n=1, display_time=False, var_sec=False, var_mat=False, var_Fx=False, var_Fy=False):

    """
    CONFIGURE STRUCTURE
    """
    # Montagem da estrutura
    # Nós
    x = np.array([0, 0, 1, 1, 2], dtype='float64')
    y = np.array([0, 1, 0, 1, 0], dtype='float64')

    # Escala da deformação (só para visualização do plot)
    scale = 5

    # Deslocamento admissível
    desloc_adm_x = 0.005  # m
    desloc_adm_y = 0.01  # m
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

    if len(secao_grupo) > 0:
        for idx, s in enumerate(secao_grupo):
            prop_group[idx][0] = s

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
        
        if display_time:
            # Calculate reamining time
            porcentagem = (_/n)*100
            end = time.time()
            if porcentagem % 20 == 0:
                duration = end - start
                total_time = duration * n
                elapsed_time = duration * _
                remaining_time = total_time - elapsed_time
                
                print(f'{porcentagem:3.0f}% completed:  ###  '
                      f'Elapsed time {convert_time(elapsed_time)} -- '  
                      f'Reamining time {convert_time(remaining_time)} -- '
                      f'Predicted time {convert_time(total_time)}')


    """
    SALVA OS DADOS EM EXCEL
    """
    if salva_excel:
        save_truss_info(data, secoes, material, group, conec, x, y, GDL_rest, forcas, no_critico, pp, gravidade, gama_d, n)
        

    """
    PLOTA A ÚLTIMA ESTRUTURA ANALISADA
    """
    if plotagem:
        plot_truss(F, sigma, forcas, x, y, conec, desloc, scale, desloc_adm_x, desloc_adm_y, prop_group, material, gama_d)

    """
    RETORNA O CUSTO DA ESTRUTURA
    """
    cost = 0
    for el in range(n_el):
        # Comprimento "L" do elemento "el"
        no1 = conec[el][-2]-1
        no2 = conec[el][-1]-1

        L = np.sqrt((x[no2] - x[no1])**2 + (y[no2] - y[no1])**2)

        # Propriedades
        s = prop_group[conec[el][1]-1][0]
        A = secoes[s-1][0]

        V = L*A*10**3 # Volume do elemento (cm**3)

        cost = cost + V

    taxa_dy = abs(dy/desloc_adm_y)
    if taxa_dy > 1:
        cost = cost + taxa_dy*10**2

    return cost

# print(trelica([1, 2, 3, 4, 5, 6, 7], True, False))
