import numpy as np
from numpy.linalg import inv

def FEM(x, y, conec, prop_group, secoes, material, forcas, GDL_rest):
    """
    Calcula a resposta estrutural da estrutura.

    Parameters:
            x (np.float array of shape (n_nos,)):
                Coordinates x of nodes
            y (np.float array of shape (n_nos,)):
                Coordinates y of nodes
            conec (np.int array of shape (n_el, 4)): 
                [elemento, grupo,  nó_1,  nó_2]
                
            prop_group (np.int array of shape (n_group, 2)):
                Seção e material adotados em cada grupo
                
            secoes (np.float array of shape (n_sec, 13)):
                [Área, b, t, Ix, Iy, rx, ry, rz_min, wdt, J, W, x, s4g]
            material (np.float array of shape (n_mat, 3)):
                [Young_modulus, fy_k, density]
                
            forcas (np.float array of shape (n_forcas, 3)):
                [nó, Fx, Fy]
                
            GDL_rest (np.int array of shape (n_rest, 3)):
                [nó, rest_x, rest_y]
            
    Returns:
            desloc (np.float array of shape (n_nos*2,)):
                displacements of nodes
            fn (np.float array of shape(n_el,)):
                axial force in the elements
            ten (np.float array of shape(n_el,))
                tension in the elements
            reacoes (np.float array of shape (n_nos*2,)):
                reactions on the nodes
    """

    no = np.arange(1, len(x)+1)  # numeração dos nós
    n_nos = no[-1]  # número de nós
    n_el = conec.shape[0]
    n_forcas = forcas.shape[0]
    n_rest = GDL_rest.shape[0]  # número de nós restringidos

    # Cálculo da estrutura
    GDL = 2*n_nos  # graus de liberdade da estrutura
    K = np.zeros((GDL, GDL))  # matriz rigidez global

    # Cálculo da matriz de cada elemento
    for el in range(n_el):
        # Comprimento do elemento "el"
        no1 = conec[el][-2]-1
        no2 = conec[el][-1]-1

        L = np.sqrt((x[no2] - x[no1])**2 + (y[no2] - y[no1])**2)

        # Propriedades
        s = prop_group[conec[el][1]-1][0]
        m = prop_group[conec[el][1]-1][1]
        A = secoes[s-1][0]
        E = material[m-1][0]

        # Cossenos diretores a partir das coordenadas dos nós do elemento
        cs = (x[no2] - x[no1])/L  # cosseno
        sn = (y[no2] - y[no1])/L  # seno

        # Matriz de rigidez do elemento "el"
        k = E*A/L
        ke = np.array([
            [k, -k],
            [-k, k]
        ])

        T = np.array([
            [cs, sn, 0, 0],
            [0, 0, cs, sn]
        ])

        kg = T.transpose() @ ke @ T

        # Inserção na matriz de rigidez global
        for i in range(2):  # superposição da sub-matriz (1-2,1-2) da matriz elementar
            ig = (no1)*2+i
            for j in range(2):
                jg = (no1)*2+j
                K[ig][jg] = K[ig][jg] + kg[i][j]

        for i in range(2):  # superposição da sub-matriz (3-4,3-4) da matriz elementar
            ig = (no2)*2+i
            for j in range(2):
                jg = (no2)*2+j
                K[ig][jg] = K[ig][jg] + kg[i+2][j+2]

        for i in range(2):  # superposição das sub-matrizes (1-2,3-4) e ((3-4,1-2) da matriz elementar
            ig = (no1)*2+i
            for j in range(2):
                jg = (no2)*2+j
                K[ig][jg] = K[ig][jg] + kg[i][j+2]
                K[jg][ig] = K[jg][ig] + kg[j+2][i]

    # Vetor de forças Global
    F = np.zeros((GDL, 1))
    for i in range(n_forcas):
        F[2*int(forcas[i][0])-2] = forcas[i][1]
        F[2*int(forcas[i][0])-1] = forcas[i][2]

    # guardamos os originais de K e F
    Kg = np.copy(K)
    Fg = np.copy(F)

    # Aplicar Restrições (condições de contorno)
    for k in range(n_rest):
        # Verifica se há restrição na direção x
        if GDL_rest[k][1] == 1:
            j = 2*GDL_rest[k][0]-2

            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j][i] = 0  # zera linha
                Kg[i][j] = 0  # zera coluna

            Kg[j][j] = 1     # valor unitário na diagonal principal
            Fg[j] = 0

        # Verifica se há restrição na direção y
        if GDL_rest[k][2] == 1:
            j = 2*GDL_rest[k][0]-1

            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j][i] = 0   # zera linha
                Kg[i][j] = 0   # zera coluna

            Kg[j][j] = 1     # valor unitário na diagonal principal
            Fg[j] = 0

    # Cálculo dos deslocamentos
    desloc = inv(Kg) @ Fg  # Vetor dos deslocamentos
    # print(f'Vetor deslocamentos: {desloc.flatten()}')

    # Reações
    reacoes = K @ desloc  # Vetor reações
    # print(f'Vetor reações: {reacoes.flatten()}')

    # Esforços nos elementos
    fn = np.zeros(n_el)
    ten = np.zeros(n_el)
    for el in range(n_el):
        # cálculo do comprimento do elemento "el"
        no1 = conec[el][-2]-1
        no2 = conec[el][-1]-1

        L = np.sqrt((x[no2] - x[no1])**2 + (y[no2] - y[no1])**2)

        # Propriedades
        s = prop_group[conec[el][1]-1][0]
        m = prop_group[conec[el][1]-1][1]
        A = secoes[s-1][0]
        E = material[m-1][0]

        # Cossenos diretores a partir das coordenadas dos nós do elemento
        cs = (x[no2] - x[no1])/L    # cosseno
        sn = (y[no2] - y[no1])/L    # seno

        # pega os valores dos deslocamentos dos nós do elemento "el"
        u1 = desloc[(no1+1)*2-2]
        u2 = desloc[(no2+1)*2-2]
        v1 = desloc[(no1+1)*2-1]
        v2 = desloc[(no2+1)*2-1]

        # constante de rigidez do elemento "el"
        k = E*A/L

        # força e tensão atuante no elemento "el"
        # cálculo da força normal no elemento
        fn[el] = k*(-(u1-u2)*cs - (v1-v2)*sn)
        # cálculo da tensão normal no elemento
        ten[el] = fn[el]/A
    return desloc, fn, ten, reacoes