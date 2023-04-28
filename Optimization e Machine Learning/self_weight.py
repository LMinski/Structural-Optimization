import numpy as np

def self_weight(x, y, conec, prop_group, secoes, material, pp, gravidade, forcas, GDL_rest):
    """
    Adiciona o peso próprio, se houver.

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
            pp (np.float):
                self weight multiplier
            gravidade (np.float):
                gravity in m/s**2
            forcas (np.float array of shape (n_forcas, 3)):
                [nó, Fx, Fy]
            GDL_rest (np.int array of shape (n_rest, 3)):
                [nó, rest_x, rest_y]
            
    Returns:
            forcas (np.float array of shape (n_forcas, 3)):
                [nó, Fx, Fy]
    """
    
    no = np.arange(1, len(x)+1)  # numeração dos nós
    n_nos = no[-1]  # número de nós
    n_el = conec.shape[0]
    n_forcas = forcas.shape[0]
    n_rest = GDL_rest.shape[0]  # número de nós restringidos

    if pp > 0:
        peso_proprio = np.empty([1, 3])
        for el in range(n_el):
            s = prop_group[conec[el][1]-1][0]
            m = prop_group[conec[el][1]-1][1]
            A = secoes[s-1][0]
            density = material[m-1][2]

            no1 = conec[el][-2]-1
            no2 = conec[el][-1]-1

            L = np.sqrt((x[no2] - x[no1])**2 + (y[no2] - y[no1])**2)

            massa = L*A*density
            peso = massa*gravidade*pp

            if conec[el][3] in GDL_rest[:,0] or conec[el][4] in GDL_rest[:,0]:
                for i in range(n_rest):
                    if conec[el][3] == GDL_rest[i][0] and GDL_rest[i][2] == 1:
                        peso_proprio = np.concatenate(
                            (peso_proprio, np.array([[conec[el][4], 0, -peso]])), axis=0)
                        break
                    elif conec[el][4] == GDL_rest[i][0] and GDL_rest[i][2] == 1:
                        peso_proprio = np.concatenate(
                            (peso_proprio, np.array([[conec[el][3], 0, -peso]])), axis=0)
                        break
            else:
                peso_proprio = np.concatenate((peso_proprio,
                                            np.array([[conec[el][3], 0, -peso/2],
                                                        [conec[el][4], 0, -peso/2]])),
                                            axis=0)
        peso_proprio = np.delete(peso_proprio, 0, 0)
        
        for p in range(peso_proprio.shape[0]):
            node_p = peso_proprio[p][0]
            
            if node_p in forcas[:,0]:
                for f in range(n_forcas):
                    node_f = forcas[f][0]
                    
                    if node_p == node_f:
                        forcas[f][2] = forcas[f][2] + peso_proprio[p][2]
                        min_node_f = n_nos+1
                        break
            else:
                forcas = np.concatenate((forcas, [peso_proprio[p]]), axis=0)
            n_forcas = forcas.shape[0]

    return forcas