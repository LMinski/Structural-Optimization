import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from sklearn import preprocessing


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Montagem da estrutura
# Nós
# x = np.array([3., 6., 6., 0.], dtype='float64')
# y = np.array([3., 3., 0., 0.], dtype='float64')

x = np.array([0, 0, 2, 2, 4], dtype='float64')
y = np.array([0, 2, 0, 2, 0], dtype='float64')

no = np.arange(1, len(x)+1)  # numeração dos nós
n_nos = no[-1]  # número de nós

# Escala da deformação (só para visualização do plot)
scale = 10

# Deslocamento admissível
desloc_adm_x = 0.004  # m
desloc_adm_y = 0.004  # m

# Materiais
Young_modulus = [  # módulo de elasticidade
    200e9,
]

fy = [  # tensão de escoamento
    344.7e6,  # ASTM A572 G50
    413.6e6,  # ASTM A572 G60
]

densidade = [
    7836.41
]

# Multiplicador do peso próprio
pp = 1.3  # 0 para não considerar o peso próprio da estrutura
gravidade = 9.80665  # m/s**2

material = np.array([
    [Young_modulus[0], fy[0], densidade[0]],
    [Young_modulus[0], fy[1], densidade[0]],
], dtype='float64')

# Elementos (conectividades) [elemento,   seção,   material,    nó_1,    nó_2]
# conec = np.array([
#     [1, 1, 1, 1, 2],
#     [2, 2, 1, 1, 3],
#     [3, 2, 1, 1, 4]
# ])

conec = np.array([
    [1, 13, 1, 1, 3],
    [2, 13, 1, 3, 4],
    [3, 13, 1, 2, 4],
    [4, 1,  1, 1, 4],
    [5, 1,  1, 2, 3],
    [6, 8,  1, 3, 5],
    [7, 8,  1, 4, 5],
])

n_el = conec.shape[0]

# Seções [número da seção,  área]
# secoes = np.array([
#     [1, 0.3e-3],
#     [2, 0.3*np.sqrt(2)*1e-3]
# ], dtype='float64')

secoes = np.array([
    [2.35E-04],  # 1 L40x40x3
    [3.08E-04],  # 2 L40x40x4
    [3.79E-04],  # 3 L40x40x5
    [2.66E-04],  # 4 L45x45x3
    [3.49E-04],  # 5 L45x45x4
    [4.30E-04],  # 6 L45x45x5
    [2.96E-04],  # 7 L50x50x3
    [3.89E-04],  # 8 L50x50x4
    [4.80E-04],  # 9 L50x50x5
    [4.71E-04],  # 10 L60x60x4
    [5.82E-04],  # 11 L60x60x5
    [5.13E-04],  # 12 L65x65x4
    [6.31E-04],  # 13 L65x65x5
], dtype='float64')

n_sec = secoes.shape[0]

# Carregamentos [nó,   intensidade_x,  intensidade_y]
# forcas = np.array([
#     [1, 0, -100e3]
# ])

forcas = np.array([
    [5, 0, -1e5]
])

n_forcas = forcas.shape[0]

# Apoios
# GDL_rest = np.array([  # [nó, restringido_x, restringido_y] (1 para restringido, e 0 para livre)
#     [2, 1, 1],
#     [3, 1, 1],
#     [4, 1, 1]
# ])

GDL_rest = np.array([  # [nó, restringido_x, restringido_y] (1 para restringido, e 0 para livre)
    [1, 1, 1],
    [2, 1, 1],
])

n_rest = GDL_rest.shape[0]  # número de nós restringidos

# Cálculo da estrutura
# Adiciona o peso próprio, se houver
if pp > 0:
    peso_proprio = np.empty([1, 3])
    for el in range(n_el):
        mat = conec[el][2]-1
        sec = conec[el][1]-1

        density = material[mat][2]
        A = secoes[sec][0]

        no1 = conec[el][3]-1
        no2 = conec[el][4]-1

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

GDL = 2*n_nos  # graus de liberdade da estrutura
K = np.zeros((GDL, GDL))  # matriz rigidez global

# Cálculo da matriz de cada elemento
for el in range(n_el):
    # Comprimento do elemento "el"
    no1 = conec[el][3]-1
    no2 = conec[el][4]-1

    L = np.sqrt((x[no2] - x[no1])**2 + (y[no2] - y[no1])**2)

    # Propriedades
    A = secoes[conec[el][1]-1][0]
    E = material[conec[el][2]-1][0]

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
print(f'Vetor deslocamentos: {desloc.flatten()}')

# Reações
reacoes = K @ desloc  # Vetor reações
print(f'Vetor reações: {reacoes.flatten()}')

# Esforços nos elementos
fn = np.zeros(n_el)
ten = np.zeros(n_el)
for el in range(n_el):
    # cálculo do comprimento do elemento "el"
    no1 = conec[el][3]-1
    no2 = conec[el][4]-1

    L = np.sqrt((x[no2] - x[no1])**2 + (y[no2] - y[no1])**2)

    # Propriedades
    A = secoes[conec[el][1]-1][0]
    E = material[conec[el][2]-1][0]

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

print(f'Vetor de esforços: {fn}')
print(f'Vetor de tensões:  {ten}')
print(f'Esforço axial no elemento 2: {fn[1]} \n')


print('NÓ        UX                             UY')
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
    if desloc_taxa[0] > 1:
        falha = True
        print(
            f'NÓ {i+1} ULTRAPASSOU O DESLOCAMENTO MÁXIMO ADMISSÍVEL EM {(desloc_taxa[0]-1)*100:4.1f}%')

    if desloc_taxa[1] > 1:
        falha = True
        print(
            f'NÓ {i+1} ULTRAPASSOU O DESLOCAMENTO MÁXIMO ADMISSÍVEL EM {(desloc_taxa[1]-1)*100:4.1f}%')

print('\n')

print('ELEMENTO        FORÇA AXIAL                   TENSÃO NORMAL')
for i in range(n_el):
    print(
        f'Elemento {i+1:2.0f} --> Força axial: {float(fn[i]):15.5f}; Tensão normal: {float(ten[i]):10.5f}')

    fy_taxa = np.abs(ten[i])/material[conec[i][2]][1]
    if fy_taxa > 1:
        falha = True
        print(
            f'ELEMENTO {conec[i][0]} FALHOU. ULTRAPASSOU A TENSÃO DE ESCOAMENTO EM {(fy_taxa-1)*100:4.1f}%')

if falha:
    print('A ESTRUTURA NÃO ESTÁ SEGURA!')

# Plots
plt.style.use('_mpl-gallery')
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 20}

plt.rc('font', **font)

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [16, 1]})

cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=min(fn), vmax=max(fn))
cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                norm=norm,
                                orientation='vertical')

xmin = min(x)
xmax = max(x)

ymin = min(y)
ymax = max(y)

# Plota treliça
for i in range(n_el):
    x_el = np.array(x[conec[i][3]-1], x[conec[i][4]-1])
    y_el = np.array(y[conec[i][3]-1], y[conec[i][4]-1])

    x_el_deformed = np.array(
        [x_deformed[conec[i][3]-1], x_deformed[conec[i][4]-1]])
    y_el_deformed = np.array(
        [y_deformed[conec[i][3]-1], y_deformed[conec[i][4]-1]])

    idx = find_nearest(cb1._values, fn[i])

    ax1.plot(x_el_deformed, y_el_deformed, linewidth=3, c=cm.cool(idx))

# Plota forças (representadas por linhas tracejadas pretas)
normalized = preprocessing.normalize([forcas[:,1:].flatten()])
for i in range(n_forcas):
    fx_i = x_deformed[int(forcas[i][0]-1)]
    fy_i = y_deformed[int(forcas[i][0]-1)]

    fx_size = normalized[0][i*2]*L/7
    fy_size = normalized[0][i*2+1]*L/7

    ax1.plot(np.array([fx_i, fx_i+fx_size]), np.array([fy_i,
             fy_i+fy_size]), linewidth=2, c='k', linestyle='--')

# Ajusta algumas configurações dos plots
plt.subplots_adjust(bottom=0.05, right=0.95, left=0.05, top=0.95)

ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax2.set_xlabel('Esforço axial (N)')

plt.grid(color='0.5', linestyle=':', linewidth=1)
ax1.set_title(f'Treliça na configuração deformada: Escala {scale}')

plt.show()
