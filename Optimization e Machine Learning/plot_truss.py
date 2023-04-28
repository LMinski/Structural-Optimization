import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from sklearn import preprocessing


def plot_truss(F, sigma, forcas, x, y, conec, desloc, scale, desloc_adm_x, desloc_adm_y, prop_group, material, gama_d):
    """
    PLOTA A ÚLTIMA ESTRUTURA ANALISADA
    """
    fn = F
    ten = sigma
    no = np.arange(1, len(x)+1)  # numeração dos nós
    n_nos = no[-1]  # número de nós
    n_forcas = forcas.shape[0] # número de forças
    n_el = conec.shape[0] # número de elementos

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
    for el in range(n_el):
        print(
            f'Elemento {el+1:2.0f} --> Força axial: {float(fn[el]):15.5f}; Tensão normal: {float(ten[el]):10.5f}')
        m = prop_group[conec[el][1]-1][1]
        fy_taxa = np.abs(ten[el])/(material[m-1][1]/gama_d)
        if fy_taxa > 1:
            falha = True
            print(
                f'ELEMENTO {conec[el][0]} FALHOU. ULTRAPASSOU A TENSÃO DE ESCOAMENTO EM {(fy_taxa-1)*100:4.1f}%')
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

    cmap = mpl.cm.bwr
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

        ax1.plot(x_el_deformed, y_el_deformed, linewidth=3, c=cm.bwr(idx))

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
    
    # Plota texto nos elementos
    for el in range(n_el):
        # Comprimento "L" do elemento "el"
        no1 = conec[el][-2]-1
        no2 = conec[el][-1]-1

        x_el_deformed = np.array(
            [(x_deformed[no1]+x[no1])/2, (x_deformed[no2]+x[no2])/2])
        y_el_deformed = np.array(
            [(y_deformed[no1]+y[no1])/2, (y_deformed[no2]+y[no2])/2])

        x_el = np.linspace(x_el_deformed[0], x_el_deformed[1], 8)
        y_el = np.linspace(y_el_deformed[0], y_el_deformed[1], 8)

        ax1.text(x_el[int(len(x_el)/2)], y_el[int(len(y_el)/2)], str(el), color='c')

    # Plota texto nos nós
    for no in range(n_nos):
        ax1.text(x_deformed[no], y_deformed[no], str(no), color='m')


    # Ajusta algumas configurações dos plots
    plt.subplots_adjust(bottom=0.1, right=0.9, left=0.1, top=0.9)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax2.set_xlabel('Esforço axial (N)')

    plt.grid(color='0.5', linestyle=':', linewidth=1)
    ax1.set_title(f'Treliça na configuração deformada: Escala {scale}')

    plt.show()