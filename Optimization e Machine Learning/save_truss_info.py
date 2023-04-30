import pandas as pd
import numpy as np
import sqlite3 as sq


def save_truss_info(data, secoes, material, group, conec, x, y, GDL_rest, forcas, no_critico, pp, gravidade, gama_d, n):
    filename = 'output_' + str(n) + '.xlsx'

    df_sec = pd.DataFrame(secoes, index=list(range(1, secoes.shape[0]+1)), columns = ["A", "b", "t", "Ix", "Iy", "rx", "ry", "rz_min", "wdt", "J", "W", "x", "s4g"])
    df_mat = pd.DataFrame(material, index=list(range(1, material.shape[0]+1)), columns = ["E", "fy_k", "density"])
    df_gru = pd.DataFrame(group, index=list(range(1, group.shape[0]+1)), columns=list(range(1, group.shape[1]+1)))
    coord = np.array([x,y])
    df_coo = pd.DataFrame(coord, index=['x', 'y'], columns=list(range(1, x.shape[0]+1)))
    df_con = pd.DataFrame(conec[:,-2:], index=list(range(1, conec.shape[0]+1)), columns=['node 1', 'node 2'])
    df_res = pd.DataFrame(GDL_rest, index=list(range(1, GDL_rest.shape[0]+1)), columns=['node', 'rest x', 'rest y'])
    df_for = pd.DataFrame(forcas, index=list(range(1, forcas.shape[0]+1)), columns=['node', 'Fx', 'Fy'])
    df_inf = pd.DataFrame([pp, gravidade, gama_d, no_critico], index=['self weight coef', 'gravity', 'fy_k reduction', 'observed node'])

    with pd.ExcelWriter(filename) as writer:
        data.to_excel(writer, sheet_name="analysis")

        # Salva os materiais e seção na database
        df_sec.to_excel(writer, sheet_name="sections")
        df_mat.to_excel(writer, sheet_name="materials")

        # Salva os grupos
        df_gru.to_excel(writer, sheet_name="groups")
        
        # Salva as coordenadas e conectividades
        df_coo.to_excel(writer, sheet_name="coord")
        df_con.to_excel(writer, sheet_name="elements")
        
        # Salva os GDL restringidos
        df_res.to_excel(writer, sheet_name="GDL_rest")
        
        # Salva as forças
        df_for.to_excel(writer, sheet_name="ext forces")
        
        # Salva informações
        df_inf.to_excel(writer, sheet_name="info")
    

    ## SALVA EM UMA DATABASE SQLITE3
    filename = 'output_' + str(n) + '.db'
    conn = sq.connect(filename) # creates file

    data.to_sql(name="analysis", con=conn, if_exists='replace')

    # Salva os materiais e seção na database
    df_sec.to_sql(name="sections", con=conn, if_exists='replace')
    df_mat.to_sql(name="materials", con=conn, if_exists='replace')

    # Salva os grupos
    df_gru.to_sql(name="groups", con=conn, if_exists='replace')
    
    # Salva as coordenadas e conectividades
    df_coo.to_sql(name="coord", con=conn, if_exists='replace')
    df_con.to_sql(name="elements", con=conn, if_exists='replace')
    
    # Salva os GDL restringidos
    df_res.to_sql(name="GDL_rest", con=conn, if_exists='replace')
    
    # Salva as forças
    df_for.to_sql(name="ext forces", con=conn, if_exists='replace')
    
    # Salva informações
    df_inf.to_sql(name="info", con=conn, if_exists='replace')

    conn.close() # good practice: close connection
