import pandas as pd

def set_up_datatable(group):
    """Set up data table header."""

    # Monta a tabela de dados
    string_sec = []
    string_mat = []
    string_sigma = []
    # string_taxa_fy = []
    string_total = []
    for g in range(group.shape[0]):
        string_sec.append("Section G"+str(g))
        string_mat.append("Material G"+str(g))
        string_sigma.append("Tension G"+str(g))
        # string_taxa_fy.append("sigma/sigma_adm G"+str(g)) # > 1 indica falha
    string_total.extend(string_sec)
    string_total.extend(string_mat)
    string_total.extend(["Fx", "Fy", "desloc_x", "desloc_y"])
    string_total.extend(string_sigma)
    # string_total.extend(string_taxa_fy)
    data = pd.DataFrame(columns=(string_total))
    
    return data