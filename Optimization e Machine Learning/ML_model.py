import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from scipy.optimize import differential_evolution, shgo, dual_annealing

data = pd.read_excel(f"C:\\Users\\Leonardo\\Desktop\\Projetos Python\\Pesquisa\\Optimization e Machine Learning\\output_100.xlsx")

data = data.drop(['Unnamed: 0', 'Material G0', 'Material G1', 'Material G2',
                  'Material G3', 'Material G4', 'Material G5', 'Material G6', 'Fx', 'desloc_x',
                  'Tension G0', 'Tension G1', 'Tension G2', 'Tension G3', 'Tension G4',
                  'Tension G5', 'Tension G6', 'Fy'], axis=1)
x = data.drop('desloc_y', axis=1).values
y = data['desloc_y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) 
lr = LinearRegression().fit(x_train, y_train)
model = lr

def trelica_ML(x):
    x = x.reshape(1, -1)  # Reshape input to have shape (1, n_features)
    dy_adm = 0.01
    taxa = np.abs(model.predict(x) / dy_adm)
    volume = x * np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2)])
    vol = np.sum(volume)
    if taxa > 1:
        cost = vol + 10 ** 8 * taxa
    else:
        cost = vol
    
    return cost

bounds = [(0.000235, 0.000631)] * 7  # List of bounds for each variable

# Use one of the optimization methods:
# result1 = differential_evolution(trelica_ML, bounds)
# result2 = shgo(trelica_ML, bounds)aa
#result3 = dual_annealing(trelica_ML, bounds)