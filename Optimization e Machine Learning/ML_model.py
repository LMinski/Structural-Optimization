import pandas as pd
import statsmodels.api as sm
import numpy as np

data = pd.read_excel(f"output_10.xlsx")

data = data.drop(['Unnamed: 0', 'Material G0', 'Material G1', 'Material G2',
                  'Material G3', 'Material G4', 'Material G5', 'Material G6', 'Fx', 'desloc_x',
                  'Tension G0', 'Tension G1', 'Tension G2', 'Tension G3', 'Tension G4',
                  'Tension G5', 'Tension G6', 'Fy'], axis=1)
x = data.drop('desloc_y', axis = 1)
y = data['desloc_y']

x_train_sm = sm.add_constant(x)
lr = sm.OLS(y, x_train_sm).fit()
model = lr

def trelica_ML(x):
    dy_adm = 0.01
    taxa = np.abs(model.predict(x)/dy_adm)
    volume = x*np.array([0, 1, 1, 1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2)])
    vol = np.sum(volume)
    if taxa > 1:
        cost = vol + 10**8*taxa
    
    return cost