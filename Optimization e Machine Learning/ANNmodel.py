from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_excel(f"C:\\Users\\Leonardo\\Desktop\\Projetos Python\\Pesquisa\\Optimization e Machine Learning\\output_1000.xlsx")

data['desloc_y'] = data['desloc_y']
data = data.drop(['Unnamed: 0','Material G0', 'Material G1', 'Material G2',
       'Material G3', 'Material G4', 'Material G5', 'Material G6', 'Fx','desloc_x', 'Tension G0', 'Tension G1', 'Tension G2',
       'Tension G3', 'Tension G4', 'Tension G5', 'Tension G6', 'Fy'], axis = 1)
x = data.drop('desloc_y',axis = 1).values
y = data['desloc_y']*-1
y = y.values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 42)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(7,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation='linear')
])

model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
history = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2, callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])


def trelica_ANN(x):   
    x = np.reshape(x, (1, 7)) 
    dy_adm = 0.01
    taxa = np.abs(model.predict(x)/dy_adm)
    volume = x*np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2)])
    vol = np.sum(volume)
    if taxa > 1:
        cost = vol + 10**8*taxa
    else:
        cost = vol
        
    return cost
