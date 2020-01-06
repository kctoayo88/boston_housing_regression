import pandas as pd
import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

model = Sequential()
model.add(Dense(13, activation = 'relu', input_shape = (13,)))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1))
optimizer = Adam()
model.compile(loss='mse', optimizer = optimizer)
model.summary()

for step in range(100):
    value = model.fit(X_train, y_train, verbose = 0, batch_size = 100, epochs = 500, validation_split = 0.1)
    if step % 1 == 0:
        print('Epochs:', (step + 1) * 500)
        print('Training Loss:', value.history['loss'][-1], ' ', 'Val Loss:', value.history['val_loss'][-1])

results = model.evaluate(X_test, y_test)
print(results)
model.save('./trained_model.h5')

pred_y_test = model.predict(X_test)

plt.figure()
plt.plot(range(len(y_test)), y_test, ls='-.', lw=2, c='r')
plt.plot(range(len(pred_y_test)), pred_y_test, ls='-', lw=2, c='b')

plt.legend()
plt.xlabel('number')
plt.ylabel('people')

plt.show()