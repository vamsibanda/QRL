from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam



def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
  model = Sequential()
  model.add(Dense(128, activation='relu', input_dim=n_obs))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(27, activation='linear'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model