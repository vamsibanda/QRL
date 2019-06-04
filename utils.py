import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
  # Training
  episodes=1000,
  batch_size=32,
  initial_investment=50000,
  training_mode='train',

  # Eval
  testing_mode='test')

def hparams_helper():
  return hparams.values()

def get_data(col='close'):
    all_csv_files = glob.glob(os.path.curdir + '/' +"*.csv")
    data_array = None
    for csv_file in all_csv_files:
        if not isinstance(data_array,np.ndarray):
            data_array = np.array([pd.read_csv(csv_file)['close'].values[::-1]])
        else:
            data_array = np.append(data_array, [pd.read_csv(csv_file)['close'].values[::-1]],axis=0)
    return data_array


def get_scaler(env):
  """ Takes a env and returns a scaler for its observation space """
  low = [0] * (env.n_stock * 2 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 2 # double the initial investment
  max_stock_owned = max_cash // min_price
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler

def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
