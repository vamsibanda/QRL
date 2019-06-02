import pickle
import time
import numpy as np
import argparse
import re

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir, hparams_helper



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  hparam = hparams_helper()
  parser.add_argument('-e', '--episode', type=int, default=hparam['episodes'], help='number of episode to run')
  parser.add_argument('-b', '--batch_size', type=int, default=hparam['batch_size'], help='batch size for experience replay')
  parser.add_argument('-i', '--initial_invest', type=int, default=hparam['initial_investment'], help='initial investment amount')
  parser.add_argument('-m', '--mode', type=str, default=hparam['training_mode'], help='either "train" or "test"')
  parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
  args = parser.parse_args()

  timestamp = time.strftime('%Y%m%d%H%M')

  data = np.around(get_data())
  train_data = data[:, :4000]
  test_data = data[:, 4001:]

  env = TradingEnv(train_data, args.initial_invest)
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  portfolio_value = []

  if args.mode == 'test':
    # remake the env with test data
    env = TradingEnv(test_data, args.initial_invest)
    # load trained weights
    agent.load(args.weights)
    # when test, the timestamp is same as time when weights was trained
    timestamp = re.findall(r'\d{12}', args.weights)[0]

  for e in range(args.episode):
    state = env._reset()
    state = scaler.transform([state])
    for time in range(env.n_step):
      action = agent.act(state)
      next_state, reward, done, info = env._step(action)
      next_state = scaler.transform([next_state])
      if args.mode == 'train':
        agent.remember(state, action, reward, next_state, done)
      state = next_state
      if done:
        print("episode: {}/{}, episode end value: {}".format(e + 1, args.episode, info['cur_val']))

        f = open(timestamp + ".txt", "a")
        f.write("episode: {}/{}, episode end value: {}".format(e + 1, args.episode, info['cur_val']) + "\n")
        f.close()

        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        break
      if args.mode == 'train' and len(agent.memory) > args.batch_size:
        agent.replay(args.batch_size)
    if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
      agent.save('{}-dqn.h5'.format(timestamp))

  # save portfolio value history to disk
  with open('{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    pickle.dump(portfolio_value, fp)