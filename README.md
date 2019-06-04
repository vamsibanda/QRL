
Deep Reinforcement Learning - Final Project
Team Members: Noah Caldwell-Gatsos, Vamsi Banda

For our final project for Deep Reinforcement Learning, we decided to simulate a FOREX (foreign currency exchange) trading environment and train an agent to make a decision based on the state of the environment.

High Level Overview of Concepts
Our project makes use of Deep Q-learning - for any finite Markov Decision-Making process, it finds an optimal policy to maximize the reward over time, starting from the current state. A Markov Decision-Making Process consists of 1) states, 2) actions, and 3) rewards, all of which are included in our environment. The action state for our model is simple, there are only three options available for the agent to take 1) buy, 2) sell, or 3) hold. The number of actions for any given timestep is 3^n, where n is the number of stocks in our portfolio.

FOREX trading is a bit different from stock trading. The volume of transactions is vastly divergent, with FOREX markets trading around 5 trillion per day, while the stock market trades in around 200 billion per day. FOREX markets are highly liquid, run 24 hours a day, have little to no commissions for each trad, and have a narrow focus (there are really only eight major options to chose from, where the stock market has thousands of options. FOREX is also traded in pairs (one currency for another), so it is important for invstors to look for diverging and converging trends between the currencies to match up a pair to trade. There are a number of variables that affect the major currencies, but we won't focus on those for the purposes of this model.

Model Overview
extract.py = extracts FOREX data from the API 
agent.py = a deep Q-learning agent used to exploit the environment
envs.py = a three-stock trading environment 
model.py = multi-layer network used as the function approximator 

data/
All of our data is derived from Alpha Vantage, a free API for FOREX data that can be found here

Step One - Data Extraction using extract.py
Background: Remember that FOREX data is conducted in pairs. When you want to trade in one, or make predictions based on past data, you need to look up both the currency you'd like to trade AND the currency you'd like to recieve.

Below is our .fetch_data() function that takes in the following as its arguments:

functional_value is the time series of your choice: FX_DAILY, FX_WEEKLY, FX_MONTHLY
conversion_from is the input currency you'd like to trade
conversion_to is the output currency you'd like to recieve
out_size is meh
key is the unique ID for the API
format_type is the document type (e.g. '.csv')
Fetch data essentially takes in all of the above as its inputs, creates the remote URL call based on the inputs, reads in the associated .csv into a Pandas dataframe, and storing the result as a new .csv file in our folder.

def fetch_data(functional_value,conversion_from, conversion_to, out_size, key, format_type):
    
    url = base_url
    url = url + function + functional_value
    url = url + from_symbol + conversion_from
    url = url + to_symbol + conversion_to
    url = url + outputsize + out_size
    url = url + apikey + key
    url = url + datatype + format_type
    
    r = requests.get(url, allow_redirects=True)
    df = None
    if r.ok:
       data = r.content.decode('utf8')
       df = pd.read_csv(io.StringIO(data))
    else:
        print('Issue with the url - ', url)
    
    if isinstance(df, pd.DataFrame):
       df.to_csv(conversion_from+'_'+conversion_to+'.csv',index = False, header=True)
    else:
       print('Currently only csv format is supported')
Step Two - Creating the Environment w/ envs.py
Class TradingEnv is a three currency package trading environment. 

The state is the number of currency packages owned, the current package prices, and the cash in hand of the agent. NOTE: The agent doesn't know what the current environment looks like. It has to go through the exploratory phase that will be explained in Step Three. 

In this class, we have three different action spaces that the agent can choose:

(0) = Sell, in this use case, the agent sells all the currency packages.
(1) = Hold, in this use case, the agent doesn't make any changes to the environment in buying or selling. Essentially, a neutral option.
(2) = Buy, the agent tries to purchase the maximum number of packages of currency that are available given the amount of cash on hand.

class TradingEnv():
    
    def __init__(self, train_data, init_invest=20000):
        # data
        self.stock_price_history = np.around(train_data) # round up to integer to reduce state space
        self.n_stock, self.n_step = self.stock_price_history.shape
        # instance attributes
        
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # action space
        self.action_space = spaces.Discrete(3**self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        price_range = [[0, mx] for mx in stock_max_price]
        cash_in_hand_range = [[0, init_invest * 2]]
        self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

        # seed and start
        self._seed()
        self._reset()
    
    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        action_vec = action_combo[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
          if a == 0:
            sell_index.append(i)
          elif a == 2:
            buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
          for i in sell_index:
            self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
            self.stock_owned[i] = 0
        if buy_index:
          can_buy = True
          while can_buy:
            for i in buy_index:
              if self.cash_in_hand > self.stock_price[i]:
                self.stock_owned[i] += 1 # buy one share
                self.cash_in_hand -= self.stock_price[i]
              else:
                can_buy = False
Step Three - Creating the Agent using agent.py
The agent explores the environment, learns it, and then uses this information from its exploration phase to exploit it.

In this stage, because this is a Markov Decision-Process, we use a variable epsilon to express the rate of exploration - a floating point between 0 and 1. The exploration stage is the means of obtaining representative data - the agent must be exposed to as many potential states as possible to eventually maximize its decision making process and obtaining the highest reward. In unsupervised learning settings, the agent only has access to its own actions as an exploratory setting. The exploration phase counters the chicken/egg problem - the tradeoff value (the epsilon) starts at one, and then gradually decays it by .005 as it begins to explore the environment. Once it reaches a certain threshold, it will begin to make decisions (i.e. exploiting) the environment. The optimal way of identifying this point is through the implementation of a neural network approach, which defines the next action for the agent to take.

In our network, our agent possesses a memory that keeps track of the last 32 states and uses those states to make its next prediction.

In our method replay(), at each time step the agent selects an action, observes a reward, and entres a new state while Q is updated.

class DQNAgent(object):
  """ A simple Deep Q agent """
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = mlp(state_size, action_size)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # returns action

  def replay(self, batch_size=32):
    """ vectorized implementation; 30x speed up compared with for loop """
    minibatch = random.sample(self.memory, batch_size)

    states = np.array([tup[0][0] for tup in minibatch])
    actions = np.array([tup[1] for tup in minibatch])
    rewards = np.array([tup[2] for tup in minibatch])
    next_states = np.array([tup[3][0] for tup in minibatch])
    done = np.array([tup[4] for tup in minibatch])

    # Q(s', a)
    target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
    # end state target is reward itself (no lookahead)
    target[done] = rewards[done]

    # Q(s, a)
    target_f = self.model.predict(states)
    # make the agent to approximately map the current state to future discounted reward
    target_f[range(batch_size), actions] = target

    self.model.fit(states, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
      
Step 3.5 - Using our Neural Network in model.py
Our model is a simple dense neural network with three hidden layers and a dropout of .5 implemented in Keras.

We use the Adam optimizer and mean-squared error loss to better fit the model.
The ReLU activation functions normalize linearity throughout the hidden layers.

```sql
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,activation='relu', loss='mse'):
  model = Sequential()
  model.add(Dense(128, activation='relu', input_dim=n_obs))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(27, activation='linear'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model
```

Using TensorFlow backend.
Training the Model in run.py
Our run.py trains and tests our logic for the program. It contains arguments to run the program from the command line. Our run.py program follows the following steps:

Initializes the Q table with initial states
Chooses an action
Performs the action
Measures the reward
Updates Q
At the end of the training, the output will be a maximizing strategy that allows our model to exploit whichever environment its given.

Demo
As displayed in the graph, our agent is very unstable. While the result is a decent profit by chance, the portfolio values are so variant that a long-term strategy is important. Some of this is endemic to the nature of FOREX trading, while additional data could help reduce some of that variance that we show in the graph.

References
Q Learning for Trading, by Siraj Raval

Foundational Model that We Improved and Expanded On

