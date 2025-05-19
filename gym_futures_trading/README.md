
# gym-futures-trading

`FuturesTrading` is a gym for trading BTC futures, modified from `AnyTrading`.

### Modifications

#### Env states

* `_total_profit` -> `_realized_profit`
* added `_unrealized_profit`
* added `_long_position`

#### Env methods

* `_calculate_reward`
* `_update_profit`


## Installation

### From Repository

Clone this repository. Use `pwd` to get its path on your system and copy the path.

```bash
git clone https://github.com/leafoliage/gym-anytrading.git
cd gym-anytrading
pwd
```

cd to AI project directory, then install `gym-futures-trading`.

```bash
cd /path/to/AI-final-project
pip install -e /path/to/gym-futures-trading
```

## Environment Properties

### Trading Actions

Actions allowed are a list of numbers, meaning buying how many BTC:

``` 
5,4,3,2,1, 0.8, 0.6, 0.4, 0.2, 0 -0.2, -0.4, -0.6, -0.8, -1, -2, -3, -4, -5
```

### Trading Positions

* `_position` (Legacy): Long / Short
* `_long_position`: number, meaning how much BTC you hold

## Trading Environments

### TradingEnv

Inheriting `gym.Env`

* Properties:
> `df`: An abbreviation for **DataFrame**. It's a **pandas'** DataFrame which contains your dataset and is passed in the class' constructor.
>
> `prices`: Real prices over time. Used to calculate profit and render the environment.
>
> `signal_features`: Extracted features over time. Used to create *Gym observations*.
>
> `window_size`: Number of ticks (current and previous ticks) returned as a *Gym observation*. It is passed in the class' constructor.
>
> `action_space`: The *Gym action_space* property. Containing discrete values of **0=Sell** and **1=Buy**.
>
> `observation_space`: The *Gym observation_space* property. Each observation is a window on `signal_features` from index **current_tick - window_size + 1** to **current_tick**. So `_start_tick` of the environment would be equal to `window_size`. In addition, initial value for `_last_trade_tick` is **window_size - 1** .
>
> `shape`: Shape of a single observation.
>
> `history`: Stores the information of all steps.

* Methods:
> `seed`: Typical *Gym seed* method.
>
> `reset`: Typical *Gym reset* method.
>
> `step`: Typical *Gym step* method.
>
> `render`: Typical *Gym render* method. Renders the information of the environment's current tick.
>
> `render_all`: Renders the whole environment.
>
> `close`: Typical *Gym close* method.

* Abstract Methods:
> `_process_data`: It is called in the constructor and returns `prices` and `signal_features` as a tuple. In different trading markets, different features need to be obtained. So this method enables our TradingEnv to be a general-purpose environment and specific features can be returned for specific environments such as *FOREX*, *Stocks*, etc.
>
> `_calculate_reward`: The reward function for the RL agent.
>
> `_update_profit`: Calculates and updates total profit which the RL agent has achieved so far. Profit indicates the amount of units of currency you have achieved by starting with *1.0* unit (Profit = FinalMoney / StartingMoney).
>
> `max_possible_profit`: The maximum possible profit that an RL agent can obtain regardless of trade fees.


### Create an environment


```python
import gym
import gym_futures_trading

env = gym.make('futures1-v0')

```

* This will create the default environment. You can change any parameters such as dataset, frame_bound, etc.

### Create an environment with custom parameters
I put two default datasets for [*FOREX*](https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/datasets/data/FOREX_EURUSD_1H_ASK.csv) and [*Stocks*](https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/datasets/data/STOCKS_GOOGL.csv) but you can use your own.


```python
from gym_futures_trading.datasets import FUTURES_01

custom_env = gym.make('futures1-v0',
               df = FUTURES_01,
               window_size = 10,
               frame_bound = (10, 300),
               unit_side = 'right')
```

* It is to be noted that the first element of `frame_bound` should be greater than or equal to `window_size`.

### Print some information


```python
print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())

print()
print("custom_env information:")
print("> shape:", custom_env.shape)
print("> df.shape:", custom_env.df.shape)
print("> prices.shape:", custom_env.prices.shape)
print("> signal_features.shape:", custom_env.signal_features.shape)
print("> max_possible_profit:", custom_env.max_possible_profit())
```

    env information:
    > shape: (24, 2)
    > df.shape: (6225, 5)
    > prices.shape: (6225,)
    > signal_features.shape: (6225, 2)
    > max_possible_profit: 4.054414887146586
    
    custom_env information:
    > shape: (10, 2)
    > df.shape: (6225, 5)
    > prices.shape: (300,)
    > signal_features.shape: (300, 2)
    > max_possible_profit: 1.122900180008982
    

- Here `max_possible_profit` signifies that if the market didn't have trade fees, you could have earned **4.054414887146586** (or **1.122900180008982**) units of currency by starting with **1.0**. In other words, your money is almost *quadrupled*.

### Plot the environment


```python
env.reset()
env.render()
```

### A complete example


```python
import gym
import gym_futures_trading
import matplotlib.pyplot as plt

env = gym.make('futures1-v0')

observation = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # env.render()
    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()
```


- You can use `render_all` method to avoid rendering on each step and prevent time-wasting.


## Related Projects

* [gym-anytrading](https://github.com/AminHP/gym-anytrading)
