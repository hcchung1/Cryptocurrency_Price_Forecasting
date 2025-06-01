import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
from torch import Tensor
import os
from tqdm import tqdm
import gym_futures_trading
import math
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

K_LINE_NUM = 48
INPUT_SIZE = K_LINE_NUM * 5 + 4

def test(env):
    """
    Test the given trading environment by using a Moving Average Crossover strategy.
    - average50: list of 50-day moving averages (calculated over the first 4 features).
    - average200: list of 200-day moving averages.
    """
    # Precompute the 50-day and 200-day moving averages over the entire price history
    average50 = []
    average200 = []
    total_steps = env.frame_bound[1]

    for i in range(total_steps):
        if i >= 50:
            # Sum of the first 4 features over the last 50 steps, divided by (50 * 4)
            window = env.signal_features[i-50:i, 0:4]
            average50.append(np.sum(window) / (50 * 4))
        if i >= 200:
            window200 = env.signal_features[i-200:i, 0:4]
            average200.append(np.sum(window200) / (200 * 4))

    count = 0            # overall step counter across entire testing loop
    env.reset()
    count1 = 0           # count of 'buy' actions (action = 4)
    count0 = 0           # count of 'sell' actions (action = 14)

    # Prepare TensorBoard writer for tracking profit over time
    tb_dir = 'tb_record_1/comp_profit_train/baseline'
    os.makedirs(tb_dir, exist_ok=True)
    w = SummaryWriter(tb_dir)

    t = 0                # per-episode time index (for TensorBoard)
    start_tick = K_LINE_NUM
    profit_rate = []
    profit_rate_tick = []

    # Continue running episodes until we reach the end of price data
    while True:
        if start_tick >= len(env.prices) - 4:
            break

        # Start a new episode from start_tick
        state = env.reset(start_tick=start_tick)
        t = 0

        while True:
            # Skip the first 200 steps (warm-up period) by always holding (action = 9)
            if count < 200:
                env.step(9)
                w.add_scalar('Profit', env.get_profit_rate(), t)
                t += 1
                count += 1
                continue

            # Once past 200 steps, use Moving Average Crossover to pick buy/sell
            # Compare the 50-day MA at index (count - 200 + 150) to the 200-day MA at index (count - 200)
            idx50 = count - 200 + 150
            idx200 = count - 200
            if average50[idx50] > average200[idx200]:
                action = 4    # buy
                count1 += 1
            else:
                action = 14   # sell
                count0 += 1

            next_state, _, done, info = env.step(action)
            w.add_scalar('Profit', env.get_profit_rate(), t)
            t += 1
            count += 1

            if done:
                # Record final metrics for this episode
                profit_rate.append(env.get_profit_rate())
                profit_rate_tick.append(info["done_tick"])

                # Convert certain info fields to int for cleaner printing
                info['total_reward'] = int(info.get('total_reward', 0))
                info['total_asset'] = int(info.get('total_asset', 0))
                info['cash'] = int(info.get('cash', 0))
                info['long_position'] = int(info.get('long_position', 0))
                info['unrealized_profit'] = int(info.get('unrealized_profit', 0))
                print(info)

                # Move start_tick to the tick where this episode ended
                start_tick = info["done_tick"]
                break

    # After all episodes, plot price vs. profit_rate
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(range(len(env.prices)), env.prices, label='prices', color='blue')
    ax1.set_ylabel('prices', color='blue')

    ax2.plot(profit_rate_tick, profit_rate, label='profit_rate', color='red')
    ax2.set_ylabel('profit_rate', color='red')

    # Annotate the last profit_rate on the plot
    last_tick = profit_rate_tick[-1]
    last_profit_rate = profit_rate[-1]
    ax2.annotate(
        f'{int(last_profit_rate)}',
        xy=(last_tick, last_profit_rate),
        xytext=(last_tick + 1, last_profit_rate),
        arrowprops=dict(facecolor='red', shrink=0.05)
    )

    plt.title('prices & profit_rate')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.show()

    # Print summary statistics
    print("count action - buy : " + str(count1))
    print("count action - sell : " + str(count0))
    print("total profit rate: " + str(env.get_profit_rate()))

if __name__ == "__main__":
    env = gym.make('futures4-v0')
    test(env)
