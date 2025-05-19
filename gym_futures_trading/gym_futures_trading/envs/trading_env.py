import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import random
import matplotlib.pyplot as plt


Actions = [
    5,
    4,
    3,
    2,
    1,
    0.8,
    0.6,
    0.4,
    0.2,
    0,
    -0.2,
    -0.4,
    -0.6,
    -0.8,
    -1,
    -2,
    -3,
    -4,
    -5,
    -np.inf,
]  # Buy


class TradingEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size, frame_bound, cash=3000, test=False):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.prices, self.signal_features, self.volumes = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self._future_latency = 3

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64
        )

        # stats
        self.win_count = 0
        self.dead_count = 0
        self.cumulative_profit = 0
        self.cumulative_loss = 0
        self._cumulative_funds = cash

        # episode
        self._start_tick = (
            (random.randint(self.window_size, len(self.prices) - self.window_size))
            if not test
            else self.window_size
        )
        self._end_tick = len(self.prices) - 1
        self._start_time = time.time()
        self._end_time = None
        self._start_cash = cash
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._total_reward = None
        self._cash = None
        self._first_rendering = True
        self.history = None
        self._unrealized_profit = None
        self._long_position = None
        self._total_asset = None
        self._average_bid = None
        self._position_history = self.window_size * [0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, start_tick=None):
        if (
            self.cumulative_profit - self.cumulative_loss + self._cumulative_funds
            < self._start_cash
        ):
            self._cumulative_funds += (
                self._start_cash
                - self.cumulative_profit
                + self.cumulative_loss
                - self._cumulative_funds
            )
        # self.cumulative_profit -= self._start_cash
        self._done = False
        self._start_tick = (
            (random.randint(self.window_size, len(self.prices) / 2))
            if start_tick == None
            else start_tick
        )
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.0
        self._cash = self._start_cash
        self._unrealized_profit = 0
        self._total_asset = self._start_cash
        self._long_position = 0
        self._average_bid = 0
        # self._first_rendering = True
        self.history = {}
        self._start_time = time.time()
        # self._position_history = (self.window_size * [0]) + [self._long_position]
        return self._get_observation()

    def is_win(self):
        return self._total_asset > self._start_cash * 4

    def is_dead(self):
        return self._total_asset <= 0

    def step(self, action):
        self._done = False
        self._current_tick += 1
        spend_time = time.time() - self._start_time

        self._update_profit(action)
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        if (
            self._current_tick == self._end_tick - self._future_latency
            or self.is_dead()
            or self.is_win()
        ):
            if self.is_win():
                self.win_count += 1
            elif self.is_dead():
                self.dead_count += 1

            if self._total_asset > 0:
                self.cumulative_profit += self._total_asset - self._start_cash
            else:
                self.cumulative_loss += -(self._total_asset - self._start_cash)
            self._done = True

        if Actions[action] != 0:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._long_position)

        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_asset=self._total_asset,
            cash=self._cash,
            long_position=self._long_position,
            unrealized_profit=self._unrealized_profit,
            start_tick=self._start_tick,
            done_tick=self._current_tick,
            spend_time=spend_time,
            cumulative_profit=self.cumulative_profit,
            cumulative_loss=self.cumulative_loss,
            cumulative_funds=self._cumulative_funds,
            average_bid=self._average_bid,
            profit_rate=self.get_profit_rate(),
            current_price=self.prices[self._current_tick],
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        state = self.signal_features[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ].reshape(self.window_size * 4)
        #print(state)
        tempstate = state
        for i in range(48):
                for j in range(4):
                    if(state[188+j] == 0): print(state)
                    tempstate[i*4+j] = (state[188+j] - state[4*i+j])/state[188+j]
        state = tempstate
        state = np.append(state, self._unrealized_profit)
        state = np.append(state, self._total_asset)
        state = np.append(state, self._cash)
        # state = state / self._start_cash
        # print(len(state))
        state = np.append(state, self._long_position)
        state = np.append(
            state,
            self.volumes[
                (self._current_tick - self.window_size + 1) : self._current_tick + 1
            ].reshape(self.window_size),
        )
        return state

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode="human"):

        # def _plot_position(_long_position, tick):
        #     color = None
        #     if _long_position >= 0:
        #         color = "red"
        #     else:
        #         color = "green"
        #     if color:
        #         plt.scatter(tick, self.prices[tick], color=color)

        # get_profit_rate
        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            # plt.plot(self.prices)
            # print("_position_history len = " + str(len(self._position_history)))
            start_position = self._position_history[self._start_tick]
            # _plot_position(start_position, self._start_tick)
            plt.scatter(self._start_tick, self.get_profit_rate(), color="red")

        # _plot_position(self._long_position, self._current_tick)
        plt.scatter(self._current_tick, self.get_profit_rate(), color="red")

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._cash
        )

        plt.pause(0.01)
        plt.savefig("rendered_plot.png")

    def render_all(self, mode="human"):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        print(self._position_history[:20])
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] >= 0:
                short_ticks.append(tick)
            else:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], "ro")
        plt.plot(long_ticks, self.prices[long_ticks], "go")

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._cash
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        prices = self.df["close"].to_numpy()
        prices = prices[start:end]
        signal_features = self.df.loc[:, ["open", "close", "high", "low"]].to_numpy()[
            start:end
        ]
        volumes = self.df.loc[:, ["volume"]].to_numpy()[start:end]
        return prices, signal_features, volumes

    def _calculate_reward(self, action):
        if self._total_asset <= 0:
            return -10000
        future_price = self.prices[self._current_tick + self._future_latency]
        reward = (
            (future_price - self._average_bid) * self._long_position / self._start_cash
        )
        return reward

    def _update_profit(self, action):
        buy_amount = (
            Actions[action] if Actions[action] != -np.inf else -self._long_position
        )

        current_price = self.prices[self._current_tick]
        # update profit (time)
        self._unrealized_profit = (
            current_price - self._average_bid
        ) * self._long_position

        # 同向操作
        if self._long_position * buy_amount >= 0:
            self._average_bid = (
                (
                    (
                        self._average_bid * self._long_position
                        + buy_amount * current_price
                    )
                    / (self._long_position + buy_amount)
                )
                if self._long_position + buy_amount != 0
                else 0
            )

        # 多空轉換
        if self._long_position * buy_amount < 0 and abs(buy_amount) >= abs(
            self._long_position
        ):
            self._cash += self._unrealized_profit
            self._unrealized_profit = 0
            self._average_bid = current_price
        # 反向操作
        elif self._long_position * buy_amount < 0 and abs(buy_amount) < abs(
            self._long_position
        ):
            position_discount = abs(buy_amount) / abs(self._long_position)
            realized_profit = (
                position_discount
                * (current_price - self._average_bid)
                * self._long_position
            )
            self._cash += realized_profit
            self._unrealized_profit -= realized_profit

        self._long_position += buy_amount

        self._total_asset = self._unrealized_profit + self._cash

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError

    def get_win_rate(self):
        return round(self.win_count / (self.win_count + self.dead_count) * 100, 2)

    def get_cumulative_profit_loss_ratio(self):
        if self.cumulative_loss == 0:
            return ["inf", ""]
        profit_ratio = round(self.cumulative_profit / self.cumulative_loss, 2)
        return [profit_ratio, 1]

    def get_profit_rate(self):
        return (
            (self.cumulative_profit - self.cumulative_loss)
            / self._cumulative_funds
            * 100
        )

    def get_total_asset(self):
        return self._total_asset
