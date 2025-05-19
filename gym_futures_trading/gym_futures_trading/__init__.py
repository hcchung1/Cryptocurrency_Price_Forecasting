from gym.envs.registration import register
from copy import deepcopy

from . import datasets


register(
    id="futures1-v0",
    entry_point="gym_futures_trading.envs:TradingEnv",
    kwargs={
        "df": deepcopy(datasets.FUTURES_01),
        "window_size": 48,
        "frame_bound": (48, len(datasets.FUTURES_01)),
    },
)

register(
    id="futures2-v0",
    entry_point="gym_futures_trading.envs:TradingEnv",
    kwargs={
        "df": deepcopy(datasets.FUTURES_02),
        "window_size": 48,
        "frame_bound": (48, len(datasets.FUTURES_02)),
    },
)

register(
    id="futures3-v0",
    entry_point="gym_futures_trading.envs:TradingEnv",
    kwargs={
        "df": deepcopy(datasets.FUTURES_03),
        "window_size": 48,
        "frame_bound": (48, len(datasets.FUTURES_03)),
    },
)

register(
    id="futures4-v0",
    entry_point="gym_futures_trading.envs:TradingEnv",
    kwargs={
        "df": deepcopy(datasets.FUTURES_04),
        "window_size": 48,
        "frame_bound": (48, len(datasets.FUTURES_04)),
    },
)
