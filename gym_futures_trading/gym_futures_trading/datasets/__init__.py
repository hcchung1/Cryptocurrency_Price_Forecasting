from .utils import load_dataset as _load_dataset

FUTURES_01 = _load_dataset('BTCUSDT-5m-2024-01', 'close_time')
FUTURES_02 = _load_dataset('BTCUSDT-5m-2024-02', 'close_time')
FUTURES_03 = _load_dataset('BTCUSDT-5m-2024-03', 'close_time')
FUTURES_04 = _load_dataset('BTCUSDT-5m-2024-04', 'close_time')
