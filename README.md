# Cryptocurrency Price Forecasting
## 2025 人工智慧概論期末專題  Introduction to A.I. Final Project
### Arthur: Henry Chung(鍾承翰), Eric Chen(陳景寬), 范伯綱, Patrick Wu


### Requirements: Pytorch
### Environments: Python 3.10
---
Data Source: [Binance API](https://github.com/binance/binance-public-data)

---

## Trading Profit Rate (ETHUSDT, Jan – Apr 2025)

|             |              |
| ----------- | ------------ |
| **January** | **February** |
| **March**   | **April**    |

---

## Features

- Custom `` trading environment (`trading_env.py`)
- **Deep Q-Network** training script (`DQN.py`)
- Replay buffer with prioritized sampling and soft-update target network
- 19 discrete trading actions (scalable long/short & hold)
- Evaluation mode with test-time profit plots
- Compatible with hourly ETHUSDT Binance Futures klines (Jan–Apr 2025)

---
## Quick Start

### 1  Clone the Repository

```bash
git clone --branch Model-DQN https://github.com/hcchung1/Cryptocurrency_Price_Forecasting.git
cd Cryptocurrency_Price_Forecasting
```

### 2  Install Requirements

```bash
pip install -r requirements.txt
```

### 3  Install gym-futures-trading

```bash
git clone https://github.com/leafoliage/gym-futures-trading.git
cd gym-futures-trading
git checkout dev
pip install -e .
cd ..
```

### 4  Run Training/Testing

```bash
python DQN.py
```
You can change the model and env in __main__.


Related Projects: 
- [AI-2024-Final-Project](https://github.com/Otmeal/AI-2024-final-project)
- [gym Future Trading Environment](https://github.com/leafoliage/gym-futures-trading/tree/dev)
