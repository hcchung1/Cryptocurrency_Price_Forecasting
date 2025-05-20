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

total_rewards = []

K_LINE_NUM = 48
INPUT_SIZE = K_LINE_NUM * 5 + 4

class replay_buffer:
    # store experience
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        # Add experience when training
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        # fetch experience when learning in training
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)
    
class ActorNet(nn.Module):
    def __init__(self, num_actions, hidden_layer_size=1500): # Added num_actions here
        super(ActorNet, self).__init__()
        self.input_state = INPUT_SIZE
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.input_state, hidden_layer_size)
        self.bn1 = nn.BatchNorm1d(hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.bn2 = nn.BatchNorm1d(hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.bn3 = nn.BatchNorm1d(hidden_layer_size)
        # You can adjust the number of layers if needed
        self.fc_actor = nn.Linear(hidden_layer_size, num_actions) # Output layer for actions

    def forward(self, states):
        if states.dim() == 1 and states.shape[0] == self.input_state: # Check if it's a single state
            states = states.unsqueeze(0)
        
        x = F.relu(self.bn1(self.fc1(states)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        action_logits = self.fc_actor(x)
        return F.softmax(action_logits, dim=-1) # Return action probabilities

class CriticNet(nn.Module):
    def __init__(self, hidden_layer_size=1500): # No num_actions needed here
        super(CriticNet, self).__init__()
        self.input_state = INPUT_SIZE
        self.fc1 = nn.Linear(self.input_state, hidden_layer_size)
        self.bn1 = nn.BatchNorm1d(hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.bn2 = nn.BatchNorm1d(hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.bn3 = nn.BatchNorm1d(hidden_layer_size)
        # You can adjust the number of layers if needed
        self.fc_critic = nn.Linear(hidden_layer_size, 1) # Output layer for state value (V(s))

    def forward(self, states):
        if states.dim() == 1 and states.shape[0] == self.input_state: # Check if it's a single state
            states = states.unsqueeze(0)

        x = F.relu(self.bn1(self.fc1(states)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        value = self.fc_critic(x)
        return value


class Agent:
    def __init__(
        self, env, actor_lr=0.0001, critic_lr=0.0005, GAMMA=0.99, batch_size=50, capacity=2000 # Adjusted LRs
    ):
        self.env = env
        self.n_actions = 19  # the number of actions (make sure this aligns with env.action_space.n if available)
        self.count = 0 # This seems to be a step counter, can be kept

        # self.epsilon = epsilon # Epsilon is not typically used in A2C for action selection
        # self.epsilon_decay_rate = (0.05/epsilon) ** (1/500) # Remove
        
        self.actor_learning_rate = actor_lr # New
        self.critic_learning_rate = critic_lr # New
        # self.learning_rate_decay_rate = (0.002/learning_rate) ** (1/500) # Can adapt if learning rate decay is desired for actor/critic

        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity # Capacity of replay buffer

        self.buffer = replay_buffer(self.capacity) # Keep replay buffer for now

        # Initialize Actor and Critic networks
        self.actor = ActorNet(self.n_actions).to(device)
        self.critic = CriticNet().to(device)

        # Optimizers for Actor and Critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        
        # self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=self.learning_rate) # Remove old optimizer

    def learn(self):
        # No target network update for actor like in DQN.
        # If you implement a target critic, you would update it here periodically.
        # For simplicity, we'll omit the target critic for this initial version.

        if len(self.buffer) < self.batch_size: # Ensure buffer has enough samples
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        # Ensure actions are long type for gather/log_prob and correct shape
        actions = torch.tensor(np.array(actions), dtype=torch.int64).squeeze().to(device) # Ensure it's 1D
        if actions.dim() == 0: # Handle if batch_size is 1 and squeeze removed all dims
            actions = actions.unsqueeze(0)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float).to(device) # 0 for not done, 1 for done

        # --- Critic Update ---
        self.critic.train() # Set critic to training mode
        
        # Get V(s) and V(s')
        # V_s current shape is [batch_size, 1], squeeze to [batch_size] if rewards and dones are [batch_size]
        V_s = self.critic(states).squeeze(-1) # Squeeze to [batch_size]
        V_s_prime = self.critic(next_states).squeeze(-1).detach() # Detach as we don't want gradients flowing back from V_s_prime

        # TD Target: R + gamma * V(s') * (1 - done)
        # Ensure shapes are compatible for broadcasting if needed.
        # rewards: [batch_size], V_s_prime: [batch_size], dones: [batch_size]
        TD_target = rewards + self.gamma * V_s_prime * (1 - dones)
        
        # Advantage: A(s,a) = TD_target - V(s)
        advantage = (TD_target - V_s).detach() # Detach: advantage is treated as a constant for actor update

        # Critic loss (MSE)
        critic_loss = F.mse_loss(V_s, TD_target) # TD_target already detached effectively by V_s_prime.detach()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        self.actor.train() # Set actor to training mode

        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        
        # log_prob for the actions taken from the batch
        # `actions` should be 1D tensor of actions taken
        log_probs_taken_actions = dist.log_prob(actions)
        
        actor_loss = -(log_probs_taken_actions * advantage).mean() 
        
        entropy_bonus = dist.entropy().mean()
        actor_loss -= 0.001 * entropy_bonus 

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # torch.save(self.actor.state_dict(), "./Tables/A2C_actor.pt")
        # torch.save(self.critic.state_dict(), "./Tables/A2C_critic.pt")

    def choose_action(self, state, testing=False):
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        # Ensure the state is 2D [batch_size, num_features] if it's a single state
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad(): # Important for inference
            self.actor.eval() # Set actor to evaluation mode
            action_probs = self.actor(state_tensor)
            self.actor.train() # Set back to train mode if this was just for inference within a training step. Or handle mode outside.

            if testing: # During testing, choose the best action
                action = torch.argmax(action_probs).item()
            else: # During training, sample from the distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
        return action

def train(env, episode=200): # Consider renaming episode to num_episodes for clarity
    # actor_lr, critic_lr can be passed to Agent or hardcoded there
    agent = Agent(env) # Initializes A2C agent

    rewards_history = [] # To store total reward per episode
    # writer = SummaryWriter(f'runs/A2C_experiment_{time.time()}') # Example for TensorBoard

    for i_episode in range(episode):
        state = env.reset()
        episode_reward = 0 # Renamed 'count' to 'episode_reward' for clarity, or use for steps
        done = False # Initialize done
        current_episode_steps = 0

        while not done: # Loop until episode is done
            current_episode_steps += 1
            agent.count += 1 # Global step counter

            # Choose action (testing=False during training to enable sampling)
            action = agent.choose_action(state, testing=False)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            agent.buffer.insert(state, int(action), reward, next_state, int(done))

            if len(agent.buffer) >= agent.batch_size: # Start learning when buffer has enough
                # Consider learning more frequently or after N steps rather than just if buffer is full
                # For example, learn every K steps: if agent.count % K == 0: agent.learn()
                agent.learn() 
            
            state = next_state

        rewards_history.append(episode_reward)
        # writer.add_scalar('reward/episode_reward', episode_reward, i_episode) # Tensorboard
        # writer.add_scalar('params/actor_lr', agent.actor_learning_rate, i_episode) # If LR changes
        # writer.add_scalar('params/critic_lr', agent.critic_learning_rate, i_episode)

        print(f"Episode: {i_episode}, Total Reward: {episode_reward}, Info: {info}")

        # Epsilon and LR decay logic needs to be removed or adapted for actor/critic LRs
        # if(i_episode % math.ceil(episode/500) == 0):
        #     # agent.epsilon *= agent.epsilon_decay_rate # Epsilon not used
        #     # agent.learning_rate *= agent.learning_rate_decay_rate # Adapt for actor/critic LR if needed
        #     pass 
        # print("epsilon: ", agent.epsilon, "learning_rate: ", agent.learning_rate) # Remove or adapt

    # Save models after training
    os.makedirs("./Tables", exist_ok=True) # Ensure directory exists
    torch.save(agent.actor.state_dict(), "./Tables/A2C_actor_final.pt")
    torch.save(agent.critic.state_dict(), "./Tables/A2C_critic_final.pt")
    print("Training finished. Models saved.")
    
    # total_rewards.append(rewards_history) # If total_rewards is a global list
    return rewards_history # Return rewards for plotting or analysis


def test(env, actor_model_path): # actor_model_path 變數名已針對 A2C
    """
    Test the A2C agent on the given environment.
    Parameters:
        env: the given environment.
        actor_model_path: path to the trained A2C actor model.
    """
    print(f"Testing with actor model: {actor_model_path}") # 增加日誌
    testing_agent = Agent(env) # 初始化 Agent (A2C 版本)

    # 加載訓練好的 Actor 模型
    try:
        testing_agent.actor.load_state_dict(torch.load(actor_model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Actor model not found at {actor_model_path}")
        return
    except Exception as e:
        print(f"Error loading actor model: {e}")
        return

    testing_agent.actor.eval() # 設置 Actor 為評估模式

    start_tick = K_LINE_NUM
    profit_rate = []
    profit_rate_tick = []

    current_start_tick = start_tick
    # test_episode_count = 0 # 如果您移除了 max_test_episodes，這個可以不用
    # max_test_episodes = 10 # 您之前提到移除了這個條件

    while True:
        # 您修改後的終止條件
        if current_start_tick >= len(env.prices) - K_LINE_NUM - 4:
            print(f"Testing finished: Reached end of price data. current_start_tick ({current_start_tick}) >= threshold.")
            break

        try:
            state = env.reset(start_tick=current_start_tick)
            # print(f"  New test segment starting from tick: {current_start_tick}") # 可選日誌
        except AttributeError: # Fallback if env doesn't have start_tick or if it's for the first episode only
            # 確保 env.reset() 在不帶參數時也能工作，或者在循環開始前處理好第一次 reset
            if current_start_tick == K_LINE_NUM: # 通常只在第一次 reset 時不帶 start_tick
                 state = env.reset()
            else:
                print("Environment cannot be reset with start_tick for subsequent segments, or data exhausted.")
                break
        except Exception as e:
            print(f"Error resetting environment at tick {current_start_tick}: {e}")
            break

        done = False
        # episode_steps = 0 # 可選：追蹤每個片段的步數

        while not done:
            # episode_steps += 1
            action = testing_agent.choose_action(state, testing=True) # 使用 Actor 選擇動作
            next_state, _, done, info = env.step(action)

            if done:
                current_profit = env.get_profit_rate()
                current_done_tick = info.get("done_tick")

                if current_done_tick is not None:
                    profit_rate.append(current_profit)
                    profit_rate_tick.append(current_done_tick)

                    # 打印每個片段結束時的信息 (與您之前的日誌一致)
                    info_to_print = {}
                    for key_info in ['total_reward', 'total_asset', 'cash', 'long_position', 'unrealized_profit', 'done_tick']:
                        info_to_print[key_info] = int(info.get(key_info, 0))
                    print(f"Segment done at tick {current_done_tick}. Final Profit Rate for segment: {current_profit:.2f}%. Info: {info_to_print}")

                    current_start_tick = current_done_tick # 更新下一個片段的起始點
                else:
                    print(f"Warning: 'done_tick' not found in info when done. Current profit: {current_profit:.2f}%. Ending test.")
                    current_start_tick = len(env.prices) # 強制結束外部迴圈
                break # 跳出內層 while not done 迴圈
            state = next_state
        # test_episode_count += 1 # 如果您移除了 max_test_episodes，這個可以不用

    # --- 繪圖邏輯 ---
    if not profit_rate_tick or not profit_rate:
        print("No data to plot. Profit rates were not recorded.")
        return

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # 繪製價格曲線
    ax1.plot(range(len(env.prices)), env.prices, label='prices', color='blue')
    ax1.set_ylabel('prices', color='blue')
    ax1.set_xlabel('Time Ticks (Overall Index)')

    # 繪製 profit_rate 線 (使用小點標記所有中間點)
    ax2.plot(profit_rate_tick, profit_rate, color='red', linestyle='-', marker='.', linewidth=1.5, markersize=4, label='profit_rate (%)')
    ax2.set_ylabel('profit_rate (%)', color='red')

    # 僅在最後一個點加上三角形標記並註解
    if profit_rate_tick: # 確保列表不為空
        last_tick_plot = profit_rate_tick[-1]
        last_profit_rate_plot = profit_rate[-1]
        # 繪製最後一個點的三角形
        ax2.plot(last_tick_plot, last_profit_rate_plot, color='red', marker='^', markersize=10, markeredgecolor='black') # 稍大且帶邊框的三角形

        ax2.annotate(
            f'{last_profit_rate_plot:.2f}%',
            xy=(last_tick_plot, last_profit_rate_plot),
            xytext=(10, 0), # 在點的右邊偏移 10 個點
            textcoords='offset points',
            ha='left', # 水平對齊
            va='center', # 垂直對齊
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black') # 可選的箭頭樣式
        )

    plt.title('ETH Price & Agent Profit Rate Over Time')
    # 合併圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 過濾掉 ax2.plot 三角形時可能產生的重複圖例項（如果label被重複設置）
    # 一個簡單的方法是只取第一個 label（假設 'profit_rate (%)'）
    if labels2: # 確保 labels2 不是空的
        ax2.legend([lines[0], lines2[0]], [labels[0], labels2[0]], loc='upper left', bbox_to_anchor=(0.05, 0.95))
    else: # 如果 ax2 沒有圖例項（例如 profit_rate 為空）
        ax1.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))


    plt.show()

    


if __name__ == "__main__":
    env = gym.make('futures1-v0') 
    os.makedirs("./Tables", exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device: ', device)

    # training section:


    # for i in range(1):
    #     time0 = time.time()
    #     print(f"#{i + 1} training progress")
    #     train(env, 200)
    #     time1 = time.time()
    #     print(f"Training time: {time1 - time0} seconds")
    #     print ("Win rate: ", env.win_count ,"/", env.win_count + env.dead_count, f"({env.get_win_rate()})")
    #     [profit, loss] = env.get_cumulative_profit_loss_ratio()
    #     print("Profit Loss Ratio: ",f"{profit} : {loss}" )
    #     print ("Final profit rate: ", env.get_profit_rate())


    # testing section:
    test(env, "./Tables/A2C_actor_final.pt")
    env.close()

