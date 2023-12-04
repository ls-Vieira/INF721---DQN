import inf721_model
import torch
import numpy as np

class Agent():
  def __init__(self, gamma, epsilon, learning_rate, state_dim, batch_size, n_actions,
               max_mem_size=100000, eps_min=0.01, eps_dec=5e-4, replace=1000, env_name=None, save_dir='models/'):
    ## Parameters
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.lr = learning_rate
    self.action_space = [i for i in range(n_actions)]
    self.mem_size = max_mem_size
    self.batch_size = batch_size

    ## DDQN
    self.Q_eval = inf721_model.DeepQNetwork(self.lr, n_actions=n_actions, state_dim=state_dim)

    self.Q_next = inf721_model.DeepQNetwork(self.lr, n_actions=n_actions, state_dim=state_dim)
    
    ## Experience Replay
    self.state_memory = np.zeros((self.mem_size,  state_dim), dtype=np.float32)
    self.new_state_memory = np.zeros((self.mem_size,  state_dim), dtype=np.float32)
    self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    ##
    self.mem_cntr = 0
    self.replace_target_cnt = replace
    self.learn_step_counter = 0

    self.env_name = env_name
    self.save_dir = save_dir

  def store_transition(self, state, action, reward, state_, done):
    idx = self.mem_cntr % self.mem_size
    self.state_memory[idx] = state
    self.new_state_memory[idx] = state_
    self.action_memory[idx] = action
    self.reward_memory[idx] = reward
    self.terminal_memory[idx] = done

    self.mem_cntr +=1


  def choose_action(self, observation):
    if np.random.random() > self.epsilon:
      state = torch.tensor(observation).to(self.Q_eval.device)
      actions = self.Q_eval.forward(state)
      action = torch.argmax(actions).item()
    else:
      action = np.random.choice(self.action_space)

    return action

  def replace_target_network(self):
    if self.replace_target_cnt is not None and \
      self.learn_step_counter % self.replace_target_cnt == 0:
      self.Q_next.load_state_dict(self.Q_eval.state_dict())

  def learn(self):
    if self.mem_cntr < self.batch_size:
      return

    self.Q_eval.optimizer.zero_grad()

    self.replace_target_network()

    max_mem = min(self.mem_cntr, self.mem_size)
    batch = np.random.choice(max_mem, self.batch_size, replace=False)

    batch_idx = np.arange(self.batch_size, dtype=np.int32)

    state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
    new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
    reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
    terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

    action_batch = self.action_memory[batch]

    q_pred = self.Q_eval.forward(state_batch)[batch_idx, action_batch]
    q_next = self.Q_next.forward(new_state_batch)
    q_eval = self.Q_eval.forward(new_state_batch)
    max_actions = torch.argmax(q_eval, dim=1)
    q_next[terminal_batch] = 0.0

    q_target = reward_batch + self.gamma * q_next[batch_idx, max_actions]

    loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
    loss.backward()
    self.Q_eval.optimizer.step()
    self.learn_step_counter += 1


    if self.epsilon > self.eps_min:
      self.epsilon -= self.eps_dec
    else:
      self.epsilon = self.eps_min
