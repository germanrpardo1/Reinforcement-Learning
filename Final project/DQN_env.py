import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io

from collections import deque, namedtuple


BUFFER_SIZE = int(1e5) # replay buffer size
BATCH_SIZE = 32        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-5              # for soft update of target parameters
LR = 0.01               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 



class Board_DQN:
    def __init__(self, rows, cols):
        self.rows = rows
        self.board = np.zeros((self.rows, self.rows))
        self.state = str(self.board.reshape(self.rows ** 2))
        self.reward = 0.0
        self.isTerminal = False
        self.initial_actions = self.availablePositions()
        self.actions = self.initial_actions
        self.state_actions = {}
        
    def winner(self):
        # row
        for i in range(self.rows):
            if sum(self.board[i, :]) == self.rows:
                self.isTerminal = True
                return 1
            if sum(self.board[i, :]) == -self.rows:
                self.isTerminal = True
                return -1
        # col
        for i in range(self.rows):
            if sum(self.board[:, i]) == self.rows:
                self.isTerminal = True
                return 1
            if sum(self.board[:, i]) == -self.rows:
                self.isTerminal = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(self.rows)])
        diag_sum2 = sum([self.board[i, self.rows - i - 1] for i in range(self.rows)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == self.rows:
            self.isTerminal = True
            if diag_sum1 == self.rows or diag_sum2 == self.rows:
                return 1
            else:
                return -1
        # tie
        # no available positions
        if len(self.actions) == 0:
            self.isTerminal = True
            return 0
        # not end
        self.isTerminal = False
        return 0

    def availablePositions(self):
        positions = []
        for i in range(self.rows):
            for j in range(self.rows):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def step(self, action):
        self.board[action] = 1
        self.actions = self.availablePositions()
        self.state = str(self.board.reshape(self.rows ** 2))
        self.reward = self.winner()
        if not self.isTerminal and self.actions != []:
            self.random_policy()
                
        return self.state,  self.actions, self.reward, self.isTerminal

    def random_policy(self):
        # Random opponent
        opp_action = np.random.randint(0, len(self.actions))
        self.board[self.actions[opp_action]] = -1
        self.actions = self.availablePositions()
        self.state = str(self.board.reshape(self.rows ** 2))
        self.reward = self.winner()

    # board reset
    def reset(self):
        self.board = np.zeros((self.rows, self.rows))
        self.state = str(self.board.reshape(self.rows ** 2))
        self.actions = self.initial_actions
        self.reward = 0.0
        self.isTerminal = False
        
        return self.state, self.actions, self.isTerminal
        
    def add_avaliable_actions(self, s, q):
        self.state_actions[s] = self.actions
        for new_a in self.state_actions[s]:
            q[s, new_a] = 0
        return q

    def showBoard(self):
        for i in range(0, self.rows):
            print('----'*self.rows)
            out = '| '
            for j in range(0, self.rows):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('----'*self.rows)



class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, input_size, seed = 1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state, action):
        """Build a network that maps (state, action) -> action value."""
        if len(action) > 2:
            input = [list(state[i]) + list(action[i]) for i in range(round(len(action)))]
            input = torch.tensor(input, dtype = torch.float)
        else:
            input = torch.tensor(list(state) + list(action), dtype = torch.float)
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, input_size, seed = 1):
        """Initialize an Agent object.
        
        Params
        ======
            input_size (int): dimension of the input (state + 1)
            seed (int): random seed
        """
        self.input_size = input_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(input_size, seed).to(device)
        self.qnetwork_target = QNetwork(input_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, av_actions):
        # Save experience in replay memory
        experience = self.memory.add(state, action, reward, next_state, done, av_actions)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
            
    def max_action(self, state, av_actions, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        self.qnetwork_local.eval()
        with torch.no_grad():
            max = -10e10
            for action in av_actions:
                if max < self.qnetwork_local(state, action):
                    max = self.qnetwork_local(state, action)
                    max_action = action
        self.qnetwork_local(state, max_action)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return max_action
        else:
            return av_actions[random.randint(0, len(av_actions) - 1)]

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones, av_actions = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network

        with torch.no_grad():
            max_as = []
            for k in range(len(next_states)):
                max = -10e6
                if av_actions[k] == []:
                    q_target_next = 0
                    max_a = (0,0)
                else:
                    for a in av_actions[k]:
                        if self.qnetwork_target(states[k], a) > max:
                            max = self.qnetwork_target(states[k], a).detach().numpy()[0]
                            max_a = a
                            
                max_as.append(max_a)
        q_target_next = self.qnetwork_target(states, max_as)

        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_target_next * (1 - dones)
        
        q_expected = self.qnetwork_local(states, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", 'av_actions'])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, av_actions):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, av_actions)
        self.memory.append(e)

    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        av_actions = [e.av_actions for e in experiences if e is not None]#torch.from_numpy(np.vstack([e.av_actions for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones, av_actions)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)