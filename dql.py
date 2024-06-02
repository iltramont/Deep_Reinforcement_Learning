# For our baseline model, we decided to use Deep Q-Learning. To do so, we had to discretize action space.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os

from collections import deque
from IPython.display import clear_output
import matplotlib.pyplot as plt


# Neural network model
# With this model we need to discretize the action space
class DQN(nn.Module):
    """
    Class for a feedforward neural network with:
    - Input layer
    - three hidden layers
    - output layer
    All layers are fully connected.
    Network takes state as input and return state_action_value for each action.
    """

    def __init__(self, state_size: int, action_size: int, activation_function: str = 'relu'):
        """
        :param activation_function: activation function to be used in forward method. Options: 'relu', 'tanh'.
        """
        super(DQN, self).__init__()
        self.activation_function = activation_function
        self.fc1 = nn.Linear(state_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 32)  # Third fully connected layer
        self.fc4 = nn.Linear(32, action_size)  # Fourth fully connected layer

    def forward(self, x):
        if self.activation_function == 'tanh':
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            return self.fc4(x)
        else:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return self.fc4(x)


class DQAgent:
    """
    DQAgent Class follows the same structure as the one seen during lectures.
    """

    def __init__(self, state_size: int,
                 action_size: int,
                 epsilon_decay: float,
                 epsilon_min: float,
                 memory_size: int = 20_000,
                 activation_function: str = 'relu'):
        """
        :param state_size: environment state size.
        :param action_size: number of action in which we want to discretize the action space.
        :param epsilon_decay: epsilon decay for epsilon-greedy policy.
        :param epsilon_min: min value of epsilon.
        :param memory_size: max length of replay buffer.
        :param activation_function: activation function to be used by DQN.
        """
        self.activation_function = activation_function
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon_min = epsilon_min

        # Evenly discretize action space and create set of possible actions.
        self.actions = np.linspace(-1, 1, action_size)

        # Set hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Starting epsilon
        self.epsilon_decay = epsilon_decay  # The higher, the faster epsilon decays
        self.learning_rate = 0.0005  # Learning rate used to update network
        self.checkpoint_episode = 20  # Used to save network
        self.tau = 0.005  # Used to update target network

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize neural networks
        self.model = DQN(state_size, action_size, activation_function).to(self.device)
        self.target_model = DQN(state_size, action_size, activation_function).to(self.device)
        # Set equal initial weights for both networks
        self.target_model.load_state_dict(self.model.state_dict())

        # initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.training_returns = []
        self.training_steps = []

    def update_target_model(self):
        """
        Soft update of the target network's weights: θ′ ← τ θ + (1 −τ )θ′
        """
        target_state_dict = self.target_model.state_dict()
        state_dict = self.model.state_dict()
        for key in state_dict:
            target_state_dict[key] = self.tau * state_dict[key] + (1 - self.tau) * target_state_dict[key]
        self.target_model.load_state_dict(target_state_dict)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training: bool):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            if training and np.random.rand() <= self.epsilon:
                return random.choice(self.actions)
            else:
                action_values = self.model(state)
                action_index = torch.argmax(action_values, dim=1).item()
                return self.actions[action_index]

    def replay(self, batch_size: int = 32):
        minibatch = random.sample(self.memory, batch_size)
        for (state, action, reward, next_state, done) in minibatch:
            action_index = np.where(self.actions == action)[0][0]
            state = torch.from_numpy(state).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.tensor(reward).float().to(self.device)

            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(self.target_model(next_state).detach())

            Q_sa = self.model(state)[0][action_index]
            loss = self.criterion(target, Q_sa)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def save_model(self, directory: str):
        torch.save(self.model.state_dict(), directory)

    def generate_greedy_episode(self,
                                env,
                                render: bool = False,
                                frame_rate: float = 1 / 30,
                                background_text_color: str = 'white') -> (float, int):
        """
        Generate an episode selecting action fully greedily.
        :param background_text_color: background color of text when displaying environment.
        :param env: the environment.
        :return: episode return, episode length
        """
        state, _ = env.reset()
        state = np.reshape(state, [1, self.state_size])
        terminated, truncated = False, False
        episode_return = 0
        episode_steps = 0
        while not (terminated | truncated):

            # Display if render
            if render:
                time.sleep(frame_rate)
                clear_output()
                plt.imshow(env.render())
                plt.text(100, 400, f'Return = {episode_return:.2f}\nSteps = {episode_steps}',
                         backgroundcolor=background_text_color)
                plt.show()

            # Choose action
            action = self.act(state, training=False)
            # Take action
            next_state, reward, terminated, truncated, _ = env.step((action,))
            episode_return += reward
            episode_steps += 1
            state = np.reshape(next_state, [1, self.state_size])

        # Handle terminal state
        if render:
            time.sleep(frame_rate)
            clear_output()
            plt.imshow(env.render())
            plt.text(100, 400, f'Return = {episode_return:.2f}\nSteps = {episode_steps}',
                     backgroundcolor=background_text_color)
            plt.show()
        return episode_return, episode_steps

    def train(self, env,
              num_episodes: int = 1000,
              save_freq: int = 1000,
              batch_size: int = 32,
              save_dir: str = 'models',
              save_name: str = 'DQL') -> (list[float], list[int]):
        """
        Used to train Q-learning agent
        :return: list of returns and list of steps achieved for each episode
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize variables
        returns = []
        steps = []
        total_steps = 0
        # Loop over episodes
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state, info = env.reset()
            state = np.reshape(state, [1, self.state_size])
            terminated, truncated = False, False
            episode_return = 0
            episode_loss = 0
            episode_steps = 0

            # Loop over steps
            while not (terminated | truncated):
                # Choose action
                action = (self.act(state, training=True),)
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward = reward if not terminated else -10.0
                next_state = np.reshape(next_state, [1, self.state_size])
                # Store transition
                self.remember(state, action, reward, next_state, terminated)

                if len(self.memory) > batch_size:
                    self.replay()
                    self.update_target_model()

                # Update state
                state = next_state
                # Update episode variables
                episode_return += reward
                episode_steps += 1
                total_steps += 1
            # Store episode variables
            self.training_returns.append(episode_return)
            self.training_steps.append(episode_steps)
            # Print episode information
            print(f'Episode {episode}/{num_episodes} | Reward: {episode_return:.2f} | Steps: {episode_steps}'
                  f' | Epsilon: {self.epsilon:.2f} | Memory: {len(self.memory)}')
            # Save model
            if episode % save_freq == 0:
                self.save_model(os.path.join(save_dir, f'{save_name}_{episode}.pt'))
        # Save final model
        self.save_model(os.path.join(save_dir, f'{save_name}_final.pt'))
        return self.training_returns, self.training_steps
