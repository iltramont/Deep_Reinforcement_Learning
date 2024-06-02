# In this file there will be:
# - Actor network class
# - Critic network class
# - Agent class
# The logic used to build the agent follows the paper of Lillicrap, Hunt et al. "Continuous Control With Deep
# Reinforcement Learning"

import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import copy
import os
import time

from collections import deque
from IPython.display import clear_output
import matplotlib.pyplot as plt


class ActorNetwork(nn.Module):
    """
    Class for a feedforward neural network with
    - Input layer
    - two hidden layers
    - output layer
    All layers are fully connected.
    The output is mu(s|theta): the action mu given state s and weights theta.
    """

    def __init__(self, state_size: int):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 32)  # Second fully connected layer
        self.fc3 = nn.Linear(32, 1)  # Output fully connected layer

    def forward(self, x):
        """
        We use tanh as activation function in order to have an action between -1 and +1.
        :param x: torch tensor of size [state_size],  representing the state.
        :return: torch tensor of size [1], representing the action.
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """
    Class for a feedforward neural network with
    - Input layer
    - two hidden layers
    - output layer
    All layers are fully connected.
    The output is Q(s, a|w): state_action given state s, action a and weights w.
    IMPORTANT! Input must be a tensor of size [state_size + 1] (the state plus the action).
    """

    def __init__(self, state_size: int):
        super(CriticNetwork, self).__init__()
        # Input size is state_size + 1 (action is a real number)
        self.fc1 = nn.Linear(state_size + 1, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 32)  # Second fully connected layer
        self.fc3 = nn.Linear(32, 1)  # Output fully connected layer

    def forward(self, x):
        """
        With Critic network we use relu as activation function.
        :param x: torch tensor of size [state_size + 1], representing the state_action.
        :return: torch tensor of size [1], representing state_action value.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DDPG Actor-Critic agent
class DDPGAgent:
    def __init__(self, state_size: int,
                 gamma: float,
                 learning_rate: float,
                 tau: float,
                 exploration_sd: float,
                 memory_size: int):
        """
        :param state_size: environment state size.
        :param gamma: rewards' discount factor.
        :param learning_rate: learning rate used to update networks (is the same for both critic and actor).
        :param tau: coefficient used to smooth the update of target networks (higher values leads to faster updates).
        :param exploration_sd: standard deviation of the noise added to action in training mode.
        :param memory_size: replay buffer's maximum size.
        """
        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = state_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        self.exploration_sd = exploration_sd

        # Initialize lists for training. Used to keep track of training performance.
        self.training_returns = []
        self.training_steps = []
        self.training_time = []

        # Initialize memory
        self.memory = deque(maxlen=memory_size)

        # Initialize networks
        self.actor = ActorNetwork(state_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        self.target_actor = ActorNetwork(state_size).to(self.device)
        self.target_critic = CriticNetwork(state_size).to(self.device)
        # Set equal weights for target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # initialize optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        #self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_networks(self):
        """
        Soft update of the target networks' weights: θ′ ← τ θ + (1 −τ )θ′.
        Used in training.
        """
        # Update Critic
        target_state_dict = self.target_critic.state_dict()
        state_dict = self.critic.state_dict()
        for key in state_dict:
            target_state_dict[key] = self.tau * state_dict[key] + (1 - self.tau) * target_state_dict[key]
        self.target_critic.load_state_dict(target_state_dict)
        # Update Actor
        target_state_dict = self.target_actor.state_dict()
        state_dict = self.actor.state_dict()
        for key in state_dict:
            target_state_dict[key] = self.tau * state_dict[key] + (1 - self.tau) * target_state_dict[key]
        self.target_actor.load_state_dict(target_state_dict)

    def act(self, state, training: bool) -> float:
        """
        Act according to Actor. Add noise if training is True (Explore).
        :return: action (float) between -1 and +1
        """
        self.actor.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            mu = self.actor(state).item()  # Between -1 and 1 due to tanh activation function
            self.actor.train()
            if training:
                # Explore: deterministic part + noise
                noise = np.random.normal(0, self.exploration_sd)
                return max(-1.0, min(1.0, mu + noise))  # Action must be between -1 and +1
            else:
                # Fully deterministic
                return mu

    def remember(self, state, action, reward, next_state, done):
        """
        Add transition to replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 32):
        """
        Used to update Critic and Actor networks.
        Step 1: Update Critic network by minimizing the loss between target and Q(s, a).
            target is y = reward + gamma * Q'(s_next, mu'(s_next)).
        Step 2: Update Actor network using the sampled policy gradient.
        """

        # Update Critic
        # Initialize vector y and batch X
        y = []  # Target tensor. Shape = (batch_size, 1)
        X = []  # State-action batch tensor. Shape = (batch_size, state_size + 1)
        # Sample a random minibatch of transitions from replay buffer
        minibatch = random.sample(self.memory, batch_size)
        for (state, action, reward, next_state, done) in minibatch:
            state_action: np.ndarray = np.concatenate((state, (action, )))
            X.append(state_action)
            next_state: torch.Tensor = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.tensor(reward).float().to(self.device)

            # Set networks to eval mode. (No need to track gradients).
            self.target_actor.eval()
            self.target_critic.eval()
            self.critic.eval()

            if done:
                target = reward
            else:
                mu_target = self.target_actor(next_state)
                target = reward + self.gamma * self.target_critic(torch.cat([next_state, mu_target]))
            y.append(target.item())

        # Transform X and y to torch tensor
        # At the moment X is a list of numpy arrays. Must be converted to an array.
        X = torch.tensor(np.array(X), dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)  # Must be reshaped
        y = y.reshape(batch_size, 1)

        # Set critic network to train mode. (Gradient is needed).
        self.critic.train()
        # Compute Q_sa with X
        Q_sa = self.critic(X)
        # Compute loss as MSE between Q_sa and target y
        self.critic_optimizer.zero_grad()
        loss = self.criterion(y, Q_sa)
        loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        # Is fundamental to deepcopy the state dict in order to update the actor only after having looped all over the
        # batch
        state_dict = copy.deepcopy(self.actor.state_dict())  # Actor state dict before update
        for (state, action, reward, next_state, done) in minibatch:
            state = torch.from_numpy(state).float().to(self.device)
            # Compute sampled policy gradient
            # 1) Compute gradient of critic with respect to the action = mu
            mu = self.actor(state)
            mu.retain_grad()
            self.critic.zero_grad()
            state_action_value = self.critic(torch.cat([state, mu]))
            state_action_value.backward()
            # Compute gradient of critic with respect to action mu
            grad_critic = mu.grad  # Is a tensor of size [1]
            # 2) For each parameter of actor, compute gradient of actor with respect of that parameter and update
            self.actor.zero_grad()
            mu = self.actor(state)
            mu.backward()  # Compute gradient
            with torch.no_grad():
                for param, key in zip(self.actor.parameters(), state_dict):
                    # theta_new = theta_old + lr * sampled_policy_gradient
                    state_dict[key] = state_dict[key] + self.learning_rate * (1 / batch_size) * grad_critic * param.grad
        # Update actor after for cycle
        self.actor.load_state_dict(state_dict)

    def save_networks(self, save_name: str, save_dir: str = 'models', info = 0):
        """
        Save Actor and Critic state dictionaries
        :param save_dir: target directory
        :param info: used to add additional info to file (episode / final)
        """
        torch.save(self.actor.state_dict(), os.path.join(save_dir, f'{save_name}_{info}_actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, f'{save_name}_{info}_critic.pt'))

    def load_networks(self, actor_filepath: str, critic_filepath: str):
        """
        Used to load Actor and Critic networks from saved files
        """
        # Actor
        self.actor.load_state_dict(torch.load(actor_filepath))
        self.target_actor.load_state_dict(torch.load(actor_filepath))
        # Critic
        self.critic.load_state_dict(torch.load(critic_filepath))
        self.target_critic.load_state_dict(torch.load(critic_filepath))

    def train(self,
              env,
              num_episodes: int,
              save_freq: int = 100,
              batch_size: int = 32,
              save_dir: str = 'models',
              save_name: str = 'DDPG',
              stop_with_full_memory: bool = True) -> None:
        """
        We decided to add an extra penalty term for the distance between the cart and the origin.
        :param env: environment in which to be trained
        :param num_episodes: max number of training episodes
        :param save_freq: saving frequency of networks
        :param batch_size: batch size used for replay function
        :param save_dir: saving directory
        :param save_name: prefix for saving name
        :param stop_with_full_memory: if True, stop training before memory is full.
        """

        # Create save_dir if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Loop over episodes
        for episode in range(1, num_episodes + 1):
            # Stop episode generation if memory is full
            if stop_with_full_memory & (len(self.memory) > (self.memory.maxlen - 1000)):
                break

            # Keep track of time spent
            t1 = time.perf_counter()

            # Initialize environment
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_return = 0
            episode_steps = 0

            # Loop over steps
            while not (terminated | truncated):
                # Choose action with noise
                action: float = self.act(state, True)
                # Execute action and observe reward and next state
                next_state, reward, terminated, truncated, _ = env.step((action, ))

                # Add penalty to cart if it is far from origin penalty = 10 * x^2 where x is the distance.
                # Set reward to -10 if episode terminates (fail).
                reward = reward - 10 * next_state[0]**2 if not terminated else -10

                # Store transition to replay buffer
                self.remember(state, action, reward, next_state, terminated)

                if len(self.memory) > batch_size:
                    self.replay(batch_size)  # Update Critic and Actor
                    self.update_target_networks()  # Update target networks

                state = next_state
                episode_return += reward
                episode_steps += 1

            # Store episode data
            self.training_returns.append(episode_return)
            self.training_steps.append(episode_steps)
            t2 = time.perf_counter()
            training_time = t2 - t1
            self.training_time.append(training_time)
            # Print episode info
            print(f'Episode {episode}/{num_episodes} | Return: {episode_return:.2f} | Steps: {episode_steps}'
                  f'| Memory: {len(self.memory)} | Time: {training_time:.2f} seconds | Cart Position: {state[0]:.2f}')

            # Save networks
            if episode % save_freq == 0:
                self.save_networks(save_name, save_dir, episode)
                print(f'Networks saved at episode {episode}')

        # Save final networks
        self.save_networks(save_name, save_dir, 'final')
        print(f'Final networks saved')

    def generate_greedy_episode(self,
                                env,
                                malfunction_probability: float = 0,
                                push_probability: float = 0.5,
                                render: bool = False,
                                frame_rate: float = 1 / 30,
                                text_color: str = 'black',
                                background_text_color: str = 'white') -> (float, int):
        """
        Generate an episode selecting action without noise.
        :param text_color: text color of episode data.
        :param frame_rate: the lower, the higher the environment is displayed.
        :param push_probability: probability to have malfunction of type 'push'.
        :param malfunction_probability: probability to have malfunction actuators (push too much or do nothing).
        :param render: if True, show environment
        :param background_text_color: background color of text when displaying environment.
        :param env: the environment.
        :return: episode return, episode length
        """
        # Initialize environment
        state, _ = env.reset()
        terminated, truncated = False, False
        episode_return = 0
        episode_steps = 0
        action = 0
        while not (terminated | truncated):

            # Display if render
            if render:
                time.sleep(frame_rate)
                clear_output()
                plt.imshow(env.render())
                plt.text(100, 400,
                         f'Return = {episode_return:.2f}\nSteps = {episode_steps}\n'
                         f'Cart Position = {state[0]:.2f}\nAction = {action:.2f}\n'
                         f'Angle1 = {np.arcsin(state[1]):.2f}\nAngle2 = {np.arcsin(state[2]):.2f}',
                         backgroundcolor=background_text_color, color=text_color)
                plt.show()

            # Choose action
            action = self.act(state, False)  # Important: training = False to remove noise
            if np.random.random() < malfunction_probability:
                # Simulate two kind of malfunctions: push too much and or do not push at all.
                if np.random.random() < push_probability:
                    action = action / np.abs(action) if action != 0 else action
                else:
                    action = 0.
            # Take action
            next_state, reward, terminated, truncated, _ = env.step((action, ))
            reward = reward - 10 * next_state[0] ** 2 if not terminated else -10
            episode_return = self.gamma * episode_return + reward
            episode_steps += 1
            state = next_state

        # Handle terminal state
        if render:
            time.sleep(frame_rate)
            clear_output()
            plt.imshow(env.render())
            plt.text(100, 400,
                     f'Return = {episode_return:.2f}\nSteps = {episode_steps}\n'
                     f'Cart Position = {state[0]:.2f}\nAction = {action:.2f}',
                     backgroundcolor=background_text_color, color=text_color)
            plt.show()
        return episode_return, episode_steps
