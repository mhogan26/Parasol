import time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque
from data_reporter import DataReporter
from simulator import *

# constants
NUM_TRAINING_EPISODES = 100
REPORT_FILE = "results.csv"

# Distribution of the packet: header information

class NetworkingEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, opt_info, o, sim_process):
        super(NetworkingEnv, self).__init__()
        self.opt_info = opt_info
        self.o = o
        self.sim_process = sim_process
        # Just deciding the number of packet, 0 for not sending
        # having the maximum number of packets
        # sending will clean the filter, if they send packet to clean sketch, should be enough such that they clean,
        # but not too much to overwhelm the filter
        # Fix the number of the number packet
        self.action_space = gym.spaces.Discrete(501)  # Actions sending from 0 to 500 packets
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        # 5 * budget + 5 * action + 5* error rate
        self.initial_budget = 500
        self.budget = 500
        self.state_dim = 15  # 3 * [budget, action, avg_err]
        self.state = np.zeros(self.state_dim)  # Initialize state with zeros
        self.observation_space = gym.spaces.Box(low=np.zeros(self.state_dim), high=np.array([500]*10 + [np.inf]*5), dtype=np.float32)

        self.reward = 0
        self.episode = 0
        self.round_counter = 0
        self.sequence_errors = []

        self.reporter = DataReporter()

    def reset(self):
        self.state = np.zeros(self.state_dim)  # Reset state for three rounds
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        self.round_counter += 1
        done = self._is_done()
        bits_to_send = action
        # Simulate sending packets and calculate bits set and average error
        self.bits_set, self.avg_err, num_backgroundpkts = send_pkts(bits_to_send, self.opt_info, self.o, self.sim_process)
        self.sequence_errors.append(self.avg_err)

        # Calculate reward
        self.reward = self._get_reward()
        
        # Update budget, decrement by action taken
        if action > self.budget:
            action = self.budget  # Cap the action at the current budget level
        self.budget -= action
        # Encourage the agent to send as few as possible
        if self.budget <= 0:
            action = 0

        # Shift the state to include the new values and remove the oldest values
        new_state = np.roll(self.state, -3)
        new_state[-3:] = [self.budget, action, self.avg_err]
        self.state = new_state
        
        self.reporter.add_frame(
            self.reward, num_backgroundpkts, self.budget, action
        )
        
        if done:
            # Reset for a new episode
            self.sequence_errors = []
            self.round_counter = 0
            self.episode += 1
            self.budget = self.initial_budget

        return self.state, self.reward, num_backgroundpkts, done, {}
   
    def _take_action(self, action):
        bits_to_send = action  # Action is now directly the number of packets to send

    def _get_reward(self):
        # Calculate and return the reward for the current state     
        return self.avg_err

    def _is_done(self):
        # Check if the episode is done based on the round counter
        if self.round_counter >= 5:  # Assuming an episode ends after 5 rounds
            return True
        return False


    def render(self, mode='console'):
        # Implement rendering for human consumption, if necessary
        if mode != 'console':
            raise NotImplementedError()
        print(f"Current state: {self.state}")

# Define the DQN Neural Network Model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, hidden_size=64, gamma=0.99, epsilon=0.1, learning_rate=1e-3, memory_size=10000, batch_size=64, target_update=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)

        # Get the shape of the observation space and action space
        # observation_space_dim = env.observation_space.n
        observation_space_dim = env.state_dim 
        action_space_dim = env.action_space.n
        
        # print(observation_space_dim)
        # print(action_space_dim)
        # Initialize DQN and Target Networks with correct dimensions
        self.policy_net = DQN(observation_space_dim, hidden_size, action_space_dim)
        self.target_net = DQN(observation_space_dim, hidden_size, action_space_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.steps_done = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, test=False):
        self.steps_done += 1
        sample = random.random()
        eps_threshold = self.epsilon
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_values = self.policy_net(state)
                return action_values.max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.env.action_space.n)

    # NOT IN USE
    def one_hot_encode(self, state, num_states):
        """One-hot encode a discrete state."""
        one_hot = torch.zeros(num_states, dtype=torch.float32)
        one_hot[state] = 1.0
        return one_hot

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        # print("batch 0", batch[0])
        # print("batch 1", batch[1])
        # print("batch 2", batch[2])
        # print("batch 3", batch[3])
        # print("batch 4", batch[4])
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch[3] if s is not None])
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch[0]])
        action_batch = torch.cat([torch.tensor([a], dtype=torch.long) for a in batch[1]])
        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch[2]])


        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, episodes=100):
        for e in range(episodes):
            start = time.time()
            print("episode", e)
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                print("i am training...")
                next_state, reward, num_backgroundpkts, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
            self.experience_replay()
            end = time.time()
            t = end - start
            print(f"Training Time for 1 episode is {t} seconds!!")

    def test(self, episodes=10):
        total_rewards = 0
        state_seq = []
        for episode in range(episodes):
            state = self.env.reset()
            episode_rewards = 0
            done = False
            while not done:
                # action = 0 # For no attacker baseline
                action = self.choose_action(state, test=True) # Notice the test flag
                next_state, reward, num_backgroundpkts, done, _ = self.env.step(action)
                episode_rewards += reward
                state_seq.append([num_backgroundpkts, state])
                state = next_state
            total_rewards += episode_rewards
            print(f"Episode {episode + 1}: Total Reward = {episode_rewards}")
        avg_reward = total_rewards / episodes
        # print(f"Background Traffic and State Seq over {episodes} episodes: {state_seq}")
        print(f"Average Reward over {episodes} episodes: {avg_reward}")

# Define the A2C Neural Network Model
class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value
        )

    def forward(self, x):
        action_log_probs = self.actor(x)
        state_values = self.critic(x)
        return action_log_probs, state_values

class ActorCriticAgent:
    def __init__(self, env, hidden_size=128, gamma=0.99, learning_rate=1e-3, memory_size=10000, batch_size=64, target_update=10):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)

        observation_space_dim = env.state_dim
        action_space_dim = env.action_space.n
        
        self.model = ActorCritic(observation_space_dim, hidden_size, action_space_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.steps_done = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state)
            if test:
                # Choose the action with the highest probability during testing
                action = torch.argmax(action_probs).item()
            else:
                # Sample an action according to the distribution during training
                action = torch.distributions.Categorical(action_probs).sample().item()
        return action

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch[3] if s is not None])
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch[0]])
        action_batch = torch.cat([torch.tensor([a], dtype=torch.long) for a in batch[1]])
        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch[2]])

        # Predict action probabilities and state values for the current states
        current_action_probs, current_state_values = self.model(state_batch)
        current_state_values = current_state_values.squeeze(1)
        current_log_probs = torch.log(current_action_probs.gather(1, action_batch.unsqueeze(-1)).squeeze(-1))
        
        # Predict state values for the next states
        with torch.no_grad():
            _, next_state_values = self.model(non_final_next_states)
            next_state_values = next_state_values.squeeze(1)
            next_state_values = torch.cat((next_state_values, torch.zeros(self.batch_size)[non_final_mask.logical_not()]))
        
        # Calculate expected state values
        expected_state_values = reward_batch + self.gamma * next_state_values
        
        # Calculate losses
        value_loss = nn.MSELoss()(current_state_values, expected_state_values.detach())
        policy_loss = -(current_log_probs * (expected_state_values.detach() - current_state_values)).mean()
        
        # Perform backprop
        self.optimizer.zero_grad()
        total_loss = value_loss + policy_loss
        total_loss.backward()
        # Clip gradients here, right before the optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()
        
        if self.steps_done % self.target_update == 0:
            self.model.load_state_dict(self.model.state_dict())

    def train(self, episodes=100):
        for e in range(episodes):
            start = time.time()
            print("episode", e)
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                print("i am training A2C...")
                next_state, reward, num_backgroundpkts, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
            self.experience_replay()
            end = time.time()
            print(f"Training Time for 1 episode is {end - start} seconds!!")


    def test(self, episodes=10):
        total_rewards = 0
        state_seq = []
        for episode in range(episodes):
            state = self.env.reset()
            episode_rewards = 0
            done = False
            while not done:
                action = self.choose_action(state, test=True)
                next_state, reward, num_backgroundpkts, done, _ = self.env.step(action)
                episode_rewards += reward
                state_seq.append([num_backgroundpkts, state])
                state = next_state
            total_rewards += episode_rewards
            print(f"Episode {episode + 1}: Total Reward = {episode_rewards}")
        avg_reward = total_rewards / episodes
        # print(f"Background Traffic and State Seq over {episodes} episodes: {state_seq}")
        print(f"A2C Average Reward over {episodes} episodes: {avg_reward}")

# Define the PPO Neural Network Model
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque

class PPO(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPO, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_values = self.critic(x)
        return action_probs, state_values

class PPOAgent:
    def __init__(self, env, hidden_size=128, gamma=0.99, learning_rate=1e-3, clip_epsilon=0.2, update_frequency=2048, k_epochs=10, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_frequency = update_frequency
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        observation_space_dim = env.state_dim
        action_space_dim = env.action_space.n
        
        self.model = PPO(observation_space_dim, hidden_size, action_space_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def train(self, episodes=100):
        for e in range(episodes):
            self.env.reporter.start_or_advance_step(episodes)
            state = self.env.reset()
            done = False
            while not done:
                action, log_prob = self.choose_action(state)
                next_state, reward, num_backgroundpkts, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done, log_prob)
                state = next_state
                if len(self.memory) >= self.update_frequency:
                    self.experience_replay()

    def experience_replay(self):
        transitions = list(self.memory)
        self.memory.clear()

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_prob_batch = zip(*transitions)
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)
        old_log_probs = torch.stack(log_prob_batch)

        for _ in range(self.k_epochs):
            _, state_values = self.model(state_batch)
            state_values = state_values.squeeze()
            advantages = reward_batch + self.gamma * state_values * (1 - done_batch) - state_values.detach()

            new_probs, _ = self.model(state_batch)
            new_dist = Categorical(new_probs)
            new_log_probs = new_dist.log_prob(action_batch)

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(state_values, reward_batch + self.gamma * state_values * (1 - done_batch))

            self.optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            self.optimizer.step()

    def test(self, episodes=10):
        total_rewards = 0
        state_seq = []
        for episode in range(episodes):
            state = self.env.reset()
            episode_rewards = 0
            done = False
            while not done:
                action, _ = self.choose_action(state)
                next_state, reward, num_backgroundpkts, done, _ = self.env.step(action)
                episode_rewards += reward
                state_seq.append([num_backgroundpkts, state])
                state = next_state
            total_rewards += episode_rewards
            print(f"Episode {episode + 1}: Total Reward = {episode_rewards}")
        avg_reward = total_rewards / episodes
        # print(f"Background Traffic and State Seq over {episodes} episodes: {state_seq}")
        print(f"PPO Average Reward over {episodes} episodes: {avg_reward}")


def main():
    parser = argparse.ArgumentParser(description="RL agent for sketchprobe")
    parser.add_argument("optfile", metavar="optfile", help="name of json file with sketchprobe initialization parameters")
    args = parser.parse_args()

    # start the interpreter
    opt_info, o, sim_process = init_simulation(args.optfile)

    # Initialize RL env
    env = NetworkingEnv(opt_info, o, sim_process)
    # Initialize the Q-learning agent
    # agent = QLearningAgent(env)

    # Initialize and train with Different RL Agent
    # agent = DQNAgent(env)
    # agent = ActorCriticAgent(env)
    agent = PPOAgent(env)

    # Train the Q-learning agent
    start = time.time() # Record Time for Training
    agent.train(episodes=NUM_TRAINING_EPISODES)
    end = time.time() # Record Time for Training
    training_time = end - start
    print(f"Training Time for {NUM_TRAINING_EPISODES} episodes is {training_time} seconds!!")
    
    # Test the trained Agent
    print("\nTesting the trained agent...")
    agent.test()
    env.reporter.report(REPORT_FILE)

    end_simulation(sim_process)

if __name__ == "__main__":
    main()



'''
### OLD CODE:
import gym
import numpy as np

# Distribution of the packet: header information


class NetworkingEnv(gym.Env):
    def __init__(self, opt_info, o, sim_process):
        super(NetworkingEnv, self).__init__()
        self.opt_info = opt_info
        self.o = o
        self.sim_process = sim_process
        # Just deciding the number of packet, 0 for not sending
        # having the maximum number of packets
        # sending will clean the filter, if they send packet to clean sketch, should be enough such that they clean,
        # but not too much to overwhelm the filter
        # Fix the number of the number packet
        self.action_space = gym.spaces.Discrete(2)  # 0: Don't send packets, 1: Send packets
        self.observation_space = gym.spaces.Discrete(8)  # Observation space size 8
        self.reward = 0
        self.episode = 0
        self.round_counter = 0
        self.sequence_errors = []
        self.pattern = []

    def reset(self):
        return self.observation_space.sample()  # Placeholder observation

    def step(self, action):
        # Placeholder values for the next observation and done flag
        next_observation = self.observation_space.sample()
        done = False
        
        # Determine whether to send or not send based on the action
        if action == 1:
            # Change 260 to 500 if not working
            bits_to_send = 500
        else:
            bits_to_send = 0

        # Calculate error for this case
        bits_set, avg_err = send_pkts(bits_to_send, self.opt_info, self.o, self.sim_process)
        self.sequence_errors.append(avg_err)
        if self.round_counter % 2 == 0:
            self.reward = avg_err
        else:
            self.reward = 0
        print("c", self.round_counter)
        print("re", self.reward)
        self.round_counter += 1
        self.pattern.append(action)
    
        if self.round_counter == 5: 
            done = True  
            print("Episode #:", self.episode)
            print("Sequence Errors:", self.sequence_errors)
            self.sequence_errors = []
            self.reward = 0
            self.round_counter = 0  # Reset the round counter
            self.pattern = []  # Reset the pattern
            self.episode += 1

        # Update pattern with sliding window (keep only the latest 3 windows)
        self.pattern = self.pattern[-3:]
        print("Pattern:", ''.join(map(str, self.pattern)))
        # Encode binary sequence as a decimal number
        if self.pattern:
            print("lalala", self.pattern)
            observation = int(''.join(map(str, self.pattern)), 2)
            print("sososo", observation)
        else:
            observation = 0
        print("ob", observation)

        return observation, self.reward, done, {}

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((8, env.action_space.n))  # 2^n is the number of possible observations, n is the length of window

    def choose_action(self, observation):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Choose random action with epsilon probability
        else:
            return np.argmax(self.q_table[observation])  # Choose action with highest Q-value

    def train(self, episodes=100):
        for episode in range(episodes):
            observation = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                q_predict = self.q_table[observation][action]
                q_target = reward + self.gamma * np.max(self.q_table[next_observation])
                self.q_table[observation][action] += self.alpha * (q_target - q_predict)
                print("qq", self.q_table)
                observation = next_observation
                print(observation)

        return self.q_table

    def test(self, episodes=10):
        for episode in range(episodes):
            observation = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.q_table[observation])
                next_observation, _, done, _ = self.env.step(action)
                observation = next_observation



def main():
    parser = argparse.ArgumentParser(description="optimization of lucid symbolics in python, default uses layout script instead of compiler")
    parser.add_argument("optfile", metavar="optfile", help="name of json file with optimization info")
    args = parser.parse_args()

    # start the interpreter
    opt_info, o, sim_process = init_simulation(args.optfile)

    # Initialize RL env
    env = NetworkingEnv(opt_info, o, sim_process)
    # Initialize the Q-learning agent
    agent = QLearningAgent(env)

    # Train the Q-learning agent
    num_episodes = 100 # change episodes
    agent.train(episodes=num_episodes)
    
    counter = 0
    # Test Sketch Running
    while True:
        # generate a trace w/ 260 pkts
        bits_set, avg_err = send_pkts(500, opt_info, o, sim_process)
        print(bits_set, avg_err)
        counter += 1
        if counter >= 5:
            break

    end_simulation(sim_process)


if __name__ == "__main__":
    main()

'''
