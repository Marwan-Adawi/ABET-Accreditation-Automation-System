from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from .environments.coordinator_env import CoordinatorEnvironment

class CoordinatorNetwork(nn.Module):
    """Neural network for the coordinator agent."""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        
        # Encoder for different state components
        self.task_encoder = nn.Sequential(
            nn.Linear(40, 64),  # 10 tasks x 4 status
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(9, 32),  # 3 agents x 3 status
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.dependency_encoder = nn.Sequential(
            nn.Linear(100, 64),  # 10x10 dependency matrix
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Main network
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state_dict: Dict) -> torch.Tensor:
        # Encode different state components
        task_features = self.task_encoder(
            torch.flatten(state_dict['task_status'])
        )
        
        agent_features = self.agent_encoder(
            torch.flatten(state_dict['agent_status'])
        )
        
        dep_features = self.dependency_encoder(
            torch.flatten(state_dict['dependencies'])
        )
        
        # Concatenate all features
        x = torch.cat([
            task_features,
            agent_features,
            dep_features,
            state_dict['task_priorities'],
            state_dict['resource_utilization']
        ])
        
        # Main network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CoordinatorAgent:
    """Agent for coordinating multiple ABET automation agents."""
    
    def __init__(self, 
                 env: CoordinatorEnvironment,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Calculate state and action sizes
        self.state_size = self._calculate_state_size()
        self.action_size = (
            env.num_tasks * 
            env.num_agents * 
            4  # number of action types
        )
        
        # Initialize networks
        self.network = CoordinatorNetwork(self.state_size, self.action_size)
        self.target_network = CoordinatorNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
    def _calculate_state_size(self) -> int:
        """Calculate the size of the flattened state vector."""
        task_status_size = self.env.num_tasks * 4
        agent_status_size = self.env.num_agents * 3
        dependencies_size = self.env.num_tasks * self.env.num_tasks
        priorities_size = self.env.num_tasks
        resource_size = self.env.num_agents
        
        return (task_status_size + 
                agent_status_size + 
                dependencies_size + 
                priorities_size + 
                resource_size)
        
    def select_action(self, state: Dict) -> Dict:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Random action
            task_id = random.randrange(self.env.num_tasks)
            agent_id = random.randrange(self.env.num_agents)
            action_type = random.randrange(4)
            
        else:
            with torch.no_grad():
                # Convert state to tensor
                state_tensor = {k: torch.FloatTensor(v) for k, v in state.items()}
                
                # Get Q-values
                q_values = self.network(state_tensor)
                
                # Get action with highest Q-value
                action_idx = q_values.argmax().item()
                
                # Convert to action dict
                task_id = action_idx // (self.env.num_agents * 4)
                remaining = action_idx % (self.env.num_agents * 4)
                agent_id = remaining // 4
                action_type = remaining % 4
                
        return {
            'task_id': task_id,
            'agent_id': agent_id,
            'action_type': action_type
        }
        
    def train(self, num_episodes: int = 1000) -> List[float]:
        """Train the coordinator agent."""
        rewards_history = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                # Select and take action
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                
                # Store transition
                self.memory.append((state, action, reward, next_state, done))
                
                # Perform optimization
                self._optimize_model()
                
                if done:
                    break
                    
                state = next_state
            
            # Update target network
            if episode % 10 == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon * self.epsilon_decay)
            
            rewards_history.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.2f}")
        
        return rewards_history
        
    def _optimize_model(self):
        """Perform one step of optimization."""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch
        transitions = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        
        # Convert to tensors
        state_tensor = {k: torch.FloatTensor(np.stack([s[k] for s in state_batch]))
                       for k in state_batch[0].keys()}
        next_state_tensor = {k: torch.FloatTensor(np.stack([s[k] for s in next_state_batch]))
                           for k in next_state_batch[0].keys()}
        
        action_tensor = torch.tensor([
            a['task_id'] * self.env.num_agents * 4 +
            a['agent_id'] * 4 +
            a['action_type']
            for a in action_batch
        ])
        
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32)
        done_tensor = torch.tensor(done_batch, dtype=torch.float32)
        
        # Compute current Q values
        current_q_values = self.network(state_tensor).gather(1, action_tensor.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor).max(1)[0]
            target_q_values = reward_tensor + self.gamma * next_q_values * (1 - done_tensor)
            
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path: str):
        """Save model to disk."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']