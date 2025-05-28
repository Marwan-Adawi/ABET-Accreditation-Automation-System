from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from .environments.action_planner_env import ActionPlannerEnvironment

class ActionPlannerNetwork(nn.Module):
    """Neural network for action planning."""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        
        # State feature extractors
        self.inconsistency_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.document_encoder = nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            nn.Linear(24, 12)
        )
        
        self.resource_encoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 6)
        )
        
        self.stakeholder_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.priority_encoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 6)
        )
        
        # Action output heads
        self.action_type_head = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        
        self.resource_head = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
        
        self.stakeholder_head = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        
        self.timeline_head = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
    def forward(self, state: Dict) -> Dict[str, torch.Tensor]:
        # Encode state components
        inc_features = self.inconsistency_encoder(state['inconsistency_type'])
        doc_features = self.document_encoder(state['document_state'])
        res_features = self.resource_encoder(state['resources'])
        stake_features = self.stakeholder_encoder(state['stakeholders'])
        prio_features = self.priority_encoder(state['priority'])
        
        # Combine features
        combined = torch.cat([
            inc_features, doc_features, res_features,
            stake_features, prio_features
        ], dim=-1)
        
        # Generate action components
        return {
            'action_type': self.action_type_head(combined),
            'resource_allocation': self.resource_head(combined),
            'stakeholder_assignment': self.stakeholder_head(combined),
            'timeline': self.timeline_head(combined)
        }

class ActionPlannerAgent:
    """Agent for planning corrective actions in ABET documentation."""
    
    def __init__(self,
                 env: ActionPlannerEnvironment,
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
        
        # Initialize networks
        state_size = self._calculate_state_size()
        action_size = self._calculate_action_size()
        
        self.network = ActionPlannerNetwork(state_size, action_size)
        self.target_network = ActionPlannerNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
    def _calculate_state_size(self) -> int:
        """Calculate the size of the flattened state vector."""
        return (4 +  # inconsistency_type
                6 +  # document_state
                3 +  # resources
                4 +  # stakeholders
                3)   # priority
                
    def _calculate_action_size(self) -> int:
        """Calculate the size of the flattened action vector."""
        return (6 +  # action_type
                3 +  # resource_allocation
                4 +  # stakeholder_assignment
                4)   # timeline
                
    def select_action(self, state: Dict) -> Dict:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Random action
            return {
                'action_type': random.randint(0, 5),
                'resource_allocation': np.random.rand(3),
                'stakeholder_assignment': np.random.randint(0, 2, 4),
                'timeline': random.randint(0, 3)
            }
        else:
            with torch.no_grad():
                # Convert state to tensor
                state_tensor = {k: torch.FloatTensor(v) for k, v in state.items()}
                
                # Get action components
                action_components = self.network(state_tensor)
                
                return {
                    'action_type': action_components['action_type'].argmax().item(),
                    'resource_allocation': action_components['resource_allocation'].numpy(),
                    'stakeholder_assignment': (action_components['stakeholder_assignment'] > 0.5).numpy().astype(int),
                    'timeline': action_components['timeline'].argmax().item()
                }
                
    def train(self, num_episodes: int = 1000) -> List[float]:
        """Train the agent."""
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
        
        action_type_batch = torch.tensor([a['action_type'] for a in action_batch])
        resource_batch = torch.FloatTensor(np.stack([a['resource_allocation'] for a in action_batch]))
        stakeholder_batch = torch.FloatTensor(np.stack([a['stakeholder_assignment'] for a in action_batch]))
        timeline_batch = torch.tensor([a['timeline'] for a in action_batch])
        
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32)
        done_tensor = torch.tensor(done_batch, dtype=torch.float32)
        
        # Get current Q values
        current_q_values = self.network(state_tensor)
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor)
            
        # Calculate losses for each action component
        action_type_loss = nn.CrossEntropyLoss()(
            current_q_values['action_type'],
            action_type_batch
        )
        
        resource_loss = nn.MSELoss()(
            current_q_values['resource_allocation'],
            resource_batch
        )
        
        stakeholder_loss = nn.BCELoss()(
            current_q_values['stakeholder_assignment'],
            stakeholder_batch
        )
        
        timeline_loss = nn.CrossEntropyLoss()(
            current_q_values['timeline'],
            timeline_batch
        )
        
        # Combine losses
        total_loss = (action_type_loss + 
                     resource_loss + 
                     stakeholder_loss + 
                     timeline_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
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