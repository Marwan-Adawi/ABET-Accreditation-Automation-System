from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from datetime import datetime

class DocGenNetwork(nn.Module):
    """Neural network for document generation and management."""
    
    def __init__(self):
        super().__init__()
        
        # Content encoders
        self.clo_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.plo_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.assessment_encoder = nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            nn.Linear(24, 12)
        )
        
        self.material_encoder = nn.Sequential(
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        self.prerequisite_encoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 6)
        )
        
        # Quality metrics encoder
        self.quality_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Feedback encoder
        self.feedback_encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Action heads
        self.operation_head = nn.Sequential(
            nn.Linear(92, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        
        self.section_head = nn.Sequential(
            nn.Linear(92, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        self.content_head = nn.Sequential(
            nn.Linear(92, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Tanh()
        )
        
        self.metadata_head = nn.Sequential(
            nn.Linear(92, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 3 for priority + 4 for review_status
        )
        
    def forward(self, state: Dict) -> Dict[str, torch.Tensor]:
        # Encode content
        clo_features = self.clo_encoder(state['content']['clos'])
        plo_features = self.plo_encoder(state['content']['plos'])
        assessment_features = self.assessment_encoder(state['content']['assessments'])
        material_features = self.material_encoder(state['content']['materials'])
        prerequisite_features = self.prerequisite_encoder(state['content']['prerequisites'])
        
        # Encode quality metrics
        quality_features = self.quality_encoder(torch.cat([
            state['quality']['completeness'],
            state['quality']['consistency'],
            state['quality']['clarity'],
            state['quality']['alignment']
        ]))
        
        # Encode metadata
        metadata_features = self.metadata_encoder(torch.cat([
            state['metadata']['last_modified'],
            state['metadata']['version'],
            torch.tensor([state['metadata']['review_status']], dtype=torch.float32),
            torch.tensor([state['metadata']['priority']], dtype=torch.float32)
        ]))
        
        # Encode feedback
        feedback_features = self.feedback_encoder(torch.cat([
            state['feedback']['instructor'],
            state['feedback']['coordinator'],
            state['feedback']['evaluator']
        ]))
        
        # Combine features
        combined = torch.cat([
            clo_features, plo_features, assessment_features,
            material_features, prerequisite_features,
            quality_features, metadata_features, feedback_features
        ])
        
        # Generate actions
        operation_logits = self.operation_head(combined)
        section_logits = self.section_head(combined)
        content_values = self.content_head(combined)
        metadata_values = self.metadata_head(combined)
        
        return {
            'operation': operation_logits,
            'section': section_logits,
            'content': content_values,
            'metadata_update': {
                'priority': metadata_values[:3],
                'review_status': metadata_values[3:]
            }
        }

class DocGenAgent:
    """Agent for generating and managing ABET documentation."""
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 10):
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.current_user = "Marwan-Adawi"
        
        # Initialize networks
        self.policy_net = DocGenNetwork()
        self.target_net = DocGenNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize step counter
        self.steps_done = 0
        
    def select_action(self, state: Dict) -> Dict:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Random action
            return {
                'operation': random.randint(0, 5),
                'section': random.randint(0, 4),
                'content': np.random.uniform(-1, 1, 10),
                'metadata_update': {
                    'priority': random.randint(0, 2),
                    'review_status': random.randint(0, 3)
                }
            }
        
        with torch.no_grad():
            # Convert state to tensors
            state_tensors = {
                'content': {
                    k: torch.FloatTensor(v) for k, v in state['content'].items()
                },
                'quality': {
                    k: torch.FloatTensor(v) for k, v in state['quality'].items()
                },
                'metadata': {
                    k: torch.FloatTensor(v) if isinstance(v, np.ndarray) else torch.tensor([v], dtype=torch.float32)
                    for k, v in state['metadata'].items()
                },
                'feedback': {
                    k: torch.FloatTensor(v) for k, v in state['feedback'].items()
                }
            }
            
            # Get action values
            action_values = self.policy_net(state_tensors)
            
            # Convert to numpy arrays
            return {
                'operation': action_values['operation'].argmax().item(),
                'section': action_values['section'].argmax().item(),
                'content': action_values['content'].numpy(),
                'metadata_update': {
                    'priority': action_values['metadata_update']['priority'].argmax().item(),
                    'review_status': action_values['metadata_update']['review_status'].argmax().item()
                }
            }
    
    def optimize_model(self):
        """Perform one step of optimization."""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert batch to tensors
        state_batch = self._prepare_state_batch(batch[0])
        action_batch = self._prepare_action_batch(batch[1])
        reward_batch = torch.FloatTensor(batch[2])
        next_state_batch = self._prepare_state_batch(batch[3])
        done_batch = torch.FloatTensor(batch[4])
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            
        # Compute expected Q values
        expected_q_values = self._compute_expected_q_values(
            current_q_values, next_q_values, reward_batch, done_batch
        )
        
        # Compute loss
        loss = self._compute_loss(current_q_values, expected_q_values, action_batch)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, 
                         self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.steps_done += 1
        
    def _prepare_state_batch(self, states: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Convert state batch to tensors."""
        return {
            'content': {
                k: torch.FloatTensor(np.stack([s['content'][k] for s in states]))
                for k in states[0]['content'].keys()
            },
            'quality': {
                k: torch.FloatTensor(np.stack([s['quality'][k] for s in states]))
                for k in states[0]['quality'].keys()
            },
            'metadata': {
                k: (torch.FloatTensor(np.stack([s['metadata'][k] for s in states]))
                    if isinstance(states[0]['metadata'][k], np.ndarray)
                    else torch.tensor([[s['metadata'][k]] for s in states], dtype=torch.float32))
                for k in states[0]['metadata'].keys()
            },
            'feedback': {
                k: torch.FloatTensor(np.stack([s['feedback'][k] for s in states]))
                for k in states[0]['feedback'].keys()
            }
        }
        
    def _prepare_action_batch(self, actions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert action batch to tensors."""
        return {
            'operation': torch.LongTensor([a['operation'] for a in actions]),
            'section': torch.LongTensor([a['section'] for a in actions]),
            'content': torch.FloatTensor(np.stack([a['content'] for a in actions])),
            'metadata_update': {
                'priority': torch.LongTensor([a['metadata_update']['priority'] for a in actions]),
                'review_status': torch.LongTensor([a['metadata_update']['review_status'] for a in actions])
            }
        }
        
    def _compute_expected_q_values(self,
                                 current_q_values: Dict[str, torch.Tensor],
                                 next_q_values: Dict[str, torch.Tensor],
                                 reward_batch: torch.Tensor,
                                 done_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute expected Q values for each action component."""
        # For discrete actions (operation, section, metadata updates)
        expected_operation = current_q_values['operation'].clone()
        expected_section = current_q_values['section'].clone()
        expected_metadata_priority = current_q_values['metadata_update']['priority'].clone()
        expected_metadata_status = current_q_values['metadata_update']['review_status'].clone()
        
        # For continuous actions (content)
        expected_content = current_q_values['content'].clone()
        
        # Update expected values using Bellman equation
        next_value = (torch.max(next_q_values['operation'], dim=1)[0] +
                     torch.max(next_q_values['section'], dim=1)[0] +
                     torch.mean(next_q_values['content'], dim=1))
                     
        expected_value = reward_batch + (1 - done_batch) * self.gamma * next_value
        
        return {
            'operation': expected_operation,
            'section': expected_section,
            'content': expected_content,
            'metadata_update': {
                'priority': expected_metadata_priority,
                'review_status': expected_metadata_status
            }
        }
        
    def _compute_loss(self,
                     current_q_values: Dict[str, torch.Tensor],
                     expected_q_values: Dict[str, torch.Tensor],
                     action_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total loss for all action components."""
        # Cross entropy loss for discrete actions
        operation_loss = nn.CrossEntropyLoss()(
            current_q_values['operation'],
            action_batch['operation']
        )
        
        section_loss = nn.CrossEntropyLoss()(
            current_q_values['section'],
            action_batch['section']
        )
        
        metadata_priority_loss = nn.CrossEntropyLoss()(
            current_q_values['metadata_update']['priority'],
            action_batch['metadata_update']['priority']
        )
        
        metadata_status_loss = nn.CrossEntropyLoss()(
            current_q_values['metadata_update']['review_status'],
            action_batch['metadata_update']['review_status']
        )
        
        # MSE loss for continuous actions
        content_loss = nn.MSELoss()(
            current_q_values['content'],
            action_batch['content']
        )
        
        # Combine losses with weights
        total_loss = (operation_loss * 1.0 +
                     section_loss * 1.0 +
                     content_loss * 2.0 +
                     metadata_priority_loss * 0.5 +
                     metadata_status_loss * 0.5)
                     
        return total_loss
        
    def store_transition(self, state: Dict, action: Dict, reward: float,
                        next_state: Dict, done: bool):
        """Store transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def save_model(self, path: str):
        """Save model to disk."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'memory': self.memory,
            'current_user': self.current_user,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }, path)
        
    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.memory = checkpoint['memory']
        self.current_user = checkpoint['current_user']