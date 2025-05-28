from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

class ConsistencyCheckerAgent:
    """Agent for checking consistency in ABET documentation using Expected SARSA."""
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table using defaultdict for automatic initialization
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Action space
        self.actions = [
            {'clo_index': i, 'issue_type': j, 'flag': k}
            for i in range(10)  # 10 CLOs
            for j in range(4)   # 4 issue types
            for k in range(2)   # flag or not
        ]
        
    def _encode_state(self, state: Dict) -> str:
        """Encode state into a hashable string."""
        return (
            f"clo_plo:{state['clo_plo_mapping'].tobytes()},"
            f"assess:{state['assessment_methods'].tobytes()},"
            f"scores:{state['scores_valid'].tobytes()},"
            f"syllabus:{state['syllabus_match'].tobytes()}"
        )
        
    def _encode_action(self, action: Dict) -> str:
        """Encode action into a hashable string."""
        return f"{action['clo_index']}_{action['issue_type']}_{action['flag']}"
        
    def select_action(self, state: Dict) -> Dict:
        """Select action using epsilon-greedy policy with Expected SARSA."""
        state_key = self._encode_state(state)
        
        if np.random.random() < self.epsilon:
            # Random action
            return np.random.choice(self.actions)
            
        # Get Q-values for current state
        q_values = {self._encode_action(a): self.q_table[state_key][self._encode_action(a)]
                   for a in self.actions}
        
        # Calculate expected value for each action
        expected_values = {}
        for action in self.actions:
            action_key = self._encode_action(action)
            # Calculate expected value using current policy
            expected_value = 0
            for next_action in self.actions:
                next_action_key = self._encode_action(next_action)
                # Probability of taking next_action under current policy
                if q_values[next_action_key] == max(q_values.values()):
                    prob = 1 - self.epsilon + self.epsilon / len(self.actions)
                else:
                    prob = self.epsilon / len(self.actions)
                expected_value += prob * q_values[next_action_key]
            expected_values[action_key] = expected_value
            
        # Select action with highest expected value
        best_action_key = max(expected_values.items(), key=lambda x: x[1])[0]
        clo_idx, issue_type, flag = map(int, best_action_key.split('_'))
        return {
            'clo_index': clo_idx,
            'issue_type': issue_type,
            'flag': flag
        }
        
    def update(self, state: Dict, action: Dict, reward: float,
              next_state: Dict, next_action: Dict, done: bool):
        """Update Q-values using Expected SARSA."""
        state_key = self._encode_state(state)
        action_key = self._encode_action(action)
        next_state_key = self._encode_state(next_state)
        
        # Get Q-values for next state
        next_q_values = {self._encode_action(a): self.q_table[next_state_key][self._encode_action(a)]
                        for a in self.actions}
        
        # Calculate expected value of next state
        expected_next_value = 0
        for next_a in self.actions:
            next_a_key = self._encode_action(next_a)
            # Probability of taking next_a under current policy
            if next_q_values[next_a_key] == max(next_q_values.values()):
                prob = 1 - self.epsilon + self.epsilon / len(self.actions)
            else:
                prob = self.epsilon / len(self.actions)
            expected_next_value += prob * next_q_values[next_a_key]
            
        # Update Q-value
        current_q = self.q_table[state_key][action_key]
        if done:
            target = reward
        else:
            target = reward + self.gamma * expected_next_value
            
        self.q_table[state_key][action_key] = current_q + self.learning_rate * (target - current_q)
        
    def save_model(self, path: str):
        """Save Q-table to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f)
            
    def load_model(self, path: str):
        """Load Q-table from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
            self.learning_rate = data['learning_rate']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon'] 