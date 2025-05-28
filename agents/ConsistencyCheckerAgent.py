import numpy as np
from utils.rewards import get_reward

class ConsistencyAgent:
    def __init__(self):
        self.q_table = {}  # State-action table (can be expanded later)
        self.actions = ['check_mapping', 'resolve_conflict', 'suggest_improvement']

    def step(self, state):
        current_state = self.encode_state(state)
        action = self.select_action(current_state)
        updates, reward, done = self.execute_action(action, state)
        self.update_q_table(current_state, action, reward, self.encode_state(updates))
        return updates, reward, done

    def encode_state(self, state):
        """
        Simplify the environment state to a tuple that identifies major consistency flags.
        """
        return (
            round(state['consistency_score'], 1),
            len(state.get('improvements', []))
        )

    def select_action(self, state_key):
        # Epsilon-greedy action selection
        if np.random.rand() < 0.1 or state_key not in self.q_table:
            return np.random.choice(self.actions)
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def execute_action(self, action, state):
        updates = {}
        reward = -0.1  # Default small penalty

        if action == 'check_mapping':
            # Assume mapping error probability
            if np.random.rand() > 0.7:
                updates['consistency_score'] = max(0, state['consistency_score'] - 0.2)
                reward = -1
            else:
                updates['consistency_score'] = min(1.0, state['consistency_score'] + 0.1)
                reward = 0.5

        elif action == 'resolve_conflict':
            updates['consistency_score'] = min(1.0, state['consistency_score'] + 0.2)
            reward = 1

        elif action == 'suggest_improvement':
            updates['improvements'] = state.get('improvements', []) + ["Suggested fix"]
            reward = 0.2

        done = updates['consistency_score'] >= 1.0
        return updates, reward, done

    def update_q_table(self, s, a, r, s_):
        if s not in self.q_table:
            self.q_table[s] = {act: 0 for act in self.actions}
        if s_ not in self.q_table:
            self.q_table[s_] = {act: 0 for act in self.actions}

        alpha = 0.1
        gamma = 0.9
        max_future_q = max(self.q_table[s_].values())
        self.q_table[s][a] += alpha * (r + gamma * max_future_q - self.q_table[s][a])
