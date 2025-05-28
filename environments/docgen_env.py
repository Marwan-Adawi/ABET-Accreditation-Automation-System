from typing import Dict, List, Tuple, Optional
import numpy as np
from gymnasium import Env, spaces
from datetime import datetime, timezone

class DocGenEnvironment(Env):
    """Environment for generating and managing ABET documentation."""
    
    def __init__(self, max_steps: int = 50):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.current_user = "Marwan-Adawi"
        
        # Define observation space
        self.observation_space = spaces.Dict({
            # Document content state
            'content': spaces.Dict({
                'clos': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),  # Course Learning Outcomes
                'plos': spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),   # Program Learning Outcomes
                'assessments': spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),  # Assessment methods
                'materials': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),    # Course materials
                'prerequisites': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Course prerequisites
            }),
            
            # Document quality metrics
            'quality': spaces.Dict({
                'completeness': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'consistency': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'clarity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'alignment': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            }),
            
            # Document metadata
            'metadata': spaces.Dict({
                'last_modified': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'version': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                'review_status': spaces.Discrete(4),  # [draft, in_review, approved, published]
                'priority': spaces.Discrete(3)        # [low, medium, high]
            }),
            
            # Stakeholder feedback
            'feedback': spaces.Dict({
                'instructor': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                'coordinator': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                'evaluator': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            })
        })
        
        # Define action space
        self.action_space = spaces.Dict({
            'operation': spaces.Discrete(6),  # [add, modify, delete, align, review, publish]
            'section': spaces.Discrete(5),    # [clos, plos, assessments, materials, prerequisites]
            'content': spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            'metadata_update': spaces.Dict({
                'priority': spaces.Discrete(3),
                'review_status': spaces.Discrete(4)
            })
        })
        
        # Initialize state
        self.state = self._get_initial_state()
        
    def _get_initial_state(self) -> Dict:
        """Initialize the environment state."""
        return {
            'content': {
                'clos': np.zeros(10, dtype=np.float32),
                'plos': np.zeros(8, dtype=np.float32),
                'assessments': np.zeros(6, dtype=np.float32),
                'materials': np.zeros(5, dtype=np.float32),
                'prerequisites': np.zeros(3, dtype=np.float32)
            },
            'quality': {
                'completeness': np.array([0.0], dtype=np.float32),
                'consistency': np.array([0.0], dtype=np.float32),
                'clarity': np.array([0.0], dtype=np.float32),
                'alignment': np.array([0.0], dtype=np.float32)
            },
            'metadata': {
                'last_modified': np.array([0.0], dtype=np.float32),
                'version': np.array([1.0], dtype=np.float32),
                'review_status': 0,  # draft
                'priority': 1        # medium
            },
            'feedback': {
                'instructor': np.zeros(3, dtype=np.float32),
                'coordinator': np.zeros(3, dtype=np.float32),
                'evaluator': np.zeros(3, dtype=np.float32)
            }
        }
        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._get_initial_state()
        return self.state, {}
        
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.current_step += 1
        
        # Apply action to state
        self._apply_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Update quality metrics
        self._update_quality_metrics()
        
        # Update metadata
        self._update_metadata()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        return self.state, reward, done, False, {}
        
    def _apply_action(self, action: Dict):
        """Apply action to state."""
        operation = action['operation']
        section = action['section']
        content = action['content']
        metadata_update = action['metadata_update']
        
        # Get section name
        section_names = ['clos', 'plos', 'assessments', 'materials', 'prerequisites']
        section_name = section_names[section]
        
        # Apply operation
        if operation == 0:  # add
            self.state['content'][section_name] = np.clip(
                self.state['content'][section_name] + content[:len(self.state['content'][section_name])],
                0, 1
            )
        elif operation == 1:  # modify
            self.state['content'][section_name] = np.clip(
                content[:len(self.state['content'][section_name])],
                0, 1
            )
        elif operation == 2:  # delete
            self.state['content'][section_name] *= (1 - np.abs(content[:len(self.state['content'][section_name])]))
        elif operation == 3:  # align
            self._align_content(section_name)
        elif operation == 4:  # review
            self._update_review_status()
        elif operation == 5:  # publish
            if self._can_publish():
                self.state['metadata']['review_status'] = 3
        
        # Update metadata
        self.state['metadata']['priority'] = metadata_update['priority']
        if metadata_update['review_status'] < self.state['metadata']['review_status']:
            self.state['metadata']['review_status'] = metadata_update['review_status']
            
    def _align_content(self, section_name: str):
        """Align content with other sections."""
        if section_name == 'clos':
            # Align CLOs with PLOs
            alignment = np.mean([self.state['content']['clos'], self.state['content']['plos']], axis=0)
            self.state['quality']['alignment'] = np.array([np.mean(alignment)])
        elif section_name == 'assessments':
            # Align assessments with CLOs
            alignment = np.mean([self.state['content']['assessments'], self.state['content']['clos']], axis=0)
            self.state['quality']['alignment'] = np.array([np.mean(alignment)])
            
    def _update_review_status(self):
        """Update document review status."""
        current_status = self.state['metadata']['review_status']
        quality_threshold = 0.7
        
        if (current_status == 0 and  # draft
            np.mean([v for v in self.state['quality'].values()]) > quality_threshold):
            self.state['metadata']['review_status'] = 1  # in_review
        elif (current_status == 1 and  # in_review
              all(np.mean(f) > 0 for f in self.state['feedback'].values())):
            self.state['metadata']['review_status'] = 2  # approved
            
    def _can_publish(self) -> bool:
        """Check if document can be published."""
        return (self.state['metadata']['review_status'] == 2 and  # approved
                all(v > 0.8 for v in self.state['quality'].values()))
                
    def _calculate_reward(self, action: Dict) -> float:
        """Calculate reward for action."""
        reward = 0.0
        
        # Quality improvement reward
        quality_improvement = sum(self.state['quality'].values()) / len(self.state['quality'])
        reward += quality_improvement * 2.0
        
        # Content completion reward
        content_completion = np.mean([np.mean(v) for v in self.state['content'].values()])
        reward += content_completion
        
        # Alignment reward
        reward += float(self.state['quality']['alignment']) * 1.5
        
        # Status progression reward
        if self.state['metadata']['review_status'] > 0:
            reward += self.state['metadata']['review_status']
            
        # Feedback incorporation penalty/reward
        if any(np.mean(f) < 0 for f in self.state['feedback'].values()):
            reward -= 1.0
        elif all(np.mean(f) > 0 for f in self.state['feedback'].values()):
            reward += 2.0
            
        return reward
        
    def _update_quality_metrics(self):
        """Update document quality metrics."""
        # Update completeness
        self.state['quality']['completeness'] = np.array([
            np.mean([np.mean(v) for v in self.state['content'].values()])
        ])
        
        # Update consistency
        section_means = [np.mean(v) for v in self.state['content'].values()]
        self.state['quality']['consistency'] = np.array([
            1 - np.std(section_means) if section_means else 0.0
        ])
        
        # Update clarity (based on content structure)
        self.state['quality']['clarity'] = np.array([
            np.mean([1 - np.std(v) for v in self.state['content'].values()])
        ])
        
    def _update_metadata(self):
        """Update document metadata."""
        # Update last modified timestamp
        current_time = datetime.now(timezone.utc)
        target_time = datetime.strptime("2025-05-28 19:56:42", "%Y-%m-%d %H:%M:%S")
        time_diff = (target_time - current_time).total_seconds()
        self.state['metadata']['last_modified'] = np.array([
            np.clip(time_diff / (24 * 60 * 60), 0, 1)  # Normalize to [0, 1]
        ])
        
        # Update version
        self.state['metadata']['version'] += np.array([0.1])
        
    def _is_episode_done(self) -> bool:
        """Check if episode is done."""
        # Done if maximum steps reached
        if self.current_step >= self.max_steps:
            return True
            
        # Done if document is published
        if self.state['metadata']['review_status'] == 3:
            return True
            
        # Done if quality metrics are all above threshold
        if all(v > 0.9 for v in self.state['quality'].values()):
            return True
            
        return False