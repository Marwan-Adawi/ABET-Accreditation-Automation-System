from typing import Dict, List, Tuple, Optional
import numpy as np
from gymnasium import Env, spaces

class ConsistencyCheckEnvironment(Env):
    """Environment for consistency checking in ABET documentation."""
    
    def __init__(self, documents: Dict):
        super().__init__()
        
        # Store documents to check
        self.documents = documents
        
        # Define observation space components
        self.num_clos = len(self.get_all_clos())
        self.num_plos = len(self.get_all_plos())
        self.num_assessment_methods = len(self.get_assessment_methods())
        
        # State space: Binary vectors for CLO-PLO mappings, assessment methods, and scores
        self.observation_space = spaces.Dict({
            'clo_plo_mapping': spaces.Box(low=0, high=1, shape=(self.num_clos, self.num_plos)),
            'assessment_methods': spaces.Box(low=0, high=1, shape=(self.num_clos, self.num_assessment_methods)),
            'scores_valid': spaces.Box(low=0, high=1, shape=(self.num_clos,)),
            'syllabus_match': spaces.Box(low=0, high=1, shape=(self.num_clos,))
        })
        
        # Action space: For each CLO, can flag issues with:
        # - CLO-PLO mapping
        # - Assessment method
        # - Score validity
        # - Syllabus consistency
        self.action_space = spaces.Dict({
            'clo_index': spaces.Discrete(self.num_clos),
            'issue_type': spaces.Discrete(4),  # 4 types of issues
            'flag': spaces.Discrete(2)  # 0: no issue, 1: flag issue
        })
        
        # Initialize state
        self.state = self._get_initial_state()
        self.inconsistencies = []
        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.inconsistencies = []
        return self.state, {}
        
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take action in environment."""
        clo_idx = action['clo_index']
        issue_type = action['issue_type']
        flag = action['flag']
        
        # Get current state
        prev_state = self.state.copy()
        
        # Apply action
        reward = self._apply_action(clo_idx, issue_type, flag)
        
        # Check if episode is done (all consistency checks completed)
        done = len(self.inconsistencies) == self._count_actual_inconsistencies()
        
        return self.state, reward, done, False, {}
        
    def _get_initial_state(self) -> Dict:
        """Initialize state from documents."""
        clo_plo_mapping = self._extract_clo_plo_mapping()
        assessment_methods = self._extract_assessment_methods()
        scores_valid = self._check_score_validity()
        syllabus_match = self._check_syllabus_consistency()
        
        return {
            'clo_plo_mapping': clo_plo_mapping,
            'assessment_methods': assessment_methods,
            'scores_valid': scores_valid,
            'syllabus_match': syllabus_match
        }
        
    def _apply_action(self, clo_idx: int, issue_type: int, flag: int) -> float:
        """Apply action and return reward."""
        # Get actual inconsistency status
        actual_issue = self._check_actual_inconsistency(clo_idx, issue_type)
        
        # Calculate reward
        if actual_issue and flag == 1:
            # Correctly identified inconsistency
            reward = 1.0
            self.inconsistencies.append((clo_idx, issue_type))
        elif not actual_issue and flag == 0:
            # Correctly identified consistency
            reward = 0.5
        else:
            # Incorrect identification
            reward = -1.0
            
        return reward
        
    def _check_actual_inconsistency(self, clo_idx: int, issue_type: int) -> bool:
        """Check if there is actually an inconsistency."""
        if issue_type == 0:  # CLO-PLO mapping
            return not any(self.state['clo_plo_mapping'][clo_idx])
        elif issue_type == 1:  # Assessment method
            return not any(self.state['assessment_methods'][clo_idx])
        elif issue_type == 2:  # Score validity
            return not self.state['scores_valid'][clo_idx]
        else:  # Syllabus consistency
            return not self.state['syllabus_match'][clo_idx]
            
    def _count_actual_inconsistencies(self) -> int:
        """Count total number of actual inconsistencies."""
        count = 0
        for clo_idx in range(self.num_clos):
            for issue_type in range(4):
                if self._check_actual_inconsistency(clo_idx, issue_type):
                    count += 1
        return count