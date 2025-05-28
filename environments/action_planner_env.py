from typing import Dict, List, Tuple, Optional
import numpy as np
from gymnasium import Env, spaces

class ActionPlannerEnvironment(Env):
    """Environment for planning corrective actions in ABET documentation."""
    
    def __init__(self, max_steps: int = 50):
        super().__init__()
        
        # Maximum steps per episode
        self.max_steps = max_steps
        
        # Define observation space components
        self.observation_space = spaces.Dict({
            # Inconsistency features
            'inconsistency_type': spaces.Box(
                low=0, high=1, 
                shape=(4,),  # [CLO-PLO, Assessment, Scores, Syllabus]
                dtype=np.float32
            ),
            
            # Current document state features
            'document_state': spaces.Box(
                low=0, high=1,
                shape=(6,),  # [complete, validated, has_errors, has_feedback, reviewed, approved]
                dtype=np.float32
            ),
            
            # Resource availability
            'resources': spaces.Box(
                low=0, high=1,
                shape=(3,),  # [time, personnel, tools]
                dtype=np.float32
            ),
            
            # Stakeholder features
            'stakeholders': spaces.Box(
                low=0, high=1,
                shape=(4,),  # [faculty, coordinator, evaluator, admin]
                dtype=np.float32
            ),
            
            # Priority level
            'priority': spaces.Box(
                low=0, high=1,
                shape=(3,),  # [low, medium, high]
                dtype=np.float32
            )
        })
        
        # Action space:
        # 1. Primary action type
        # 2. Resource allocation
        # 3. Stakeholder assignment
        # 4. Timeline setting
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(6),  # [review, update, validate, escalate, delegate, approve]
            'resource_allocation': spaces.Box(low=0, high=1, shape=(3,)),
            'stakeholder_assignment': spaces.MultiDiscrete([2, 2, 2, 2]),  # Binary assignment for each stakeholder
            'timeline': spaces.Discrete(4)  # [immediate, short-term, medium-term, long-term]
        })
        
        # Initialize state
        self.state = self._get_initial_state()
        self.current_step = 0
        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.current_step = 0
        return self.state, {}
        
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take action in environment."""
        self.current_step += 1
        
        # Extract action components
        action_type = action['action_type']
        resource_allocation = action['resource_allocation']
        stakeholder_assignment = action['stakeholder_assignment']
        timeline = action['timeline']
        
        # Calculate reward and update state
        reward = self._calculate_reward(action)
        self._update_state(action)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        return self.state, reward, done, False, {}
        
    def _get_initial_state(self) -> Dict:
        """Initialize environment state."""
        return {
            'inconsistency_type': np.random.rand(4),
            'document_state': np.zeros(6),
            'resources': np.random.rand(3),
            'stakeholders': np.zeros(4),
            'priority': self._generate_priority()
        }
        
    def _generate_priority(self) -> np.ndarray:
        """Generate priority levels."""
        priorities = np.zeros(3)
        priority_idx = np.random.randint(0, 3)
        priorities[priority_idx] = 1
        return priorities
        
    def _calculate_reward(self, action: Dict) -> float:
        """Calculate reward based on action and state."""
        reward = 0.0
        
        # Base reward for action appropriateness
        reward += self._get_action_appropriateness_reward(action)
        
        # Resource efficiency reward
        reward += self._get_resource_efficiency_reward(action)
        
        # Stakeholder alignment reward
        reward += self._get_stakeholder_alignment_reward(action)
        
        # Timeline appropriateness reward
        reward += self._get_timeline_reward(action)
        
        # Priority alignment penalty/reward
        reward *= self._get_priority_multiplier()
        
        return reward
        
    def _get_action_appropriateness_reward(self, action: Dict) -> float:
        """Calculate reward for action appropriateness."""
        action_type = action['action_type']
        inconsistency_type = np.argmax(self.state['inconsistency_type'])
        
        # Define action appropriateness matrix
        appropriateness = {
            0: [0, 1, 0.5, 0.2],  # review
            1: [1, 1, 1, 1],      # update
            2: [0.5, 1, 1, 0.5],  # validate
            3: [0.2, 0.5, 1, 1],  # escalate
            4: [0.5, 0.5, 0.5, 1],# delegate
            5: [0, 0.2, 0.5, 0]   # approve
        }
        
        return appropriateness[action_type][inconsistency_type]
        
    def _get_resource_efficiency_reward(self, action: Dict) -> float:
        """Calculate reward for resource efficiency."""
        resource_allocation = action['resource_allocation']
        available_resources = self.state['resources']
        
        # Penalize over-allocation
        if np.any(resource_allocation > available_resources):
            return -0.5
            
        # Reward efficient use
        efficiency = np.mean(resource_allocation / (available_resources + 1e-6))
        return 0.5 * efficiency
        
    def _get_stakeholder_alignment_reward(self, action: Dict) -> float:
        """Calculate reward for stakeholder alignment."""
        stakeholder_assignment = action['stakeholder_assignment']
        required_stakeholders = self._get_required_stakeholders(action['action_type'])
        
        # Check if required stakeholders are assigned
        alignment = np.mean(stakeholder_assignment * required_stakeholders)
        return alignment - 0.5 * np.mean(stakeholder_assignment * (1 - required_stakeholders))
        
    def _get_timeline_reward(self, action: Dict) -> float:
        """Calculate reward for timeline appropriateness."""
        timeline = action['timeline']
        priority_level = np.argmax(self.state['priority'])
        
        # Define timeline appropriateness matrix
        appropriateness = {
            0: [0.5, 1.0, 1.0],    # immediate
            1: [1.0, 0.8, 0.5],    # short-term
            2: [0.8, 0.5, 0.2],    # medium-term
            3: [0.5, 0.2, 0.0]     # long-term
        }
        
        return appropriateness[timeline][priority_level]
        
    def _get_priority_multiplier(self) -> float:
        """Get reward multiplier based on priority."""
        priority_level = np.argmax(self.state['priority'])
        multipliers = [1.0, 1.5, 2.0]
        return multipliers[priority_level]
        
    def _get_required_stakeholders(self, action_type: int) -> np.ndarray:
        """Get required stakeholders for action type."""
        # Stakeholder requirements matrix
        requirements = {
            0: [1, 1, 0, 0],  # review
            1: [1, 0, 0, 0],  # update
            2: [0, 1, 1, 0],  # validate
            3: [0, 1, 0, 1],  # escalate
            4: [0, 1, 1, 0],  # delegate
            5: [0, 1, 0, 1]   # approve
        }
        return np.array(requirements[action_type])
        
    def _update_state(self, action: Dict):
        """Update environment state based on action."""
        # Update document state
        self.state['document_state'] = self._update_document_state(action)
        
        # Update resource availability
        self.state['resources'] -= action['resource_allocation'] * 0.1
        self.state['resources'] = np.clip(self.state['resources'], 0, 1)
        
        # Update stakeholder availability
        self.state['stakeholders'] = self._update_stakeholder_state(action)
        
    def _update_document_state(self, action: Dict) -> np.ndarray:
        """Update document state based on action."""
        doc_state = self.state['document_state'].copy()
        action_type = action['action_type']
        
        # Update state based on action type
        if action_type == 0:  # review
            doc_state[0] += 0.2  # completeness
            doc_state[4] = 1.0  # reviewed
        elif action_type == 1:  # update
            doc_state[0] += 0.3  # completeness
            doc_state[2] = max(0, doc_state[2] - 0.3)  # reduce errors
        elif action_type == 2:  # validate
            doc_state[1] = 1.0  # validated
        elif action_type == 3:  # escalate
            doc_state[3] = 1.0  # has feedback
        elif action_type == 4:  # delegate
            doc_state[0] += 0.1  # completeness
        elif action_type == 5:  # approve
            doc_state[5] = 1.0  # approved
            
        return np.clip(doc_state, 0, 1)
        
    def _update_stakeholder_state(self, action: Dict) -> np.ndarray:
        """Update stakeholder state based on action."""
        stakeholder_state = self.state['stakeholders'].copy()
        assignments = action['stakeholder_assignment']
        
        # Update availability based on assignments
        stakeholder_state += assignments * 0.2
        return np.clip(stakeholder_state, 0, 1)
        
    def _is_episode_done(self) -> bool:
        """Check if episode is done."""
        # Done if maximum steps reached
        if self.current_step >= self.max_steps:
            return True
            
        # Done if document is approved
        if self.state['document_state'][5] == 1.0:
            return True
            
        # Done if all resources depleted
        if np.all(self.state['resources'] <= 0.1):
            return True
            
        return False