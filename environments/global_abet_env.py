from typing import Dict, List, Tuple, Optional
import numpy as np
from gymnasium import Env, spaces
from datetime import datetime, timezone

class Blackboard:
    """Shared memory for inter-agent communication."""
    def __init__(self):
        self.messages = []
        self.shared_state = {
            'documents': {},
            'inconsistencies': [],
            'improvements': [],
            'task_status': {},
            'last_agent': None,
            'last_action': None,
            'last_reward': None
        }
        
    def post_message(self, agent_id: str, message: Dict):
        """Post a message to the blackboard."""
        self.messages.append({
            'timestamp': datetime.now(timezone.utc),
            'agent_id': agent_id,
            'message': message
        })
        
    def get_messages(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get messages, optionally filtered by agent."""
        if agent_id:
            return [m for m in self.messages if m['agent_id'] == agent_id]
        return self.messages
        
    def update_state(self, key: str, value: any):
        """Update shared state."""
        self.shared_state[key] = value
        
    def get_state(self, key: str) -> any:
        """Get shared state value."""
        return self.shared_state.get(key)

class GlobalABETEnv(Env):
    """Unified environment for ABET automation."""
    
    def __init__(self):
        super().__init__()
        self.blackboard = Blackboard()
        
        # Define observation spaces for each agent
        self.observation_spaces = {
            'docgen': spaces.Dict({
                'content': spaces.Dict({
                    'clos': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
                    'plos': spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
                    'assessments': spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
                    'materials': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
                    'prerequisites': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
                }),
                'quality': spaces.Dict({
                    'completeness': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    'consistency': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    'clarity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    'alignment': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                })
            }),
            
            'consistency_checker': spaces.Dict({
                'clo_plo_mapping': spaces.Box(low=0, high=1, shape=(10, 8), dtype=np.float32),
                'assessment_methods': spaces.Box(low=0, high=1, shape=(10, 6), dtype=np.float32),
                'scores_valid': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
                'syllabus_match': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
            }),
            
            'action_planner': spaces.Dict({
                'inconsistency_type': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                'document_state': spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
                'resources': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                'stakeholders': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                'priority': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
            }),
            
            'coordinator': spaces.Dict({
                'task_status': spaces.Box(low=0, high=1, shape=(10, 4), dtype=np.float32),
                'agent_status': spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.float32),
                'dependencies': spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.float32),
                'task_priorities': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
                'resource_utilization': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
            })
        }
        
        # Define action spaces for each agent
        self.action_spaces = {
            'docgen': spaces.Dict({
                'operation': spaces.Discrete(6),
                'section': spaces.Discrete(5),
                'content': spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
                'metadata_update': spaces.Dict({
                    'priority': spaces.Discrete(3),
                    'review_status': spaces.Discrete(4)
                })
            }),
            
            'consistency_checker': spaces.Dict({
                'clo_index': spaces.Discrete(10),
                'issue_type': spaces.Discrete(4),
                'flag': spaces.Discrete(2)
            }),
            
            'action_planner': spaces.Dict({
                'action_type': spaces.Discrete(6),
                'resource_allocation': spaces.Box(low=0, high=1, shape=(3,)),
                'stakeholder_assignment': spaces.MultiDiscrete([2, 2, 2, 2]),
                'timeline': spaces.Discrete(4)
            }),
            
            'coordinator': spaces.Dict({
                'task_id': spaces.Discrete(10),
                'agent_id': spaces.Discrete(3),
                'action_type': spaces.Discrete(4)
            })
        }
        
        # Initialize state
        self.state = self._get_initial_state()
        self.current_step = 0
        self.max_steps = 100
        
    def _get_initial_state(self) -> Dict:
        """Initialize the global state."""
        return {
            'docgen': {
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
                }
            },
            'consistency_checker': {
                'clo_plo_mapping': np.zeros((10, 8), dtype=np.float32),
                'assessment_methods': np.zeros((10, 6), dtype=np.float32),
                'scores_valid': np.zeros(10, dtype=np.float32),
                'syllabus_match': np.zeros(10, dtype=np.float32)
            },
            'action_planner': {
                'inconsistency_type': np.zeros(4, dtype=np.float32),
                'document_state': np.zeros(6, dtype=np.float32),
                'resources': np.ones(3, dtype=np.float32),
                'stakeholders': np.zeros(4, dtype=np.float32),
                'priority': np.array([0, 1, 0], dtype=np.float32)  # medium priority
            },
            'coordinator': {
                'task_status': np.zeros((10, 4), dtype=np.float32),
                'agent_status': np.zeros((3, 3), dtype=np.float32),
                'dependencies': np.zeros((10, 10), dtype=np.float32),
                'task_priorities': np.random.uniform(0, 1, (10,)),
                'resource_utilization': np.zeros(3, dtype=np.float32)
            }
        }
        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.current_step = 0
        self.blackboard = Blackboard()
        return self._get_agent_observations(), {}
        
    def step(self, agent_id: str, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step in the environment for a specific agent."""
        self.current_step += 1
        
        # Update state based on agent and action
        reward = self._apply_action(agent_id, action)
        
        # Update blackboard
        self.blackboard.post_message(agent_id, {
            'action': action,
            'reward': reward,
            'step': self.current_step
        })
        
        # Check if episode is done
        done = self._is_episode_done()
        
        return self._get_agent_observations(), reward, done, False, {}
        
    def _get_agent_observations(self) -> Dict:
        """Get observations for each agent."""
        return {
            agent_id: self._filter_state_for_agent(agent_id)
            for agent_id in self.observation_spaces.keys()
        }
        
    def _filter_state_for_agent(self, agent_id: str) -> Dict:
        """Filter global state to get agent-specific observation."""
        return self.state[agent_id]
        
    def _apply_action(self, agent_id: str, action: Dict) -> float:
        """Apply action and calculate reward."""
        if agent_id == 'docgen':
            return self._apply_docgen_action(action)
        elif agent_id == 'consistency_checker':
            return self._apply_consistency_checker_action(action)
        elif agent_id == 'action_planner':
            return self._apply_action_planner_action(action)
        elif agent_id == 'coordinator':
            return self._apply_coordinator_action(action)
        else:
            raise ValueError(f"Unknown agent: {agent_id}")
            
    def _apply_docgen_action(self, action: Dict) -> float:
        """Apply DocGen agent action."""
        # Implementation from docgen_env.py
        reward = 0.0
        
        # Quality improvement reward
        quality_improvement = sum(self.state['docgen']['quality'].values()) / len(self.state['docgen']['quality'])
        reward += quality_improvement * 2.0
        
        # Content completion reward
        content_completion = np.mean([np.mean(v) for v in self.state['docgen']['content'].values()])
        reward += content_completion
        
        return reward
        
    def _apply_consistency_checker_action(self, action: Dict) -> float:
        """Apply ConsistencyChecker agent action."""
        # Implementation from consistency_checker_env.py
        clo_idx = action['clo_index']
        issue_type = action['issue_type']
        flag = action['flag']
        
        # Calculate reward based on actual inconsistencies
        actual_issue = self._check_actual_inconsistency(clo_idx, issue_type)
        
        if actual_issue and flag == 1:
            reward = 1.0
        elif not actual_issue and flag == 0:
            reward = 0.5
        else:
            reward = -1.0
            
        return reward
        
    def _apply_action_planner_action(self, action: Dict) -> float:
        """Apply ActionPlanner agent action."""
        # Implementation from action_planner_env.py
        reward = 0.0
        
        # Base reward for action appropriateness
        reward += self._get_action_appropriateness_reward(action)
        
        # Resource efficiency reward
        reward += self._get_resource_efficiency_reward(action)
        
        # Stakeholder alignment reward
        reward += self._get_stakeholder_alignment_reward(action)
        
        return reward
        
    def _apply_coordinator_action(self, action: Dict) -> float:
        """Apply Coordinator agent action."""
        # Implementation from coordinator_env.py
        task_id = action['task_id']
        agent_id = action['agent_id']
        action_type = action['action_type']
        
        if not self._is_action_valid(task_id, agent_id, action_type):
            return -1.0
            
        if action_type == 0:  # assign
            reward = self.state['coordinator']['task_priorities'][task_id]
        else:
            reward = 0.0
            
        return reward
        
    def _check_actual_inconsistency(self, clo_idx: int, issue_type: int) -> bool:
        """Check if there is actually an inconsistency."""
        if issue_type == 0:  # CLO-PLO mapping
            return not any(self.state['consistency_checker']['clo_plo_mapping'][clo_idx])
        elif issue_type == 1:  # Assessment method
            return not any(self.state['consistency_checker']['assessment_methods'][clo_idx])
        elif issue_type == 2:  # Score validity
            return not self.state['consistency_checker']['scores_valid'][clo_idx]
        else:  # Syllabus consistency
            return not self.state['consistency_checker']['syllabus_match'][clo_idx]
            
    def _is_action_valid(self, task_id: int, agent_id: int, action_type: int) -> bool:
        """Check if coordinator action is valid."""
        if task_id >= 10 or agent_id >= 3:
            return False
            
        if action_type == 0:  # assign
            dependencies = self.state['coordinator']['dependencies'][task_id]
            if any(dependencies) and not all(self.state['coordinator']['task_status'][i][2] 
                                           for i in range(10) 
                                           if dependencies[i]):
                return False
                
        agent_status = self.state['coordinator']['agent_status'][agent_id]
        if action_type == 0 and agent_status[1]:  # can't assign to busy agent
            return False
            
        return True
        
    def _get_action_appropriateness_reward(self, action: Dict) -> float:
        """Calculate reward for action appropriateness."""
        action_type = action['action_type']
        inconsistency_type = np.argmax(self.state['action_planner']['inconsistency_type'])
        
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
        available_resources = self.state['action_planner']['resources']
        
        if np.any(resource_allocation > available_resources):
            return -0.5
            
        efficiency = np.mean(resource_allocation / (available_resources + 1e-6))
        return 0.5 * efficiency
        
    def _get_stakeholder_alignment_reward(self, action: Dict) -> float:
        """Calculate reward for stakeholder alignment."""
        stakeholder_assignment = action['stakeholder_assignment']
        required_stakeholders = self._get_required_stakeholders(action['action_type'])
        
        alignment = np.mean(stakeholder_assignment * required_stakeholders)
        return alignment - 0.5 * np.mean(stakeholder_assignment * (1 - required_stakeholders))
        
    def _get_required_stakeholders(self, action_type: int) -> np.ndarray:
        """Get required stakeholders for action type."""
        requirements = {
            0: [1, 1, 0, 0],  # review
            1: [1, 0, 0, 0],  # update
            2: [0, 1, 1, 0],  # validate
            3: [0, 1, 0, 1],  # escalate
            4: [0, 1, 1, 0],  # delegate
            5: [0, 1, 0, 1]   # approve
        }
        return np.array(requirements[action_type])
        
    def _is_episode_done(self) -> bool:
        """Check if episode is done."""
        if self.current_step >= self.max_steps:
            return True
            
        # Check if all tasks are completed
        all_tasks_done = all(
            self.state['coordinator']['task_status'][i][2] or self.state['coordinator']['task_status'][i][3]
            for i in range(10)
        )
        
        return all_tasks_done 