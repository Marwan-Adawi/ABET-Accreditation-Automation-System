from typing import Dict, List, Tuple, Optional
import numpy as np
from gymnasium import Env, spaces

class CoordinatorEnvironment(Env):
    """Environment for coordinating multiple ABET automation agents."""
    
    def __init__(self, num_tasks: int = 10):
        super().__init__()
        
        # Number of tasks to coordinate
        self.num_tasks = num_tasks
        
        # Number of agents to coordinate
        self.num_agents = 3  # DocGen, ConsistencyChecker, ActionPlanner
        
        # Define observation space components
        self.observation_space = spaces.Dict({
            # Task status matrix (task_id x status)
            # Status: [not_started, in_progress, completed, has_error]
            'task_status': spaces.Box(
                low=0, high=1, 
                shape=(self.num_tasks, 4),
                dtype=np.float32
            ),
            
            # Agent status matrix (agent_id x status)
            # Status: [idle, busy, error]
            'agent_status': spaces.Box(
                low=0, high=1,
                shape=(self.num_agents, 3),
                dtype=np.float32
            ),
            
            # Dependencies matrix (task_id x task_id)
            # 1 if task_i depends on task_j
            'dependencies': spaces.Box(
                low=0, high=1,
                shape=(self.num_tasks, self.num_tasks),
                dtype=np.float32
            ),
            
            # Priority vector for tasks
            'task_priorities': spaces.Box(
                low=0, high=1,
                shape=(self.num_tasks,),
                dtype=np.float32
            ),
            
            # Resource utilization per agent
            'resource_utilization': spaces.Box(
                low=0, high=1,
                shape=(self.num_agents,),
                dtype=np.float32
            )
        })
        
        # Action space:
        # - Which task to assign (task_id)
        # - Which agent to assign to (agent_id)
        # - Action type (assign, pause, resume, cancel)
        self.action_space = spaces.Dict({
            'task_id': spaces.Discrete(self.num_tasks),
            'agent_id': spaces.Discrete(self.num_agents),
            'action_type': spaces.Discrete(4)  # assign, pause, resume, cancel
        })
        
        # Initialize state
        self.state = self._get_initial_state()
        self.timestep = 0
        self.max_timesteps = 100
        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.timestep = 0
        return self.state, {}
        
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take action in environment."""
        self.timestep += 1
        
        # Extract action components
        task_id = action['task_id']
        agent_id = action['agent_id']
        action_type = action['action_type']
        
        # Apply action and get reward
        reward = self._apply_action(task_id, agent_id, action_type)
        
        # Update environment state
        self._update_state()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        return self.state, reward, done, False, {}
        
    def _get_initial_state(self) -> Dict:
        """Initialize environment state."""
        return {
            'task_status': np.zeros((self.num_tasks, 4), dtype=np.float32),
            'agent_status': np.zeros((self.num_agents, 3), dtype=np.float32),
            'dependencies': np.zeros((self.num_tasks, self.num_tasks), dtype=np.float32),
            'task_priorities': np.random.uniform(0, 1, (self.num_tasks,)),
            'resource_utilization': np.zeros(self.num_agents, dtype=np.float32)
        }
        
    def _apply_action(self, task_id: int, agent_id: int, action_type: int) -> float:
        """Apply action and calculate reward."""
        reward = 0.0
        
        # Check if action is valid
        if not self._is_action_valid(task_id, agent_id, action_type):
            return -1.0  # Penalty for invalid action
        
        # Apply action based on type
        if action_type == 0:  # assign
            reward = self._assign_task(task_id, agent_id)
        elif action_type == 1:  # pause
            reward = self._pause_task(task_id, agent_id)
        elif action_type == 2:  # resume
            reward = self._resume_task(task_id, agent_id)
        else:  # cancel
            reward = self._cancel_task(task_id, agent_id)
            
        return reward
        
    def _is_action_valid(self, task_id: int, agent_id: int, action_type: int) -> bool:
        """Check if action is valid in current state."""
        # Check task_id and agent_id bounds
        if task_id >= self.num_tasks or agent_id >= self.num_agents:
            return False
            
        # Check dependencies
        if action_type == 0:  # assign
            dependencies = self.state['dependencies'][task_id]
            if any(dependencies) and not all(self.state['task_status'][i][2] 
                                           for i in range(self.num_tasks) 
                                           if dependencies[i]):
                return False
                
        # Check agent status
        agent_status = self.state['agent_status'][agent_id]
        if action_type == 0 and agent_status[1]:  # can't assign to busy agent
            return False
            
        return True
        
    def _assign_task(self, task_id: int, agent_id: int) -> float:
        """Assign task to agent."""
        # Update task status to in_progress
        self.state['task_status'][task_id] = [0, 1, 0, 0]
        
        # Update agent status to busy
        self.state['agent_status'][agent_id] = [0, 1, 0]
        
        # Reward based on priority
        reward = self.state['task_priorities'][task_id]
        
        return reward
        
    def _pause_task(self, task_id: int, agent_id: int) -> float:
        """Pause task execution."""
        if self.state['task_status'][task_id][1]:  # if in_progress
            self.state['task_status'][task_id] = [1, 0, 0, 0]
            self.state['agent_status'][agent_id] = [1, 0, 0]
            return 0.0
        return -0.5
        
    def _resume_task(self, task_id: int, agent_id: int) -> float:
        """Resume paused task."""
        if self.state['task_status'][task_id][0]:  # if not_started
            self.state['task_status'][task_id] = [0, 1, 0, 0]
            self.state['agent_status'][agent_id] = [0, 1, 0]
            return 0.0
        return -0.5
        
    def _cancel_task(self, task_id: int, agent_id: int) -> float:
        """Cancel task execution."""
        if self.state['task_status'][task_id][1]:  # if in_progress
            self.state['task_status'][task_id] = [1, 0, 0, 0]
            self.state['agent_status'][agent_id] = [1, 0, 0]
            return -0.2
        return -0.5
        
    def _update_state(self):
        """Update environment state based on time step."""
        # Simulate task progress
        for task_id in range(self.num_tasks):
            if self.state['task_status'][task_id][1]:  # if in_progress
                if np.random.random() < 0.1:  # 10% chance of completion
                    self.state['task_status'][task_id] = [0, 0, 1, 0]
                elif np.random.random() < 0.05:  # 5% chance of error
                    self.state['task_status'][task_id] = [0, 0, 0, 1]
                    
        # Update resource utilization
        self.state['resource_utilization'] = np.array(
            [np.sum(self.state['agent_status'][i][1]) 
             for i in range(self.num_agents)]
        )
        
    def _is_episode_done(self) -> bool:
        """Check if episode is done."""
        # Done if all tasks are completed or have errors
        all_tasks_done = all(
            self.state['task_status'][i][2] or self.state['task_status'][i][3]
            for i in range(self.num_tasks)
        )
        
        # Or if max timesteps reached
        timeout = self.timestep >= self.max_timesteps
        
        return all_tasks_done or timeout