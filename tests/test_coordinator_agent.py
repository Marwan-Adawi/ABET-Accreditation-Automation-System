import pytest
import numpy as np
import torch
from agents.coordinator_agent import CoordinatorAgent, CoordinatorNetwork
from agents.environments.coordinator_env import CoordinatorEnvironment

class TestCoordinatorEnvironment:
    @pytest.fixture
    def env(self):
        """Create environment fixture."""
        return CoordinatorEnvironment(num_tasks=5)
        
    def test_env_initialization(self, env):
        """Test environment initialization."""
        # Test observation space
        assert isinstance(env.observation_space.spaces['task_status'].shape, tuple)
        assert env.observation_space.spaces['task_status'].shape == (5, 4)
        assert env.observation_space.spaces['agent_status'].shape == (3, 3)
        assert env.observation_space.spaces['dependencies'].shape == (5, 5)
        assert env.observation_space.spaces['task_priorities'].shape == (5,)
        assert env.observation_space.spaces['resource_utilization'].shape == (3,)
        
        # Test action space
        assert env.action_space.spaces['task_id'].n == 5
        assert env.action_space.spaces['agent_id'].n == 3
        assert env.action_space.spaces['action_type'].n == 4
        
    def test_env_reset(self, env):
        """Test environment reset."""
        state, _ = env.reset()
        
        # Check state components
        assert all(k in state for k in [
            'task_status', 'agent_status', 'dependencies',
            'task_priorities', 'resource_utilization'
        ])
        
        # Check state shapes
        assert state['task_status'].shape == (5, 4)
        assert state['agent_status'].shape == (3, 3)
        assert state['dependencies'].shape == (5, 5)
        assert state['task_priorities'].shape == (5,)
        assert state['resource_utilization'].shape == (3,)
        
        # Check initial values
        assert np.all(state['task_status'][:, 0] == 1)  # all tasks not started
        assert np.all(state['agent_status'][:, 0] == 1)  # all agents idle
        
    def test_env_step_assign_action(self, env):
        """Test environment step with assign action."""
        state, _ = env.reset()
        
        action = {
            'task_id': 0,
            'agent_id': 0,
            'action_type': 0  # assign
        }
        
        next_state, reward, done, _, _ = env.step(action)
        
        # Check task status update
        assert next_state['task_status'][0][1] == 1  # task in progress
        assert next_state['agent_status'][0][1] == 1  # agent busy
        
        # Check reward is based on priority
        assert reward == pytest.approx(next_state['task_priorities'][0])
        
    def test_env_step_invalid_action(self, env):
        """Test environment step with invalid action."""
        state, _ = env.reset()
        
        # Assign task to busy agent
        action1 = {'task_id': 0, 'agent_id': 0, 'action_type': 0}
        env.step(action1)
        
        # Try to assign another task to same agent
        action2 = {'task_id': 1, 'agent_id': 0, 'action_type': 0}
        _, reward, _, _, _ = env.step(action2)
        
        assert reward == -1.0  # penalty for invalid action
        
    def test_env_episode_completion(self, env):
        """Test episode completion."""
        state, _ = env.reset()
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = {
                'task_id': steps % env.num_tasks,
                'agent_id': steps % env.num_agents,
                'action_type': 0
            }
            _, _, done, _, _ = env.step(action)
            steps += 1
            
        assert steps < max_steps, "Episode should complete before max steps"

class TestCoordinatorNetwork:
    @pytest.fixture
    def network(self):
        """Create network fixture."""
        return CoordinatorNetwork(state_size=200, action_size=60)
        
    def test_network_initialization(self, network):
        """Test network initialization."""
        # Check encoder layers
        assert isinstance(network.task_encoder[0], torch.nn.Linear)
        assert isinstance(network.agent_encoder[0], torch.nn.Linear)
        assert isinstance(network.dependency_encoder[0], torch.nn.Linear)
        
        # Check main network layers
        assert isinstance(network.fc1, torch.nn.Linear)
        assert isinstance(network.fc2, torch.nn.Linear)
        assert isinstance(network.fc3, torch.nn.Linear)
        
    def test_network_forward(self, network):
        """Test network forward pass."""
        # Create dummy input
        state = {
            'task_status': torch.zeros(5, 4),
            'agent_status': torch.zeros(3, 3),
            'dependencies': torch.zeros(5, 5),
            'task_priorities': torch.zeros(5),
            'resource_utilization': torch.zeros(3)
        }
        
        # Forward pass
        output = network(state)
        
        assert output.shape == (60,)  # action space size
        assert not torch.isnan(output).any()

class TestCoordinatorAgent:
    @pytest.fixture
    def env(self):
        return CoordinatorEnvironment(num_tasks=5)
        
    @pytest.fixture
    def agent(self, env):
        return CoordinatorAgent(env)
        
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert isinstance(agent.network, CoordinatorNetwork)
        assert isinstance(agent.target_network, CoordinatorNetwork)
        assert agent.epsilon == 1.0
        assert len(agent.memory) == 0
        
    def test_agent_action_selection(self, agent, env):
        """Test agent action selection."""
        state, _ = env.reset()
        
        # Test epsilon = 1.0 (random actions)
        action = agent.select_action(state)
        assert all(k in action for k in ['task_id', 'agent_id', 'action_type'])
        assert 0 <= action['task_id'] < env.num_tasks
        assert 0 <= action['agent_id'] < env.num_agents
        assert 0 <= action['action_type'] < 4
        
        # Test epsilon = 0.0 (greedy actions)
        agent.epsilon = 0.0
        action = agent.select_action(state)
        assert all(k in action for k in ['task_id', 'agent_id', 'action_type'])
        
    def test_agent_memory(self, agent, env):
        """Test agent memory management."""
        state, _ = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        # Add transition to memory
        agent.memory.append((state, action, reward, next_state, done))
        assert len(agent.memory) == 1
        
    def test_agent_optimization(self, agent, env):
        """Test agent optimization."""
        # Fill memory with some transitions
        for _ in range(agent.batch_size):
            state, _ = env.reset()
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            
        # Perform optimization
        agent._optimize_model()
        
    def test_agent_save_load(self, agent, tmp_path):
        """Test agent model saving and loading."""
        # Save model
        save_path = tmp_path / "model.pt"
        agent.save_model(str(save_path))
        
        # Load model
        agent.load_model(str(save_path))
        
        assert save_path.exists()
        
    @pytest.mark.slow
    def test_agent_training(self, agent, env):
        """Test agent training for a few episodes."""
        rewards = agent.train(num_episodes=5)
        
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)

def test_end_to_end_training():
    """Test end-to-end training process."""
    env = CoordinatorEnvironment(num_tasks=3)
    agent = CoordinatorAgent(env)
    
    # Train for a few episodes
    rewards = agent.train(num_episodes=3)
    
    # Evaluate agent
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
    assert isinstance(total_reward, float)

if __name__ == "__main__":
    pytest.main(["-v", "test_coordinator.py"])