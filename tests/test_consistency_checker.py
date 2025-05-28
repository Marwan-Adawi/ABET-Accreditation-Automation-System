import pytest
import numpy as np
from agents.consistency_checker_agent import ConsistencyCheckerAgent
from agents.environments.consistency_checker_env import ConsistencyCheckEnvironment

class TestConsistencyCheckerAgent:
    @pytest.fixture
    def sample_documents(self):
        return {
            'course_reports': [{
                'course_id': 'CS101',
                'clos': [
                    {'id': 'CLO1', 'plos': ['PLO1', 'PLO2'], 'assessment_methods': ['exam', 'project']},
                    {'id': 'CLO2', 'plos': ['PLO2'], 'assessment_methods': ['assignment']}
                ]
            }],
            'clo_assessments': [{
                'clo_id': 'CLO1',
                'scores': [85, 92, 78],
                'assessment_method': 'exam'
            }],
            'syllabus': {
                'clos': [
                    {'id': 'CLO1', 'description': 'Understanding basic concepts'},
                    {'id': 'CLO2', 'description': 'Applying concepts to problems'}
                ]
            }
        }

    @pytest.fixture
    def env(self, sample_documents):
        return ConsistencyCheckEnvironment(sample_documents)

    @pytest.fixture
    def agent(self, env):
        return ConsistencyCheckerAgent(env)

    def test_environment_initialization(self, env):
        assert env.num_clos == 2
        assert env.num_plos == 2
        assert env.num_assessment_methods == 2

    def test_state_space(self, env):
        state, _ = env.reset()
        assert 'clo_plo_mapping' in state
        assert state['clo_plo_mapping'].shape == (2, 2)
        assert 'assessment_methods' in state
        assert 'scores_valid' in state
        assert 'syllabus_match' in state

    def test_action_space(self, env):
        action = {
            'clo_index': 0,
            'issue_type': 0,
            'flag': 1
        }
        state, reward, done, _, _ = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_agent_training(self, agent):
        rewards = agent.train(num_episodes=5, return_rewards=True)
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)

    def test_model_prediction(self, agent, env):
        state, _ = env.reset()
        action = agent.select_action(state)
        assert all(k in action for k in ['clo_index', 'issue_type', 'flag'])