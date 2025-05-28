"""Configuration for ABET automation training."""

# Environment settings
ENV_CONFIG = {
    'num_clos': 10,
    'num_plos': 8,
    'num_courses': 15,
    'max_episode_steps': 100,
    'reward_scale': 1.0
}

# Agent configurations
AGENT_CONFIGS = {
    'doc_gen': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update': 10,
        'hidden_size': 256
    },
    
    'consistency_checker': {
        'learning_rate': 0.1,
        'gamma': 0.9,
        'epsilon': 0.1
    },
    
    'action_planner': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update': 10,
        'hidden_size': 256,
        'planning_steps': 5
    },
    
    'coordinator': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update': 10,
        'hidden_size': 256
    }
}

# Training settings
TRAINING_CONFIG = {
    'num_episodes': 1000,
    'eval_interval': 100,
    'save_interval': 500,
    'num_eval_episodes': 10,
    'save_dir': 'results',
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints'
}

# Reward structure
REWARD_CONFIG = {
    'doc_gen': {
        'completion': 10.0,
        'quality': 5.0,
        'consistency': 3.0,
        'penalty': -1.0
    },
    
    'consistency_checker': {
        'correct_flag': 2.0,
        'missed_issue': -1.0,
        'false_positive': -0.5
    },
    
    'action_planner': {
        'successful_improvement': 5.0,
        'partial_improvement': 2.0,
        'invalid_action': -1.0
    },
    
    'coordinator': {
        'task_completion': 3.0,
        'efficient_sequence': 1.0,
        'deadlock': -2.0
    }
}

# Communication settings
COMMUNICATION_CONFIG = {
    'message_types': [
        'task_complete',
        'issue_found',
        'improvement_needed',
        'action_taken'
    ],
    'max_message_length': 100,
    'message_ttl': 10  # Time-to-live for messages in steps
}

# Blackboard settings
BLACKBOARD_CONFIG = {
    'max_entries': 1000,
    'cleanup_interval': 100,
    'priority_levels': ['high', 'medium', 'low']
} 