import os
from pathlib import Path
from agents.consistency_checker_agent import ConsistencyCheckerAgent
from environments.global_abet_env import GlobalABETEnv
from training.base_trainer import BaseTrainer
from config.training_config import (
    AGENT_CONFIGS,
    TRAINING_CONFIG,
    ENV_CONFIG
)

def main():
    # Create directories
    for dir_name in ['results', 'logs', 'checkpoints']:
        Path(dir_name).mkdir(exist_ok=True)
        
    # Initialize environment
    env = GlobalABETEnv(
        num_clos=ENV_CONFIG['num_clos'],
        num_plos=ENV_CONFIG['num_plos'],
        num_courses=ENV_CONFIG['num_courses'],
        max_episode_steps=ENV_CONFIG['max_episode_steps']
    )
    
    # Initialize agent
    agent = ConsistencyCheckerAgent(
        learning_rate=AGENT_CONFIGS['consistency_checker']['learning_rate'],
        gamma=AGENT_CONFIGS['consistency_checker']['gamma'],
        epsilon=AGENT_CONFIGS['consistency_checker']['epsilon']
    )
    
    # Initialize trainer
    trainer = BaseTrainer(
        agent=agent,
        env=env,
        save_dir=TRAINING_CONFIG['save_dir'],
        log_dir=TRAINING_CONFIG['log_dir'],
        checkpoint_dir=TRAINING_CONFIG['checkpoint_dir'],
        eval_interval=TRAINING_CONFIG['eval_interval'],
        save_interval=TRAINING_CONFIG['save_interval'],
        num_eval_episodes=TRAINING_CONFIG['num_eval_episodes']
    )
    
    # Start training
    print("Starting ConsistencyChecker training...")
    trainer.train(num_episodes=TRAINING_CONFIG['num_episodes'])
    print("Training completed!")

if __name__ == "__main__":
    main() 