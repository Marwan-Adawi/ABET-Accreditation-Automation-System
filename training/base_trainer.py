from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from datetime import datetime

class BaseTrainer:
    """Base class for training RL agents."""
    
    def __init__(self,
                 agent: Any,
                 env: Any,
                 save_dir: str = "results",
                 log_dir: str = "logs",
                 checkpoint_dir: str = "checkpoints",
                 eval_interval: int = 100,
                 save_interval: int = 500,
                 num_eval_episodes: int = 10):
        
        self.agent = agent
        self.env = env
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.save_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.num_eval_episodes = num_eval_episodes
        
        # Setup logging
        self.setup_logging()
        
        # Training metrics
        self.episode_rewards = []
        self.eval_rewards = []
        self.episode_lengths = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def train_episode(self) -> Tuple[float, int]:
        """Train for one episode. Returns (reward, length)."""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            # Update agent
            self.agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
        return episode_reward, episode_length
        
    def evaluate(self) -> float:
        """Evaluate agent performance."""
        eval_rewards = []
        
        for _ in range(self.num_eval_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                
            eval_rewards.append(episode_reward)
            
        return np.mean(eval_rewards)
        
    def save_checkpoint(self, episode: int):
        """Save agent checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pkl"
        self.agent.save_model(str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def save_metrics(self):
        """Save training metrics."""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'eval_rewards': self.eval_rewards,
            'episode_lengths': self.episode_lengths
        }
        
        metrics_path = self.save_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
        self.logger.info(f"Saved metrics to {metrics_path}")
        
    def plot_metrics(self):
        """Plot training metrics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot rewards
        ax1.plot(self.episode_rewards, label='Training')
        ax1.plot(np.arange(0, len(self.episode_rewards), self.eval_interval),
                self.eval_rewards, label='Evaluation')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.set_title('Training and Evaluation Rewards')
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.set_title('Episode Lengths')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png')
        plt.close()
        
    def train(self, num_episodes: int):
        """Main training loop."""
        self.logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Train episode
            reward, length = self.train_episode()
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.logger.info(f"Episode {episode + 1}/{num_episodes}, "
                               f"Avg Reward: {avg_reward:.2f}, "
                               f"Length: {length}")
                
            # Evaluate
            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                self.logger.info(f"Evaluation at episode {episode + 1}: "
                               f"Average reward = {eval_reward:.2f}")
                
            # Save checkpoint
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(episode + 1)
                
        # Save final metrics and plots
        self.save_metrics()
        self.plot_metrics()
        
        self.logger.info("Training completed")
        return self.episode_rewards, self.eval_rewards 