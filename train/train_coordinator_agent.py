import torch
import numpy as np
from agents.coordinator_agent import CoordinatorAgent
from agents.environments.coordinator_env import CoordinatorEnvironment
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import logging
from typing import List, Dict, Tuple
import pandas as pd

class CoordinatorTrainer:
    def __init__(self, 
                 num_tasks: int = 10,
                 save_dir: str = "training_results",
                 log_dir: str = "logs",
                 checkpoint_dir: str = "checkpoints"):
        """Initialize the trainer with directories for saving results."""
        self.num_tasks = num_tasks
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Create environment and agent
        self.env = CoordinatorEnvironment(num_tasks=num_tasks)
        self.agent = CoordinatorAgent(self.env)
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.completion_rates = []
        self.task_distributions = []
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.log_dir, 
                               f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def train(self, 
              num_episodes: int = 1000,
              eval_interval: int = 50,
              save_interval: int = 100) -> Dict:
        """Train the coordinator agent."""
        best_avg_reward = float('-inf')
        training_start = datetime.now()
        
        for episode in range(num_episodes):
            # Run one episode
            episode_metrics = self._run_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_metrics['reward'])
            self.completion_rates.append(episode_metrics['completion_rate'])
            self.task_distributions.append(episode_metrics['task_distribution'])
            
            # Calculate average reward
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_reward)
            else:
                avg_reward = episode_metrics['reward']
                
            # Log progress
            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1, num_episodes, avg_reward)
                
            # Evaluate and save best model
            if (episode + 1) % eval_interval == 0:
                eval_metrics = self._evaluate()
                if eval_metrics['avg_reward'] > best_avg_reward:
                    best_avg_reward = eval_metrics['avg_reward']
                    self._save_model(f"best_model_{best_avg_reward:.2f}.pt")
                    
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)
                
            # Create visualizations
            if (episode + 1) % 50 == 0:
                self._create_visualizations(episode + 1)
                
        # Save final results
        training_time = datetime.now() - training_start
        final_metrics = {
            'num_episodes': num_episodes,
            'best_avg_reward': best_avg_reward,
            'training_time': str(training_time),
            'final_avg_reward': np.mean(self.episode_rewards[-10:]),
            'final_completion_rate': np.mean(self.completion_rates[-10:])
        }
        
        self._save_results(final_metrics)
        return final_metrics
    
    def _run_episode(self) -> Dict:
        """Run one training episode."""
        state, _ = self.env.reset()
        episode_reward = 0
        task_status_history = []
        
        while True:
            # Select and take action
            action = self.agent.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            
            # Store transition
            self.agent.memory.append((state, action, reward, next_state, done))
            
            # Track task status
            task_status_history.append(next_state['task_status'].copy())
            
            # Optimize model
            self.agent._optimize_model()
            
            if done:
                break
                
            state = next_state
            
        # Calculate episode metrics
        completion_rate = self._calculate_completion_rate(next_state)
        task_distribution = self._calculate_task_distribution(task_status_history)
        
        return {
            'reward': episode_reward,
            'completion_rate': completion_rate,
            'task_distribution': task_distribution
        }
    
    def _evaluate(self, num_eval_episodes: int = 5) -> Dict:
        """Evaluate the agent's performance."""
        eval_rewards = []
        eval_completion_rates = []
        
        # Store current epsilon and set to 0 for evaluation
        current_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        for _ in range(num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
                    
                state = next_state
                
            eval_rewards.append(episode_reward)
            eval_completion_rates.append(self._calculate_completion_rate(next_state))
            
        # Restore epsilon
        self.agent.epsilon = current_epsilon
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_completion_rate': np.mean(eval_completion_rates)
        }
    
    def _calculate_completion_rate(self, state: Dict) -> float:
        """Calculate task completion rate."""
        completed_tasks = np.sum(state['task_status'][:, 2])  # index 2 is completed status
        return completed_tasks / self.num_tasks
    
    def _calculate_task_distribution(self, task_history: List[np.ndarray]) -> Dict:
        """Calculate distribution of task states."""
        final_status = task_history[-1]
        return {
            'not_started': np.sum(final_status[:, 0]),
            'in_progress': np.sum(final_status[:, 1]),
            'completed': np.sum(final_status[:, 2]),
            'error': np.sum(final_status[:, 3])
        }
    
    def _create_visualizations(self, episode: int):
        """Create and save visualization plots."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        plt.plot(self.avg_rewards, label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f'rewards_{timestamp}.png'))
        plt.close()
        
        # Plot completion rates
        plt.figure(figsize=(12, 6))
        plt.plot(self.completion_rates)
        plt.xlabel('Episode')
        plt.ylabel('Completion Rate')
        plt.title('Task Completion Rates')
        plt.savefig(os.path.join(self.save_dir, f'completion_rates_{timestamp}.png'))
        plt.close()
        
        # Plot task distribution
        task_dist_df = pd.DataFrame(self.task_distributions)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=task_dist_df)
        plt.title('Task State Distribution')
        plt.ylabel('Number of Tasks')
        plt.savefig(os.path.join(self.save_dir, f'task_distribution_{timestamp}.png'))
        plt.close()
    
    def _log_progress(self, episode: int, total_episodes: int, avg_reward: float):
        """Log training progress."""
        logging.info(
            f"Episode {episode}/{total_episodes} - "
            f"Average Reward: {avg_reward:.2f}, "
            f"Completion Rate: {self.completion_rates[-1]:.2f}, "
            f"Epsilon: {self.agent.epsilon:.3f}"
        )
    
    def _save_model(self, filename: str):
        """Save the model."""
        path = os.path.join(self.checkpoint_dir, filename)
        self.agent.save_model(path)
        logging.info(f"Saved model to {path}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'agent_state': self.agent.network.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
            'episode_rewards': self.episode_rewards,
            'completion_rates': self.completion_rates
        }
        
        path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_episode_{episode}.pt"
        )
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")
    
    def _save_results(self, metrics: Dict):
        """Save training results."""
        results = {
            'metrics': metrics,
            'episode_rewards': self.episode_rewards,
            'avg_rewards': self.avg_rewards,
            'completion_rates': self.completion_rates,
            'final_task_distribution': self.task_distributions[-1]
        }
        
        path = os.path.join(
            self.save_dir, 
            f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logging.info(f"Saved training results to {path}")

def main():
    """Main training function."""
    # Training configuration
    config = {
        'num_tasks': 10,
        'num_episodes': 1000,
        'eval_interval': 50,
        'save_interval': 100
    }
    
    # Create trainer
    trainer = CoordinatorTrainer(
        num_tasks=config['num_tasks'],
        save_dir='training_results',
        log_dir='logs',
        checkpoint_dir='checkpoints'
    )
    
    # Start training
    print("Starting training...")
    results = trainer.train(
        num_episodes=config['num_episodes'],
        eval_interval=config['eval_interval'],
        save_interval=config['save_interval']
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best average reward: {results['best_avg_reward']:.2f}")
    print(f"Final average reward: {results['final_avg_reward']:.2f}")
    print(f"Final completion rate: {results['final_completion_rate']:.2f}")
    print(f"Training time: {results['training_time']}")

if __name__ == "__main__":
    main()