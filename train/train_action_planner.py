import torch
import numpy as np
from agents.action_planner_agent import ActionPlannerAgent
from agents.environments.action_planner_env import ActionPlannerEnvironment
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import logging
from typing import List, Dict, Tuple
import pandas as pd

class ActionPlannerTrainer:
    def __init__(self, 
                 save_dir: str = "training_results/action_planner",
                 log_dir: str = "logs/action_planner",
                 checkpoint_dir: str = "checkpoints/action_planner",
                 max_steps: int = 50):
        """Initialize the trainer with directories for saving results."""
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
        self.env = ActionPlannerEnvironment(max_steps=max_steps)
        self.agent = ActionPlannerAgent(self.env)
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.action_distributions = []
        self.resource_usage = []
        self.document_states = []
        self.completion_times = []
        
    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f"training_{timestamp}.log")
        
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
              save_interval: int = 100,
              visualize_interval: int = 50) -> Dict:
        """Train the action planner agent."""
        best_avg_reward = float('-inf')
        training_start = datetime.now()
        
        for episode in range(num_episodes):
            # Run one episode
            episode_metrics = self._run_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_metrics['reward'])
            self.action_distributions.append(episode_metrics['action_distribution'])
            self.resource_usage.append(episode_metrics['resource_usage'])
            self.document_states.append(episode_metrics['document_state'])
            self.completion_times.append(episode_metrics['completion_time'])
            
            # Calculate average reward
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_reward)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1, num_episodes)
            
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
            if (episode + 1) % visualize_interval == 0:
                self._create_visualizations(episode + 1)
        
        # Save final results
        training_time = datetime.now() - training_start
        final_metrics = {
            'num_episodes': num_episodes,
            'best_avg_reward': best_avg_reward,
            'training_time': str(training_time),
            'final_avg_reward': np.mean(self.episode_rewards[-10:]),
            'final_completion_time': np.mean(self.completion_times[-10:])
        }
        
        self._save_results(final_metrics)
        return final_metrics
    
    def _run_episode(self) -> Dict:
        """Run one training episode."""
        state, _ = self.env.reset()
        episode_reward = 0
        action_history = []
        resource_history = []
        doc_state_history = []
        steps = 0
        
        while True:
            # Select and take action
            action = self.agent.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            
            # Track metrics
            action_history.append(action['action_type'])
            resource_history.append(state['resources'].copy())
            doc_state_history.append(state['document_state'].copy())
            
            # Store transition
            self.agent.memory.append((state, action, reward, next_state, done))
            
            # Optimize model
            self.agent._optimize_model()
            
            if done:
                break
                
            state = next_state
            steps += 1
        
        return {
            'reward': episode_reward,
            'action_distribution': self._calculate_action_distribution(action_history),
            'resource_usage': np.mean(resource_history, axis=0),
            'document_state': doc_state_history[-1],
            'completion_time': steps
        }
    
    def _evaluate(self, num_eval_episodes: int = 5) -> Dict:
        """Evaluate the agent's performance."""
        eval_rewards = []
        eval_completion_times = []
        
        # Store current epsilon and set to 0 for evaluation
        current_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        for _ in range(num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                if done:
                    break
                    
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_completion_times.append(steps)
        
        # Restore epsilon
        self.agent.epsilon = current_epsilon
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_completion_time': np.mean(eval_completion_times)
        }
    
    def _calculate_action_distribution(self, actions: List[int]) -> Dict[str, float]:
        """Calculate distribution of actions taken."""
        action_types = ['review', 'update', 'validate', 'escalate', 'delegate', 'approve']
        counts = np.bincount(actions, minlength=6)
        return dict(zip(action_types, counts / len(actions)))
    
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
        
        # Plot action distribution
        action_dist_df = pd.DataFrame(self.action_distributions)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=action_dist_df)
        plt.title('Action Distribution')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.save_dir, f'action_distribution_{timestamp}.png'))
        plt.close()
        
        # Plot resource usage
        resource_df = pd.DataFrame(self.resource_usage, 
                                 columns=['Time', 'Personnel', 'Tools'])
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=resource_df)
        plt.title('Resource Usage')
        plt.ylabel('Usage Level')
        plt.savefig(os.path.join(self.save_dir, f'resource_usage_{timestamp}.png'))
        plt.close()
        
        # Plot completion times
        plt.figure(figsize=(12, 6))
        plt.plot(self.completion_times)
        plt.xlabel('Episode')
        plt.ylabel('Steps to Completion')
        plt.title('Task Completion Times')
        plt.savefig(os.path.join(self.save_dir, f'completion_times_{timestamp}.png'))
        plt.close()
        
        # Plot latest document states
        doc_state_df = pd.DataFrame([self.document_states[-1]], 
                                  columns=['Complete', 'Validated', 'Errors',
                                         'Feedback', 'Reviewed', 'Approved'])
        plt.figure(figsize=(12, 6))
        sns.heatmap(doc_state_df, annot=True, cmap='YlOrRd', cbar=False)
        plt.title('Latest Document State')
        plt.savefig(os.path.join(self.save_dir, f'document_state_{timestamp}.png'))
        plt.close()
    
    def _log_progress(self, episode: int, total_episodes: int):
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards[-10:])
        avg_completion_time = np.mean(self.completion_times[-10:])
        
        logging.info(
            f"Episode {episode}/{total_episodes} - "
            f"Average Reward: {avg_reward:.2f}, "
            f"Average Completion Time: {avg_completion_time:.1f}, "
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
            'action_distributions': self.action_distributions,
            'resource_usage': self.resource_usage,
            'completion_times': self.completion_times
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
            'action_distributions': self.action_distributions,
            'resource_usage': self.resource_usage,
            'completion_times': self.completion_times,
            'final_document_state': self.document_states[-1].tolist()
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
        'num_episodes': 1000,
        'eval_interval': 50,
        'save_interval': 100,
        'visualize_interval': 50,
        'max_steps': 50
    }
    
    # Create trainer
    trainer = ActionPlannerTrainer(
        max_steps=config['max_steps']
    )
    
    # Start training
    print("Starting training...")
    results = trainer.train(
        num_episodes=config['num_episodes'],
        eval_interval=config['eval_interval'],
        save_interval=config['save_interval'],
        visualize_interval=config['visualize_interval']
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best average reward: {results['best_avg_reward']:.2f}")
    print(f"Final average reward: {results['final_avg_reward']:.2f}")
    print(f"Final completion time: {results['final_completion_time']:.1f}")
    print(f"Training time: {results['training_time']}")

if __name__ == "__main__":
    main()