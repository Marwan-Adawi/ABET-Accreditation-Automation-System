import torch
import numpy as np
from agents.docgen_agent import DocGenAgent
from agents.environments.docgen_env import DocGenEnvironment
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import logging
from typing import List, Dict, Tuple
import pandas as pd

class DocGenTrainer:
    def __init__(self, 
                 save_dir: str = "training_results/docgen",
                 log_dir: str = "logs/docgen",
                 checkpoint_dir: str = "checkpoints/docgen",
                 max_steps: int = 100):
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
        self.env = DocGenEnvironment(max_steps=max_steps)
        self.agent = DocGenAgent(self.env)
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.quality_metrics_history = []
        self.abet_coverage_history = []
        self.content_progress_history = []
        self.doc_state_history = []
        self.action_history = []
        
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
        """Train the docgen agent."""
        best_avg_reward = float('-inf')
        training_start = datetime.now()
        
        for episode in range(num_episodes):
            # Run one episode
            episode_metrics = self._run_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_metrics['reward'])
            self.quality_metrics_history.append(episode_metrics['quality_metrics'])
            self.abet_coverage_history.append(episode_metrics['abet_coverage'])
            self.content_progress_history.append(episode_metrics['content_progress'])
            self.doc_state_history.append(episode_metrics['doc_state'])
            self.action_history.append(episode_metrics['actions'])
            
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
            'final_quality_metrics': np.mean(self.quality_metrics_history[-10:], axis=0),
            'final_abet_coverage': np.mean(self.abet_coverage_history[-10:], axis=0)
        }
        
        self._save_results(final_metrics)
        return final_metrics
    
    def _run_episode(self) -> Dict:
        """Run one training episode."""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_actions = []
        
        while True:
            # Select and take action
            action = self.agent.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            
            # Track actions
            episode_actions.append(action)
            
            # Store transition
            self.agent.memory.append((state, action, reward, next_state, done))
            
            # Optimize model
            self.agent._optimize_model()
            
            if done:
                break
                
            state = next_state
        
        return {
            'reward': episode_reward,
            'quality_metrics': next_state['quality_metrics'],
            'abet_coverage': next_state['abet_criteria'],
            'content_progress': next_state['course_content'],
            'doc_state': next_state['doc_state'],
            'actions': episode_actions
        }
    
    def _evaluate(self, num_eval_episodes: int = 5) -> Dict:
        """Evaluate the agent's performance."""
        eval_rewards = []
        eval_quality = []
        eval_coverage = []
        
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
                    eval_quality.append(next_state['quality_metrics'])
                    eval_coverage.append(next_state['abet_criteria'])
                    break
                    
                state = next_state
            
            eval_rewards.append(episode_reward)
        
        # Restore epsilon
        self.agent.epsilon = current_epsilon
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_quality': np.mean(eval_quality, axis=0),
            'avg_coverage': np.mean(eval_coverage, axis=0)
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
        
        # Plot quality metrics
        quality_df = pd.DataFrame(self.quality_metrics_history,
                                columns=['Completeness', 'Consistency', 'Clarity',
                                       'Alignment', 'Measurability'])
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=quality_df)
        plt.title('Document Quality Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'quality_metrics_{timestamp}.png'))
        plt.close()
        
        # Plot ABET criteria coverage
        coverage_df = pd.DataFrame(self.abet_coverage_history,
                                 columns=[f'SO{i+1}' for i in range(7)])
        plt.figure(figsize=(12, 6))
        sns.heatmap(coverage_df.iloc[-1:], annot=True, cmap='YlOrRd')
        plt.title('Final ABET Criteria Coverage')
        plt.savefig(os.path.join(self.save_dir, f'abet_coverage_{timestamp}.png'))
        plt.close()
        
        # Plot content progress
        content_df = pd.DataFrame(self.content_progress_history,
                                columns=['Syllabus', 'Outcomes', 'Assessments',
                                       'Materials', 'Schedule', 'Policies',
                                       'Prerequisites', 'Description'])
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=content_df)
        plt.title('Course Content Progress')
        plt.xlabel('Episode')
        plt.ylabel('Completion Level')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'content_progress_{timestamp}.png'))
        plt.close()
        
        # Plot document state transitions
        doc_state_df = pd.DataFrame(self.doc_state_history,
                                  columns=['Draft', 'Reviewed', 'Approved',
                                         'Published', 'Archived', 'Needs Update'])
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=doc_state_df)
        plt.title('Document State Transitions')
        plt.xlabel('Episode')
        plt.ylabel('State Probability')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'doc_states_{timestamp}.png'))
        plt.close()
        
        # Plot action distributions
        action_types = ['create', 'update', 'revise', 'align', 'verify']
        action_counts = np.zeros(len(action_types))
        for actions in self.action_history[-100:]:  # Last 100 episodes
            for action in actions:
                action_counts[action['action_type']] += 1
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=action_types, y=action_counts)
        plt.title('Action Type Distribution (Last 100 Episodes)')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'action_distribution_{timestamp}.png'))
        plt.close()
    
    def _log_progress(self, episode: int, total_episodes: int):
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards[-10:])
        avg_quality = np.mean(self.quality_metrics_history[-10:], axis=0)
        avg_coverage = np.mean(self.abet_coverage_history[-10:], axis=0)
        
        logging.info(
            f"Episode {episode}/{total_episodes}\n"
            f"Average Reward: {avg_reward:.2f}\n"
            f"Average Quality: {avg_quality.mean():.2f}\n"
            f"Average ABET Coverage: {avg_coverage.mean():.2f}\n"
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
            'quality_metrics_history': self.quality_metrics_history,
            'abet_coverage_history': self.abet_coverage_history,
            'content_progress_history': self.content_progress_history,
            'doc_state_history': self.doc_state_history
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
            'final_quality_metrics': self.quality_metrics_history[-1].tolist(),
            'final_abet_coverage': self.abet_coverage_history[-1].tolist(),
            'final_content_progress': self.content_progress_history[-1].tolist(),
            'final_doc_state': self.doc_state_history[-1].tolist()
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
        'max_steps': 100
    }
    
    # Create trainer
    trainer = DocGenTrainer(
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
    print(f"Final quality metrics average: {np.mean(results['final_quality_metrics']):.2f}")
    print(f"Final ABET coverage average: {np.mean(results['final_abet_coverage']):.2f}")
    print(f"Training time: {results['training_time']}")

if __name__ == "__main__":
    main()