import torch
import numpy as np
from agents.consistency_checker_agent import ConsistencyCheckerAgent
from agents.environments.consistency_checker_env import ConsistencyCheckEnvironment
from utils.visualization import TrainingVisualizer
import json
import os
from datetime import datetime
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_sample_data(file_path: str = "sample_data.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

def train_agent(documents: dict, 
                num_episodes: int = 1000,
                save_dir: str = "models",
                visualizer: TrainingVisualizer = None,
                eval_interval: int = 50):
    """Train the consistency checker agent with visualization and logging."""
    
    # Setup
    setup_logging()
    os.makedirs(save_dir, exist_ok=True)
    
    if visualizer is None:
        visualizer = TrainingVisualizer()
    
    # Create environment and agent
    env = ConsistencyCheckEnvironment(documents)
    agent = ConsistencyCheckerAgent(env)
    
    # Training tracking
    all_rewards = []
    best_avg_reward = float('-inf')
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Store transition and optimize
            agent.memory.append((state, action, reward, next_state, done))
            agent._optimize_model()
            
            if done:
                break
                
            state = next_state
        
        # Track rewards
        all_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            logging.info(f"Episode {episode + 1}/{num_episodes} - "
                        f"Average Reward: {avg_reward:.2f}, "
                        f"Steps: {episode_steps}, "
                        f"Epsilon: {agent.epsilon:.3f}")
            
            # Visualize current state and rewards
            if (episode + 1) % 50 == 0:
                visualizer.plot_training_rewards(all_rewards)
                visualizer.plot_state_heatmap(state)
        
        # Save best model
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(all_rewards[-eval_interval:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                model_path = os.path.join(save_dir, 
                                        f"best_model_reward_{avg_reward:.2f}.pt")
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': avg_reward,
                }, model_path)
                logging.info(f"Saved best model with average reward: {avg_reward:.2f}")
    
    return agent, all_rewards

if __name__ == "__main__":
    # Load sample data
    documents = load_sample_data()
    
    # Create visualizer
    visualizer = TrainingVisualizer()
    
    # Train agent
    trained_agent, rewards = train_agent(
        documents=documents,
        num_episodes=1000,
        visualizer=visualizer
    )
    
    # Final visualization
    visualizer.plot_training_rewards(rewards)