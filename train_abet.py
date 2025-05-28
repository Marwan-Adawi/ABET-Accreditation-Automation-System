import os
from agents.coordinator_agent import CoordinatorAgent
from environments.coordinator_env import CoordinatorEnvironment
from utils.visualization import TrainingVisualizer

def main():
    # Create necessary directories
    os.makedirs("abet_results", exist_ok=True)
    os.makedirs("abet_logs", exist_ok=True)
    os.makedirs("abet_checkpoints", exist_ok=True)
    
    # Initialize the coordinator
    coordinator = CoordinatorAgent(
        num_tasks=10,  # Number of ABET tasks to manage
        save_dir="abet_results",
        log_dir="abet_logs",
        checkpoint_dir="abet_checkpoints"
    )
    
    # Start training
    print("Starting ABET automation training...")
    rewards = coordinator.train(
        num_episodes=1000,
        eval_interval=50,
        save_interval=100
    )
    
    print("Training completed!")
    print(f"Final average reward: {sum(rewards[-10:])/10:.2f}")

if __name__ == "__main__":
    main() 