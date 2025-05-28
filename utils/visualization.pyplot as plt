import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
from datetime import datetime
import os

class TrainingVisualizer:
    def __init__(self, save_dir: str = "training_plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_rewards(self, rewards: List[float], 
                            window_size: int = 100,
                            show: bool = True,
                            save: bool = True):
        """Plot training rewards with moving average."""
        plt.figure(figsize=(12, 6))
        
        # Plot raw rewards
        plt.plot(rewards, alpha=0.3, label='Raw Rewards')
        
        # Plot moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            plt.plot(range(window_size-1, len(rewards)), 
                    moving_avg, 
                    label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards Over Time')
        plt.legend()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.save_dir}/training_rewards_{timestamp}.png")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_state_heatmap(self, state: Dict, show: bool = True, save: bool = True):
        """Plot heatmap of current state."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot CLO-PLO mapping
        sns.heatmap(state['clo_plo_mapping'], 
                   ax=axes[0,0], 
                   cmap='YlOrRd',
                   xticklabels=[f'PLO{i+1}' for i in range(state['clo_plo_mapping'].shape[1])],
                   yticklabels=[f'CLO{i+1}' for i in range(state['clo_plo_mapping'].shape[0])])
        axes[0,0].set_title('CLO-PLO Mapping')
        
        # Plot Assessment Methods
        sns.heatmap(state['assessment_methods'], 
                   ax=axes[0,1],
                   cmap='YlOrRd',
                   xticklabels=[f'M{i+1}' for i in range(state['assessment_methods'].shape[1])],
                   yticklabels=[f'CLO{i+1}' for i in range(state['assessment_methods'].shape[0])])
        axes[0,1].set_title('Assessment Methods')
        
        # Plot Score Validity
        sns.heatmap(state['scores_valid'].reshape(-1, 1), 
                   ax=axes[1,0],
                   cmap='YlOrRd',
                   xticklabels=['Valid'],
                   yticklabels=[f'CLO{i+1}' for i in range(len(state['scores_valid']))])
        axes[1,0].set_title('Score Validity')
        
        # Plot Syllabus Match
        sns.heatmap(state['syllabus_match'].reshape(-1, 1),
                   ax=axes[1,1],
                   cmap='YlOrRd',
                   xticklabels=['Match'],
                   yticklabels=[f'CLO{i+1}' for i in range(len(state['syllabus_match']))])
        axes[1,1].set_title('Syllabus Match')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.save_dir}/state_heatmap_{timestamp}.png")
        
        if show:
            plt.show()
        else:
            plt.close()