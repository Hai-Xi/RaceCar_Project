import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
import yaml
import gymnasium as gym
from tqdm import tqdm
import logging
from typing import Dict, Any, Tuple

# Update these relative imports
from src.utils.env_wrappers import make_env
from src.models.cnn_model import CNNModel
from .buffer import RolloutBuffer

class PPOTrainer:
    def __init__(self, config_path: str):
        """Initialize PPO trainer with configuration."""
        self.config = self._load_and_validate_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create environment
        try:
            self.env = make_env(self.config['environment'])
        except Exception as e:
            raise RuntimeError(f"Failed to create environment: {e}")
        
        # Initialize model and optimizer
        self.model = CNNModel(self.config['model']).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['model']['learning_rate']
        )
        
        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            self.config['training']['max_steps_per_episode'],
            self.env.observation_space.shape,
            self.env.action_space.shape[0],
            self.device
        )
        
        # Setup logging
        self.setup_logging()
        
    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required configuration keys
        required_keys = ['environment', 'model', 'training', 'ppo']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        return config
        
    def setup_logging(self):
        """Setup logging directory and configuration."""
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"logs/run_{current_time}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/models", exist_ok=True)
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Save configuration
        with open(f"{self.log_dir}/config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
    def train(self):
        """Main training loop."""
        best_reward = float('-inf')
        self.logger.info("Starting training...")
        
        try:
            for episode in tqdm(range(self.config['training']['num_episodes'])):
                # Collect trajectory
                episode_reward = self.collect_trajectory()
                
                # Update policy
                policy_loss, value_loss = self.update_policy()
                
                # Logging
                if episode % self.config['training']['log_frequency'] == 0:
                    self.logger.info(
                        f"Episode {episode}: Reward = {episode_reward:.2f}, "
                        f"Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}"
                    )
                
                # Save model if it's the best so far
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.save_model(f"{self.log_dir}/models/best_model.pt")
                
                # Regular checkpoint saving
                if episode % self.config['training']['save_frequency'] == 0:
                    self.save_model(f"{self.log_dir}/models/model_episode_{episode}.pt")
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_model(f"{self.log_dir}/models/interrupted_model.pt")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.env.close()
                
    def collect_trajectory(self) -> float:
        """Collect a trajectory of experiences."""
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0
        
        try:
            while not done:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action from policy
                with torch.no_grad():
                    action, value = self.model.act(obs_tensor)
                    
                # Take action in environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action[0].cpu().numpy())
                done = terminated or truncated
                
                # Store transition in buffer
                self.buffer.add(
                    obs,
                    action[0].cpu().numpy(),
                    reward,
                    value[0].item(),
                    action[0].cpu().numpy(),  # Using action as log_prob for now
                    done
                )
                
                obs = next_obs
                episode_reward += reward
                
            # Compute returns and advantages
            last_value = self.model.act(torch.FloatTensor(obs).unsqueeze(0).to(self.device))[1]
            self.buffer.compute_returns_and_advantages(
                last_value.item(),
                self.config['training']['gamma'],
                self.config['training']['gae_lambda']
            )
            
        except Exception as e:
            self.logger.error(f"Error during trajectory collection: {e}")
            raise
            
        return episode_reward
        
    def update_policy(self) -> Tuple[float, float]:
        """Update policy using PPO algorithm."""
        total_policy_loss = 0
        total_value_loss = 0
        num_updates = 0
        
        for _ in range(self.config['ppo']['update_epochs']):
            for batch in self.buffer.get_samples(self.config['training']['batch_size']):
                obs_batch, action_batch, returns_batch, advantage_batch, old_log_prob_batch = batch
                
                # Get current policy and value predictions
                values, action_mean = self.model.evaluate_actions(obs_batch, action_batch)
                
                # Calculate losses
                value_loss = nn.MSELoss()(values, returns_batch.unsqueeze(1))
                
                # PPO policy loss with clipping
                ratio = torch.exp(action_mean.log_prob(action_batch) - old_log_prob_batch)
                clipped_ratio = torch.clamp(ratio, 
                                          1 - self.config['ppo']['clip_range'],
                                          1 + self.config['ppo']['clip_range'])
                policy_loss = -torch.min(ratio * advantage_batch,
                                       clipped_ratio * advantage_batch).mean()
                
                # Combined loss
                loss = (policy_loss + 
                       self.config['training']['value_loss_coef'] * value_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1
                
        # Clear buffer after updates
        self.buffer.clear()
        
        return (total_policy_loss / num_updates, 
                total_value_loss / num_updates)
        
    def save_model(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'device': self.device
        }, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model and optimizer state."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current model."""
        self.model.eval()
        rewards = []
        
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _ = self.model.act(obs_tensor, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.env.step(action[0].cpu().numpy())
                    done = terminated or truncated
                    episode_reward += reward
                    
            rewards.append(episode_reward)
            
        self.model.train()
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }