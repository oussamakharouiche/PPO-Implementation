import argparse
from distutils.util import strtobool
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import wandb 
import uuid

from models import SharedModel, SplitModel
from utils import normalize, load_config
    
class Trainer:
    """
    Trainer class for running PPO training on a Gym environment.
    """

    def __init__(
        self,
        params: dict
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            config (dict): Dictionary containing training hyperparameters.
            writer (SummaryWriter): TensorBoard writer for logging.
        """
        self.num_steps = params["num_steps"]
        self.n_envs = params["n_envs"]
        self.gym_env_name = params["gym_env"]
        self.env = make_vec_env(self.gym_env_name, n_envs=self.n_envs)
        self.model = SharedModel(self.env.observation_space.shape[0], self.env.action_space.n) if params["share"] else SplitModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.batch_size = self.num_steps*self.n_envs
        self.mini_batch_size = self.batch_size//params["num_minibatches"]
        self.gamma = params["gamma"]
        self.gae_lambda = params["gae_lambda"]
        self.clip_eps = params["clip_eps"]
        self.c1 = params["vf_coeff"]
        self.c2 = params["ent_coeff"]
        self.n_epochs = params["n_epochs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() and params["cuda"] else "cpu")
        self.model.to(self.device)
        observation = self.env.reset()
        initial_obs = observation[0] if isinstance(observation, tuple) else observation
        self.obs = self.to_tensor(initial_obs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["learning_rate"], eps=1e-5)
        self.update = (params["total_timestamp"]+self.batch_size-1)//self.batch_size
        self.loss_step = 0
        self.reward_step = 0
        self.global_step = 0
        self.anneal_lr = params["anneal_lr"]
        self.lr = params["learning_rate"]
        self.total_timesteps_elapsed = 0
        os.makedirs(os.path.join(os.path.dirname(__file__),"model_checkpoints"), exist_ok=True)
        self.ckpt_file = os.path.join(os.path.dirname(__file__),f"model_checkpoints/{params['save_file']}.pt")


    def to_tensor(self,arr: np.ndarray) -> torch.Tensor:
        """
        Convert an array-like object to a PyTorch tensor.

        Args:
            arr: Input array.

        Returns:
            torch.Tensor: Converted tensor.
        """
        return torch.tensor(arr, dtype=torch.float)
        
    @torch.no_grad()
    def sample(self) -> dict:
        """
        Collect samples from the environment using the current policy.

        Returns:
            dict: Dictionary containing observations, actions, values, log probabilities,
                  advantages, and rewards.
        """
        rewards = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        actions = torch.zeros((self.n_envs, self.num_steps), dtype=torch.long)
        dones = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        observations = torch.zeros((self.n_envs, self.num_steps, self.env.observation_space.shape[0]), dtype=torch.float)
        values = torch.zeros((self.n_envs, self.num_steps+1), dtype=torch.float)
        log_probs = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)

        for t in range(self.num_steps):
            observations[:,t] = self.obs
            
            action_distribution,v = self.model(self.obs)
            action = action_distribution.sample()
            actions[:,t] = action
            values[:,t] = v.reshape(self.n_envs,).detach()
            log_probs[:,t] = action_distribution.log_prob(action).detach()

            self.obs, reward, done, info =  self.env.step(action.cpu().numpy())
            self.obs = self.to_tensor(self.obs).to(self.device)
            dones[:,t] = self.to_tensor(done)
            rewards[:,t] = self.to_tensor(reward)

            
            for item in info:
                if "episode" in item.keys():
                    wandb.log({
                        "episode/episodic_return": item["episode"]["r"],
                        "episode/episodic_length": item["episode"]["l"],
                        "global_step": self.global_step
                    })

            self.global_step+=1

        _, v = self.model(self.obs)
        values[:,self.num_steps] = v.reshape(self.n_envs)
        
        advantages = self.GAE(values, rewards, dones)

        return {
            'observations': observations.reshape(self.batch_size, *observations.shape[2:]),
            'actions': actions.reshape(self.batch_size, *actions.shape[2:]),
            'values': values[:,:-1].reshape(self.batch_size, *values.shape[2:]),
            'log_prob': log_probs.reshape(self.batch_size, *log_probs.shape[2:]),
            'advantages': advantages.reshape(self.batch_size, *advantages.shape[2:]),
            'rewards': rewards.reshape(self.batch_size, *advantages.shape[2:])
        }
    

    def train(self, samples: dict) -> None:
        """
        Train the model for a fixed number of epochs using mini-batches.

        Args:
            samples (dict): Dictionary containing training samples.
        """
        for _ in range(self.n_epochs):
            idx = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_idx = idx[start:end]

                mini_batch_samples = {
                    k: v[mini_batch_idx].to(self.device) for k,v in samples.items()
                }

                loss = self.compute_loss(mini_batch_samples)
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                # Calculate and log gradient norm
                total_norm = sum(
                    param.grad.data.norm(2).item() ** 2 for param in self.model.parameters() if param.grad is not None
                ) ** 0.5
                
                wandb.log({
                    "train/grad_norm": total_norm, 
                    "loss_step": self.loss_step
                })


                self.optimizer.step()


    def compute_loss(self, samples: dict) -> torch.Tensor:
        """
        Compute the PPO loss for a mini-batch.

        Args:
            samples (dict): Mini-batch samples.

        Returns:
            torch.Tensor: Computed loss.
        """
        sample_ret = samples["values"]+samples["advantages"]
        old_values = samples["values"]
        action_distribution,values = self.model(samples["observations"])
        values = values.squeeze(1)
        log_probs = action_distribution.log_prob(samples["actions"])
        adv_norm = normalize(samples["advantages"])

        
        # Policy loss (surrogate objective)
        policy_objective = self.ppo_clip(log_probs, samples["log_prob"], adv_norm).mean()

        # Value loss
        value_f = (sample_ret - values)**2
        value_pred_clipped = old_values + torch.clamp(values - old_values, -self.clip_eps, self.clip_eps)
        value_f_clipped = (value_pred_clipped - sample_ret)**2
        vf_loss_term = 0.5 * torch.max(value_f, value_f_clipped).mean()

        # Entropy bonus
        entropy_term = action_distribution.entropy().mean()

        loss = -policy_objective + self.c1 * vf_loss_term - self.c2 * entropy_term

        wandb.log({
            "loss/total_loss": loss.item(),
            "loss/policy_objective": policy_objective.item(), # This is the term to be maximized
            "loss/value_function_loss": vf_loss_term.item(), # This is the actual vf component in loss
            "loss/entropy_bonus": entropy_term.item(),
            "debug/value_loss_unclipped_mse": value_f.mean().item(),
            "loss_step": self.loss_step
        })

        self.loss_step+=1

        return loss

    
    def GAE(self, values: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized Advantage Estimation (GAE).

        Args:
            values (torch.Tensor): Estimated state values.
            rewards (torch.Tensor): Rewards collected.
            dones (torch.Tensor): Binary flags indicating episode termination.

        Returns:
            torch.Tensor: Computed advantages.
        """

        advantages = torch.zeros_like(rewards)
        last_advantages = 0
        for t in reversed(range(self.num_steps)):
            delta = rewards[:,t] + self.gamma * values[:,t+1] * (1.0 - dones[:,t]) - values[:,t]
            advantages[:,t] = last_advantages = delta + self.gamma * self.gae_lambda * (1.0 - dones[:,t]) * last_advantages
        return advantages
    
    def ppo_clip(self,log_prob: torch.Tensor, log_prob_old: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Calculate the PPO clipped objective.

        Args:
            log_prob (torch.Tensor): New log probabilities.
            log_prob_old (torch.Tensor): Old log probabilities.
            advantages (torch.Tensor): Advantage estimates.

        Returns:
            torch.Tensor: Clipped PPO loss.
        """
        ratio = torch.exp(log_prob-log_prob_old)
        loss = ratio * advantages
        loss_clip = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
        return torch.min(loss, loss_clip)
    
    def training_loop(self):
        """
        Main training loop.
        """
        for e in range(self.update):
            print(f"{e+1}/{self.update}")
            if self.anneal_lr:
                coeff = 1 - (e/self.update)
                self.optimizer.param_groups[0]["lr"] = coeff * self.lr
            samples = self.sample()
            wandb.log({
                "rollout/mean_reward_in_batch": samples["rewards"].mean().item(), 
                "reward_step": self.reward_step
            })

            self.reward_step += 1
            self.train(samples)
        
        torch.save(self.model.state_dict(), self.ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="The path to the YAML config file for hyperparameters")
    args = parser.parse_args()

    params = load_config(args.config_path)

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    exp_name = params["exp_name"]
    
    wandb.init(
        project=params.get("wandb_project", "PPO_Gymnasium_Implementation"),
        name=exp_name,
        id=str(uuid.uuid4()), 
        config=params,
        sync_tensorboard=False
    )
    wandb.define_metric("train/grad_norm", step_metric="loss_step")
    wandb.define_metric("loss/*", step_metric="loss_step")
    wandb.define_metric("debug/value_loss_unclipped_mse", step_metric="loss_step")
    wandb.define_metric("rollout/mean_reward_in_batch", step_metric="reward_step")
    wandb.define_metric("episode/*", step_metric="global_step")

    trainer = Trainer(params)
    trainer.training_loop()
    print(trainer.test_policy())

    trainer.record_video_demo("./tests/", name_prefix=exp_name)

