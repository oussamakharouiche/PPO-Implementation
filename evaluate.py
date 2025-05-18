import argparse
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

from models import SharedModel, SplitModel
from utils import load_config

@torch.no_grad()
def evaluate_policy(model, env, device, n_eval_episodes: int):
    rewards = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device)
            action_dist, _ = model(obs_tensor)
            action = torch.argmax(action_dist.logits, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        if (ep+1)%10 == 0:
            print(f"Episode {ep+1}: reward = {total_reward:.2f}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation over {n_eval_episodes} episodes: mean reward = {mean_reward:.2f}, std = {std_reward:.2f}")
    return mean_reward, std_reward

@torch.no_grad()
def record_video(model, env_name, video_folder, device, n_episodes: int, name_prefix: str):
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, name_prefix=name_prefix)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device)
            action_dist, _ = model(obs_tensor)
            action = torch.argmax(action_dist.logits, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    print(f"Saved videos to {video_folder}")


def main(config_path: str, eval_episodes: int, record: bool, record_episodes: int):
    params = load_config(config_path)
    np.random.seed(params.get("seed", 0))
    torch.manual_seed(params.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() and params.get("cuda", False) else "cpu")
    env_name = params["gym_env"]

    # build model
    temp_env = gym.make(env_name)
    obs_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.n
    temp_env.close()

    model = SharedModel(obs_dim, act_dim) if params.get("share", False) else SplitModel(obs_dim, act_dim)
    model.to(device)

    checkpoint_path = os.path.join(os.path.dirname(__file__),f"model_checkpoints/{params['save_file']}.pt")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("\n---- Running Evaluation ----")
    env = gym.make(env_name)
    evaluate_policy(model, env, device, eval_episodes)
    env.close()

    if record:
        print("\n---- Recording Video Demo ----")
        record_video(model, env_name, "./tests", device, record_episodes, params.get("exp_name"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and/or record a trained PPO model.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the YAML config file used during training.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes (default: 10).")
    parser.add_argument("--record", type=lambda x: bool(int(x)), default=True, help="Whether to record video demos (0 or 1).")
    parser.add_argument("--record-episodes", type=int, default=1, help="Number of episodes to record if --record is set.")
    args = parser.parse_args()

    main(
        args.config_path,
        args.eval_episodes,
        args.record,
        args.record_episodes
    )
