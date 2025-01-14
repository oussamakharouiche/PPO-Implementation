import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from torch import nn
from distutils.util import strtobool
from torch.distributions import Categorical
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, help="the name of the experiment")
    parser.add_argument("--share", type=lambda x: bool(strtobool(x)), default=False, help="if the critic and the actor share model weights")
    parser.add_argument("--seed", type=int, default=40, help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--total-timestamp", type=int, default=1000000, help="total number of steps performed during the training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=128, help="number of steps to run in each env")
    parser.add_argument("--n-envs", type=int, default=16, help="the number of env run in parallel")
    parser.add_argument("--num-minibatches", type=int, default=8, help="number of mini-batches")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="used for advantage estimation")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--n-epochs", type=int, default=10, help="number of epoch when optimizing the surrogate loss")
    parser.add_argument("--ent-coeff", type=float, default=0.01, help="Entropy coefficient for the loss calculation")
    parser.add_argument("--vf-coeff", type=float, default=0.5, help="value function coefficient for the loss calculation")

    args = parser.parse_args()

    return args
    
    


class Shared_Model(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        super(Shared_Model, self).__init__()
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.layer1 = nn.Linear(self.obs_space_shape, 64)
        self.layer2 = nn.Linear(64,64)
        self.layer3_actor = nn.Linear(64,action_space_shape)
        self.layer3_critic = nn.Linear(64,1)
        self.activation = nn.ReLU()
    
    def forward(self, obs):
        h = self.activation(self.layer1(obs))
        h = self.activation(self.layer2(h))

        pi = Categorical(logits=self.layer3_actor(h))
        value = self.layer3_critic(h)

        return pi, value

class Split_Model(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        super(Split_Model, self).__init__()
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.actor = nn.Sequential(
            nn.Linear(self.obs_space_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,action_space_shape)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_space_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    
    def forward(self, obs):
        pi = Categorical(logits=self.actor(obs))
        value = self.critic(obs)
        return pi, value
    
class Trainer:
    def __init__(
        self, 
        writer,
        params
    ):
        self.num_steps = params.num_steps
        self.n_envs = params.n_envs
        self.env = make_vec_env('LunarLander-v3', n_envs=self.n_envs)
        self.model = Shared_Model(self.env.observation_space.shape[0], self.env.action_space.n) if params.share else Split_Model(self.env.observation_space.shape[0], self.env.action_space.n)
        self.batch_size = self.num_steps*self.n_envs
        self.mini_batch_size = self.batch_size//params.num_minibatches
        self.gamma = params.gamma
        self.lambda_ = params.gae_lambda
        self.clip_eps = params.clip_eps
        self.c1 = params.vf_coeff
        self.c2 = params.ent_coeff
        self.n_epochs = params.n_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() and params.cuda else "cpu")
        self.model.to(self.device)
        self.obs = self.to_tensor(self.env.reset()).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.learning_rate, eps=1e-5)
        self.update = (params.total_timestamp+self.batch_size-1)//self.batch_size
        self.writer = writer
        self.loss_step = 0
        self.reward_step = 0

    def to_tensor(self,arr):
        return torch.tensor(arr, dtype=torch.float)
        

    def sample(self):
        rewards = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        actions = torch.zeros((self.n_envs, self.num_steps), dtype=torch.long)
        dones = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        observations = torch.zeros((self.n_envs, self.num_steps, self.env.observation_space.shape[0]), dtype=torch.float)
        values = torch.zeros((self.n_envs, self.num_steps+1), dtype=torch.float)
        log_probs = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)

        with torch.no_grad():
            for t in range(self.num_steps):
                observations[:,t] = self.obs
                
                pi,v = self.model(self.obs)

                action = pi.sample()
                actions[:,t] = action
                values[:,t] = v.reshape(self.n_envs,).detach()
                log_probs[:,t] = pi.log_prob(action).detach()

                self.obs, reward, done, _ =  self.env.step(action.cpu().numpy())
                self.obs = self.to_tensor(self.obs).to(self.device)
                dones[:,t] = self.to_tensor(done)
                rewards[:,t] = self.to_tensor(reward)


            _, v = self.model(self.obs)
            values[:,self.num_steps] = v.reshape(self.n_envs,)
        
        advantages = self.GAE(values, rewards, dones)
        # import pdb; pdb.set_trace()
        return {
            'observations': observations.reshape(self.batch_size, *observations.shape[2:]),
            'actions': actions.reshape(self.batch_size, *actions.shape[2:]),
            'values': values[:,:-1].reshape(self.batch_size, *values.shape[2:]),
            'log_prob': log_probs.reshape(self.batch_size, *log_probs.shape[2:]),
            'advantages': advantages.reshape(self.batch_size, *advantages.shape[2:]),
            'rewards': rewards.reshape(self.batch_size, *advantages.shape[2:])
        }
    

    def train(self, samples):
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

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                total_norm = 0.0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar(
                    "grad_norm",
                    total_norm, 
                    global_step = self.loss_step-1
                )


                self.optimizer.step()


    def compute_loss(self, samples):
        sample_ret = samples["values"]+samples["advantages"]

        pi,values = self.model(samples["observations"])

        log_probs = pi.log_prob(samples["actions"])

        # import pdb; pdb.set_trace()
        adv_norm = self.normalize(samples["advantages"])

        loss = (
            -self.ppo_clip(log_probs, samples["log_prob"], adv_norm).mean()
            + self.c1 * 0.5 * ((sample_ret-values)**2).mean()
            - self.c2 * pi.entropy().mean() 
        )
        self.writer.add_scalar("global_train_loss",loss, global_step = self.loss_step)
        self.writer.add_scalar(
            "ppo_train_loss",
            self.ppo_clip(log_probs, samples["log_prob"], adv_norm).mean(), 
            global_step = self.loss_step
        )
        self.writer.add_scalar(
            "value_train_loss",
            ((sample_ret-values)**2).mean(), 
            global_step = self.loss_step
        )
        self.writer.add_scalar(
            "entropy_train_loss",
            pi.entropy().mean() , 
            global_step = self.loss_step
        )
        self.loss_step+=1

        return loss

    
    def GAE(self, values, rewards, dones):
        """
        Generalized Advantage Estimation
        """

        advantages = torch.zeros_like(rewards)
        last_advantages = 0

        for t in reversed(range(self.num_steps)):
            delta = rewards[:,t] + self.gamma * values[:,t+1] * (1.0 - dones[:,t]) - values[:,t]

            advantages[:,t] = last_advantages = delta + self.gamma * self.lambda_ * (1.0 - dones[:,t]) * last_advantages

        return advantages
    
    def ppo_clip(self,log_prob, log_prob_old, advantages):
        ratio = torch.exp(log_prob-log_prob_old)

        loss = ratio * advantages
        loss_clip = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages

        return torch.min(loss, loss_clip)
    
    def normalize(self, vect):
        return (vect - vect.mean()) / (vect.std() + 1e-8)
    
    def training_loop(self):

        for e in range(self.update):
            print(f"{e+1}/{self.update}")
            samples = self.sample()
            self.writer.add_scalar(
                "mean_reward",
                samples["rewards"].mean(), 
                global_step = self.reward_step
            )
            self.reward_step += 1
            self.train(samples)

    def test_policy(self, n_eval_episodes=10):
        env = gym.make("LunarLander-v3")
        rewards = [0]*n_eval_episodes
        for n in range(n_eval_episodes):
            observation,_ = env.reset()
            done = False
            while not done:
                pi, _ = self.model(self.to_tensor(observation).reshape(1,-1).to(self.device))
                a = torch.argmax(pi.logits).item()
                observation, reward, done, _, _ = env.step(a)
                rewards[n] += reward
        return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    params = parse_args()
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    

    writer = SummaryWriter(f"runs/{params.exp_name}")
    trainer = Trainer(writer, params)
    trainer.training_loop()
    print(trainer.test_policy())

    # env = make_vec_env('LunarLander-v3', n_envs=16)
    # from stable_baselines3 import PPO

    # model = PPO(
    # policy = 'MlpPolicy',
    # env = env,
    # n_steps = 1024,
    # batch_size = 64,
    # n_epochs = 4,
    # gamma = 0.999,
    # gae_lambda = 0.98,
    # ent_coef = 0.01,
    # verbose=1)

    # model.learn(total_timesteps=1000000)

