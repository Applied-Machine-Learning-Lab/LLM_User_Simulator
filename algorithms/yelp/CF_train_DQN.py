import argparse
import os
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.distributions import CategoricalDistribution

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import TensorDict
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Callable, Tuple, Union
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

import wandb
import gymnasium as gym

from algorithms.wrappers import StableBaselineWrapperNum
from environment.yelp.configs import (
    get_enviroment_from_args,
    get_base_parser,
)


# Define arguments
def parse_args():
    parser = get_base_parser()
    parser.add_argument("--model-device", type=str, default="cuda:0")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--embedding-dim", type=int, default=32)
    args = parser.parse_args()
    return args


from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)  # number of actions
        q_net = create_mlp(
            self.features_dim, action_dim, self.net_arch, self.activation_fn
        )
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        obs = self.extract_features(obs, self.features_extractor)
        values = self.q_net(obs["obs"])
        return values

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return QNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode


class ExtractPass(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

        self.user_embedding = nn.Embedding(train_env.num_users, 32)
        self._features_dim = 32 + train_env.num_items

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations["user_id"] = observations["user_id"].int()

        user_id = observations["user_id"].squeeze(1)
        film_seen = observations["items_interact"]
        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, film_seen], dim=1)

        obs = {
            "obs": user_embedding_value,
        }
        return obs


if __name__ == "__main__":
    args = parse_args()

    train_env = get_enviroment_from_args(args)

    test_env = get_enviroment_from_args(
        args,
        seed=args.seed + 600,
    )

    # Create the custom actor-critic policy
    policy_kwargs = dict(
        features_extractor_class=ExtractPass,
    )

    train_env = StableBaselineWrapperNum(train_env)
    test_env = Monitor(StableBaselineWrapperNum(test_env))
    check_env(train_env)
    check_env(test_env)

    # Initialize wandb
    run = wandb.init(
        project="MPR",
        config=args,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,
        dir="./tmp/wandb",
    )

    model = DQN(
        DQNPolicy,
        train_env,
        verbose=1,
        buffer_size=250000,
        policy_kwargs=policy_kwargs,
        device=args.model_device,
        tensorboard_log=f"./tmp/runs/{run.id}",
        gamma=args.gamma,
    )

    wandb_callback = WandbCallback(
        model_save_path=f"./tmp/models/{run.id}",
        verbose=2,
        gradient_save_freq=100,
    )

    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=f"./tmp/models/{run.id}",
        log_path=f"./tmp/models/{run.id}",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 * 20,
        save_path=f"./tmp/models/{run.id}",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    callback = CallbackList([wandb_callback, eval_callback, checkpoint_callback])

    print(model.policy)
    print(args)
    model.learn(total_timesteps=1600000, progress_bar=False, callback=callback)

    run.finish()
