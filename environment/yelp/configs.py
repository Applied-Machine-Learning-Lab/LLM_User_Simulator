import argparse
import os

from environment.yelp.yelp_loader import YelpLoader
from ..env import UserSim
from ..items_selection import GreedySelector
from gymnasium.utils.env_checker import check_env

def get_base_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_enviroment_from_args(
    args, seed=None, render_mode=None, render_path=None, eval_mode=False
):
    """Returns the environment with the configuration specified in args."""
    if seed is None:
        seed = args.seed
    env = UserSim(
        render_mode=render_mode,
        render_path=render_path,
        items_loader=YelpLoader(),
        items_selector=GreedySelector(seed),
        evaluation=eval_mode,
    )
    env.reset(seed=seed)
    return env
