import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StableBaselineWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {
                "user_id": spaces.Box(
                    low=0, high=1, shape=(self.num_users,), dtype=np.int_
                ),
                "items_interact": spaces.Box(
                    low=0,
                    high=11,
                    shape=(self.env.num_items,),
                    dtype=np.int_,
                ),
            }
        )

    def observation(self, observation):
        id = observation["user_id"]
        items_interact = observation["items_interact"]

        user_feature = np.zeros(self.env.num_users, dtype=np.int_)
        user_feature[id] = 1

        film_feature = np.zeros(self.env.num_items, dtype=np.int_)

        for film, rating in items_interact:
            film_feature[film] = rating

        return {"user_id": user_feature, "items_interact": film_feature}


class StableBaselineWrapperNum(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {
                "user_id": spaces.Box(
                    low=0, high=self.env.num_users, shape=(1,), dtype=np.int_
                ),
                "items_interact": spaces.Box(
                    low=0,
                    high=11,
                    shape=(self.env.num_items,),
                    dtype=np.int_,
                ),
            }
        )

    def observation(self, observation):
        id = observation["user_id"]
        items_interact = observation["items_interact"]

        film_feature = np.zeros(self.env.num_items, dtype=np.int_)

        for film, rating in items_interact:
            film_feature[film] = rating

        return {
            "user_id": np.array([id], dtype=np.int_),
            "items_interact": film_feature,
        }
