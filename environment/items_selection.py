import typing
from environment.yelp.yelp import Yelp
import numpy as np
from abc import ABC, abstractmethod


class ItemsSelector(ABC):

    def __init__(self, seed=42):
        self.seed(seed)

    @abstractmethod
    def select(
        self, items: typing.List[Yelp], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Yelp], typing.List[float]]:
        """Function used to select items"""
        pass

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)


class GreedySelector(ItemsSelector):
    def __init__(self, seed=42):
        super().__init__(seed)

    def select(
        self, items: typing.List[Yelp], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Yelp], typing.List[float]]:
        items_ratings = list(zip(items, ratings))
        items_ratings.sort(key=lambda x: (x[1], x[0].vote_average), reverse=True)
        selected_items = [item for (item, rating) in items_ratings]
        selected_ratings = [0] * len(items)
        selected_ratings[0] = items_ratings[0][1]
        return selected_items, selected_ratings


class GreedySelectorRandom(ItemsSelector):
    def __init__(self, p=0.9, seed=42):
        self.greedy_selector = GreedySelector(seed)
        super().__init__(seed)
        self.p = p

    def select(
        self, items: typing.List[Yelp], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Yelp], typing.List[float]]:
        look_item = self.rng.choice([True, False], p=[self.p, 1 - self.p])
        if look_item:
            return self.greedy_selector.select(items, ratings)
        return items, [0] * len(items)

    def seed(self, seed: int):
        super().seed(seed)
        self.greedy_selector.seed(seed)


class RandomSelector(ItemsSelector):
    def __init__(self, p=0.5, seed=42):
        super().__init__(seed)
        self.p = p

    def select(
        self, items: typing.List[Yelp], ratings: typing.List[float]
    ) -> typing.Tuple[typing.List[Yelp], typing.List[float]]:
        random_vector = self.rng.choice([0, 1], size=len(ratings), p=[0.5, 0.5])
        selected_ratings = [a * b for a, b in zip(items, ratings)]
        return items, selected_ratings
