from abc import ABC, abstractmethod
from typing import List


class Item:
    id: str
    display_name: str

    def __init__(self, id, display_name):
        self.id = id
        self.display_name = display_name

    def __eq__(self, other_item: object) -> bool:
        """
        Define equality between two items if they have the same id
        """
        return self.id == other_item.id


class ItemsLoader(ABC):
    def __init__(self, name_dataset) -> None:
        pass

    @abstractmethod
    def load_all_ids(self) -> List[int]:
        pass

    @abstractmethod
    def load_items(self) -> List[Item]:
        pass

    @abstractmethod
    def load_items_from_ids(self, ids: List[int]) -> List[Item]:
        pass
