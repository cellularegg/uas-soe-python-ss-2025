from abc import ABC, abstractmethod
from typing import List, Any

class Model(ABC):
  @abstractmethod
  def load(self, path) -> None:
    pass

  @abstractmethod
  def recommend(self, movie_rating: dict, number_of_recommendations: int) -> List[Any]:
    pass
