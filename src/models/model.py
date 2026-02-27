"""This file contains an abstract model class used as a template for all other model classes."""
from abc import abstractmethod, ABC

class Model(ABC):
    @abstractmethod
    def initialise_model():
        """All models must have an initialisation method."""
        pass

    @abstractmethod
    def load_model():
        """All models must have an loading method."""
        pass
    
    @abstractmethod
    def save_model():
        """All models must have an saving method."""
        pass
    
    @abstractmethod
    def train_model():
        """All models must have an training method."""
        pass
    