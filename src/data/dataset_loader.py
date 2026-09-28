from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.utils.config import DATA_DIR


# Datasets can contain questions or only question_id, and constructs or only construct_id
# self.contains = ["question_text", "question_id", "construct_text", "construct_id"]
# To be specified in the dataset loader to known how to build the prompt.

class DatasetLoader(ABC):
    def __init__(self, min_trajectory_length=10, max_trajectory_length=100, max_n_trajectories=None, **kwargs):
        self.max_trajectory_length = max_trajectory_length  
        self.min_trajectory_length = min_trajectory_length  
        self.max_n_trajectories = max_n_trajectories  
        self.test_size = kwargs.get("test_size", 0.2)  # default to 1/5th for test

    @abstractmethod
    def load_data(self, file_path):
        pass
    
    @abstractmethod
    def preprocess(self, data):
        """Process data into student trajectories and user IDs.
        
        Args:
            data: Raw data to process
            
        Returns:
            tuple: (trajectories, user_ids) where:
                - trajectories: List of student trajectories
                - user_ids: List of corresponding user IDs
        """
        pass
    
    @abstractmethod
    def split_data(self, data, test_size=0.2, random_state=42):
        pass



