import pandas as pd
import os
from src.data.dataset_loader import DatasetLoader
from src.utils.config import DATA_DIR, PROJECT_DATA_DIR
from src.synthetic_dataset.dataset_generator import import_dataset_full


class SyntheticLoader(DatasetLoader):
    def __init__(self, name="synthetic-11", data_dir=PROJECT_DATA_DIR, **kwargs):
        """
        Args:
            dataset_name: Name of the synthetic dataset folder (e.g., 'synthetic-11')
            **kwargs: Additional arguments passed to DatasetLoader
        """
        super().__init__(**kwargs)
        self.dataset_name = name
        self.base_path = os.path.join(data_dir, self.dataset_name)

        # Specify what information this dataset contains
        self.contains = ["construct_id", "question_id", "question_text"]
        self.mapping = {"construct_id": "constructs", "question_id": "question_id", "question_text": "text", "correct": "is_correct"}

    def load_data(self, file_path=None, return_full=False):
        """Load the synthetic dataset from CSV files"""
        if file_path is None:
            file_path = self.base_path

        # Import complete dataset including students, questions and interactions
        students_df, questions_df, interactions_df = import_dataset_full(file_path)

        # Merge interactions with questions to get question text
        interactions_df["question_id"] = interactions_df["question_id"].astype(int)
        data = interactions_df.merge(questions_df[["question_id", "text"]], on="question_id", how="left")
        if return_full:
            return students_df, questions_df, data
        else:
            return data

    def preprocess(self, data):
        """Process data into student trajectories"""
        # Sort by student_id to ensure chronological order
        data = data.sort_values(["student_id"])
        grouped = data.groupby("student_id")

        trajectories = []
        user_ids = []
        for u_id, group in grouped:
            user_ids.append(u_id)
            trajectory = []
            for _, row in group.iterrows():
                entry = {}
                entry["correct"] = row[self.mapping["correct"]]
                for k in self.contains:
                    # Special handling for constructs which is stored as a string representation of a list
                    if k == "construct_id":
                        # Convert string representation of list to actual list
                        entry[k] = eval(row["constructs_ids"])[0]  # Convert string to list
                        # constructs = eval(row[self.mapping[k]])
                        # entry[k] = constructs[0] if constructs else None  # Take first construct if multiple
                    else:
                        entry[k] = row[self.mapping[k]]
                trajectory.append(entry)

            if self.max_trajectory_length is not None:
                trajectory = trajectory[: self.max_trajectory_length]

            if len(trajectory) >= self.min_trajectory_length:
                trajectories.append(trajectory)

            if (self.max_n_trajectories is not None) and (len(trajectories) >= self.max_n_trajectories):
                break

        return trajectories, user_ids

    def split_data(self, data, random_state=42):
        """Split trajectories into train and test sets"""
        from sklearn.model_selection import train_test_split

        return train_test_split(data, test_size=self.test_size, random_state=random_state)


# How to use
if __name__ == "__main__":
    loader = SyntheticLoader(name="synthetic-11", data_dir=PROJECT_DATA_DIR)
    data = loader.load_data()
    trajectories, user_ids = loader.preprocess(data)
    print(f"Loaded {len(trajectories)} trajectories")
    train, test = loader.split_data(trajectories)
    print(f"Train: {len(train)} trajectories, Test: {len(test)} trajectories")
