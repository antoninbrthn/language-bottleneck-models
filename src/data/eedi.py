import pandas as pd
import os
from src.utils.config import DATA_DIR
from .dataset_loader import DatasetLoader

class EediLoader(DatasetLoader):
    def __init__(self, data_path=None, limit_nrows=None, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path or os.path.join(DATA_DIR, "eedi_private/data/")
        self.contains = ["construct_id", "question_id", "question_text", "construct_text"]
        self.mapping = {
            "construct_id": "ConstructId",
            "construct_text": "ConstructName",
            "question_id": "QuestionId",
            "question_text": "QuestionFullText",
            "correct": "IsCorrect"
        }
        self.limit_nrows = limit_nrows

    def load_data(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_path, "answer.csv")
        df = pd.read_csv(file_path, nrows=self.limit_nrows)  # Load all rows in practice
        df = df[df.AnswerType == "Checkin"]
        df = df.sort_values(['UserId', 'DateAnswered'])
        return df

    def preprocess(self, data):
        dt_q = pd.read_csv(os.path.join(self.data_path, "question.csv"))
        dt_q['QuestionFullText'] = dt_q.apply(self.format_question, axis=1)
        
        dt_q_cons = pd.read_csv(os.path.join(self.data_path, "question-construct.csv"))
        dt_q = dt_q.merge(dt_q_cons[["ConstructId", "ConstructName"]].drop_duplicates(), on='ConstructId', how='inner')
        
        # Merge with the answer data
        data = data.merge(dt_q, on='QuestionId', how='inner')
        
        # Process data into student trajectories
        data = data.sort_values(['UserId', 'DateAnswered'])
        grouped = data.groupby('UserId')
        trajectories = []
        user_ids = []
        for u_id, group in grouped:
            user_ids.append(u_id)
            trajectory = []
            for _, row in group.iterrows():
                entry = {}
                entry["correct"] = row[self.mapping["correct"]]
                for k in self.contains:
                    entry[k] = row[self.mapping[k]]
                trajectory.append(entry)
            if self.max_trajectory_length is not None:
                trajectory = trajectory[:self.max_trajectory_length]  # Limit trajectory length
            if len(trajectory) >= self.min_trajectory_length:  # Check min trajectory length
                trajectories.append(trajectory)
            if (self.max_n_trajectories is not None) and (len(trajectories) >= self.max_n_trajectories):
                break
        return trajectories, user_ids

    def split_data(self, data, random_state=42):
        from sklearn.model_selection import train_test_split
        return train_test_split(data, test_size=self.test_size, random_state=random_state)

    @staticmethod
    def format_question(row):
        txt = row['QuestionText'] + "; "
        txt += f"A: {row['AnswerAText']}; "
        txt += f"B: {row['AnswerBText']}; "
        txt += f"C: {row['AnswerCText']}; "
        txt += f"D: {row['AnswerDText']}."
        return txt

