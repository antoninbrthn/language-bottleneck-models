from src.utils.config import XES3G5M_PREPROCESSED
import pandas as pd
from typing import Optional, Tuple, List, Dict

from src.data.xes3g5m import XES3G5MLoader

class XES3G5MFilteredLoader(XES3G5MLoader):
    """Extended XES3G5MLoader that filters by time gaps and groups sessions."""
    
    DEFAULT_SESSION_GAP_MINUTES: int = 15
    DEFAULT_MIN_SESSION_QUESTIONS: int = 40
    
    def __init__(
        self,
        session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
        min_session_questions: int = DEFAULT_MIN_SESSION_QUESTIONS,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.session_gap_minutes = session_gap_minutes
        self.min_session_questions = min_session_questions

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load data and apply session-based filtering."""
        df = super().load_data(file_path)
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        print(f"Loaded {len(df)} rows before session filtering")

        # Identify sessions based on time gaps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        time_diff = df.groupby('user_id')['timestamp'].diff()
        df['new_session'] = ((time_diff.dt.total_seconds() / 60) > self.session_gap_minutes).astype(int)
        df['session_id'] = df.groupby('user_id')['new_session'].cumsum()

        # Filter sessions by length
        session_sizes = df.groupby(['user_id', 'session_id']).size()

        valid_sessions = session_sizes[session_sizes >= self.min_session_questions].index
        print(f"Valid sessions after filtering: {len(valid_sessions)}")

        mask = df.set_index(['user_id', 'session_id']).index.isin(valid_sessions)
        df = df.loc[mask].reset_index(drop=True)
        print(f"Filtered to {len(df)} rows after session filtering")
        # print unique uid and session id combos
        print(f"Unique user-session pairs: {df[['user_id', 'session_id']].drop_duplicates().shape[0]}")
        # average length per session
        avg_length = df.groupby(['user_id', 'session_id']).size().mean()
        print(f"Average session length: {avg_length:.2f} questions")
        
        return df

    def preprocess(self, data: pd.DataFrame) -> Tuple[List[List[Dict]], List[Tuple[int, int]]]:
        """Process filtered data into trajectories, returning with (user_id, session_id) pairs."""
        trajectories = []
        identifiers = []
        
        # Group by user and session
        grouped = data.groupby(['user_id', 'session_id'])
        
        for (user_id, session_id), group in grouped:
            trajectory = []
            
            for _, row in group.iterrows():
                entry = {
                    'correct': row['correct'],
                    'question_id': row['question_id'],
                    'question_text': row['question_text'],
                    'construct_id': row['construct_id'],
                    'construct_text': row['construct_text']
                }
                trajectory.append(entry)
            
            if self.max_trajectory_length is not None:
                trajectory = trajectory[:self.max_trajectory_length]
                
            if len(trajectory) >= self.min_trajectory_length:
                trajectories.append(trajectory)
                identifiers.append((user_id, session_id))
                
            if self.max_n_trajectories and len(trajectories) >= self.max_n_trajectories:
                break
                
        return trajectories, identifiers

def example_usage():
    """Example usage of XES3G5MFilteredLoader."""
    # Initialize filtered loader
    loader = XES3G5MFilteredLoader(
        use_translation=True,
        use_test_window=False,
        limit_nrows=None,
        session_gap_minutes=10,
        min_session_questions=34,
        min_trajectory_length=5,
        max_trajectory_length=34,
        checkpoint_dir=XES3G5M_PREPROCESSED  # Enable caching
    )

    data = loader.load_data()
    trajectories, identifiers = loader.preprocess(data)
    
    print(f"Loaded {len(trajectories)} filtered trajectories")
    print("\nExample trajectory with identifier:")
    print(f"User {identifiers[0][0]}, Session {identifiers[0][1]}:")
    print(trajectories[0][:2])  # First two interactions of first trajectory
    print(len(trajectories[0]))
    lens = [len(t) for t in trajectories]
    print('Average trajectory length:', sum(lens) / len(lens))

    # split data
    train_trajectories, test_trajectories = loader.split_data(trajectories)

if __name__ == "__main__":
    example_usage()