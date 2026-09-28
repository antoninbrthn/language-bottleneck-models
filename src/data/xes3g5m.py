import os
import yaml
import pickle

import tqdm
import json
import pandas as pd
from typing import Optional, List, Dict, Tuple
from src.utils.config import (
    XES3G5M_TRAIN_PATH, XES3G5M_TEST_PATH,
    XES3G5M_QUESTIONS_PATH, XES3G5M_KC_MAP_PATH,
    XES3G5M_TRANSLATIONS_PATH, XES3G5M_PREPROCESSED
)
from src.data.dataset_loader import DatasetLoader

class XES3G5MLoader(DatasetLoader):
    """Loader for XES3G5M dataset supporting both Chinese and English text."""
    
    def __init__(
        self,
        use_translation: bool = True,
        use_test_window: bool = False,
        limit_nrows: Optional[int] = None,
        checkpoint_dir: Optional[str] = XES3G5M_PREPROCESSED,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.use_translation = use_translation
        self.use_test_window = use_test_window
        self.limit_nrows = limit_nrows
        self.checkpoint_dir = checkpoint_dir
        
        # Define what information this loader contains
        self.contains = ["question_id", "question_text", "construct_id", "construct_text"]
        self.mapping = {
            "question_id": "question_id",
            "question_text": "question_text",
            "construct_id": "construct_id",
            "construct_text": "construct_text",
            "correct": "correct"
        }
        
        # Load metadata
        self._load_metadata()

    def _get_cache_paths(self) -> Dict[str, str]:
        """Get paths for cached files."""
        if not self.checkpoint_dir:
            return {}
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return {
            'data': os.path.join(self.checkpoint_dir, 'data.pkl'),
            'info': os.path.join(self.checkpoint_dir, 'info.yaml')
        }

    def _save_cache(self, data: pd.DataFrame) -> None:
        """Save data and metadata to cache."""
        cache_paths = self._get_cache_paths()
        if not cache_paths:
            return

        # Save data
        with open(cache_paths['data'], 'wb') as f:
            pickle.dump(data, f)

        # Save info
        info = {
            'use_translation': self.use_translation,
            'use_test_window': self.use_test_window,
            'limit_nrows': self.limit_nrows,
            'min_trajectory_length': self.min_trajectory_length,
            'max_trajectory_length': self.max_trajectory_length,
            'max_n_trajectories': self.max_n_trajectories
        }
        with open(cache_paths['info'], 'w') as f:
            yaml.dump(info, f)

    def _load_cache(self) -> Optional[pd.DataFrame]:
        """Load data from cache if available and configuration matches."""
        cache_paths = self._get_cache_paths()
        if not cache_paths or not os.path.exists(cache_paths['data']):
            return None

        # Check if configuration matches
        with open(cache_paths['info'], 'r') as f:
            info = yaml.safe_load(f)

        # Verify configuration matches
        for key, value in info.items():
            if getattr(self, key) != value:
                return None

        # Load cached data
        with open(cache_paths['data'], 'rb') as f:
            return pickle.load(f)

    def _load_metadata(self):
        """Load and prepare question and KC metadata."""
        # Load question metadata (Chinese)
        with open(XES3G5M_QUESTIONS_PATH, 'r') as f:
            self.questions_meta = json.load(f)
            
        # Load KC mapping (Chinese)
        with open(XES3G5M_KC_MAP_PATH, 'r') as f:
            self.kc_meta = json.load(f)
            
        if self.use_translation:
            # Load English translations
            with open(XES3G5M_TRANSLATIONS_PATH, 'r') as f:
                self.translations = json.load(f)

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load and prepare the flat format dataset."""
        # Try to load from cache first
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        # Load main training data
        train_df = pd.read_csv(XES3G5M_TRAIN_PATH, nrows=self.limit_nrows)
        
        if self.use_test_window:
            # Optionally load and concatenate test data
            test_df = pd.read_csv(XES3G5M_TEST_PATH, nrows=self.limit_nrows)
            df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            df = train_df

        # Convert to flat format
        flat_data = []
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            # Limit to where row['selectmasks'] is 1 and not -1 (list of n masks)
                        

            questions = row['questions'].split(',')
            responses = row['responses'].split(',')
            concepts = row['concepts'].split(',')
            timestamps = row['timestamps'].split(',')
            masks = row['selectmasks'].split(',')
            
            # for q, r, c, t in zip(questions, responses, concepts, timestamps):
            for q, r, c, t, m in zip(questions, responses, concepts, timestamps, masks):
                # Skip if mask is not 1
                if m != '1':
                    continue
                entry = {
                    'user_id': row['uid'],
                    'question_id': q,
                    'correct': int(r),
                    'construct_id': c,
                    'timestamp': int(t),
                    'fold': row['fold']
                }
                
                # Add text based on language preference
                if self.use_translation:
                    q_trans = self.translations.get(q, {})
                    entry['question_text'] = q_trans['question']
                    entry['construct_text'] = q_trans['knowledge_concepts_text'].replace('\n', ';')
                else:
                    entry['question_text'] = self.questions_meta.get(q, {}).get('content', '')
                    entry['construct_text'] = self.kc_meta.get(c, '')
                
                flat_data.append(entry)
        
        result_df = pd.DataFrame(flat_data)
        
        # Cache the processed data
        self._save_cache(result_df)
        
        return result_df

    def preprocess(self, data: pd.DataFrame) -> Tuple[List[List[Dict]], List]:
        """Process data into student trajectories and return with user IDs."""
        # Sort by user and timestamp
        data = data.sort_values(['user_id', 'timestamp'])
        
        trajectories = []
        user_ids = []
        
        # Group by user_id and create trajectories
        for user_id, group in tqdm.tqdm(data.groupby('user_id'), desc="Creating student trajectories...", total=data['user_id'].nunique()):
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
                user_ids.append(user_id)
                
            if self.max_n_trajectories and len(trajectories) >= self.max_n_trajectories:
                break
                
        return trajectories, user_ids

    def split_data(self, data, random_state=42):
        """Split trajectories into train and test sets"""
        from sklearn.model_selection import train_test_split
        return train_test_split(data, test_size=self.test_size, random_state=random_state)

    # def split_data(self, data: pd.DataFrame, random_state: int = 42):
    #     """Split data using fold information."""
    #     train = data[data['fold'] != 4]
    #     test = data[data['fold'] == 4]
    #     return train, test

def example_usage():
    """Example usage of XES3G5MLoader."""
    # Initialize loader with English translations
    loader = XES3G5MLoader(
        use_translation=True,
        use_test_window=False,
        limit_nrows=100,
        min_trajectory_length=5,
        max_trajectory_length=30,
        checkpoint_dir=XES3G5M_PREPROCESSED  # Enable caching
    )
    
    # First load will process and cache
    data = loader.load_data()
    print("Data loaded and cached")
    
    # Second load will use cache
    data = loader.load_data()
    print("Data loaded from cache")
    
    trajectories, user_ids = loader.preprocess(data)
    
    print(f"Loaded {len(trajectories)} trajectories")
    print("\nExample trajectory:")
    print(trajectories[0][:2])  # First two interactions of first trajectory

    print('Unique students in dataset:', data['user_id'].nunique())
    print('Nb of interactions dataset:', data.shape[0])
    print('Unique students:', len(set(user_ids)))

    # split data
    train_trajectories, test_trajectories = loader.split_data(trajectories)


if __name__ == "__main__":
    example_usage()
