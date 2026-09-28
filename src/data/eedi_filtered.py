import os
import random
import pandas as pd
from typing import Optional

from src.utils.config import DATA_DIR
from src.data.eedi import EediLoader  # assumes EediLoader is defined in the same package


class EediFilteredLoader(EediLoader):
    """Extended :class:`EediLoader` that filters by response–time, groups
    contiguous *sessions*, optionally shuffles questions within each
    trajectory, and discards short sessions.

    Parameters
    ----------
    data_path : str | None, default ``None``
        Folder holding the raw *Eedi* CSV files. Defaults to
        ``DATA_DIR / "eedi_private/data/"``.
    limit_nrows : int | None, default ``None``
        Optional row cap when prototyping.
    min_answer_seconds : int, default ``5``
        Minimum answer latency to retain (seconds).
    session_gap_minutes : int, default ``15``
        Maximum allowed gap between two consecutive answers before a new
        session starts (minutes).
    min_session_questions : int, default ``40``
        Minimum questions required in a *(user, session)* group.
    shuffle_seed : int | None, default ``None``
        Seed for deterministic shuffling of questions within each
        trajectory. When ``None``, no shuffling is applied.

    All other keyword arguments are forwarded to :class:`DatasetLoader`.
    """

    # Default constants exposed as class attributes  ----------------------
    DEFAULT_MIN_ANSWER_SECONDS: int = 5
    DEFAULT_SESSION_GAP_MINUTES: int = 15
    DEFAULT_MIN_SESSION_QUESTIONS: int = 40

    def __init__(
        self,
        data_path: Optional[str] = None,
        limit_nrows: Optional[int] = None,
        *,
        min_answer_seconds: int = DEFAULT_MIN_ANSWER_SECONDS,
        session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
        min_session_questions: int = DEFAULT_MIN_SESSION_QUESTIONS,
        shuffle_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(data_path=data_path, limit_nrows=limit_nrows, **kwargs)

        # Exposed numeric attributes ("{N} numbers")
        self.min_answer_seconds: int = int(min_answer_seconds)
        self.session_gap_minutes: int = int(session_gap_minutes)
        self.min_session_questions: int = int(min_session_questions)

        # Shuffle settings -------------------------------------------------
        self.shuffle_seed = shuffle_seed
        # Dedicated RNG instance for deterministic but isolated shuffling
        self._rng = (
            random.Random(shuffle_seed) if shuffle_seed is not None else None
        )

    # ------------------------------------------------------------------
    # Data loading / filtering
    # ------------------------------------------------------------------
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        if file_path is None:
            file_path = os.path.join(self.data_path, "answer.csv")

        df = pd.read_csv(
            file_path,
            nrows=self.limit_nrows,
            parse_dates=["DateAnswered"],
        )

        # --- Basic filters ----------------------------------------------
        df = df[(df.AnswerType == "Checkin") & (df.SecondsToAnswer >= self.min_answer_seconds)]
        df = df.sort_values(["UserId", "DateAnswered"], ignore_index=True)

        # --- Session identification -------------------------------------
        prev_date = df.groupby("UserId")["DateAnswered"].shift(1)
        prev_secs = df.groupby("UserId")["SecondsToAnswer"].shift(1).fillna(0)
        expected_current = prev_date + pd.to_timedelta(prev_secs, unit="s")
        gap_seconds = (df["DateAnswered"] - expected_current).dt.total_seconds().fillna(float("inf"))

        df["new_session"] = ((prev_date.isna()) | (gap_seconds > self.session_gap_minutes * 60)).astype(int)
        df["session_id"] = df.groupby("UserId")["new_session"].cumsum()

        # --- Session length filter --------------------------------------
        session_sizes = df.groupby(["UserId", "session_id"]).size()
        valid_sessions = session_sizes[session_sizes >= self.min_session_questions].index
        mask = df.set_index(["UserId", "session_id"]).index.isin(valid_sessions)
        df = df.loc[mask].reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Pre‑processing / trajectory building
    # ------------------------------------------------------------------
    def preprocess(self, data: pd.DataFrame):
        # --- Merge with question metadata -------------------------------
        question_df = pd.read_csv(os.path.join(self.data_path, "question.csv"))
        question_df["QuestionFullText"] = question_df.apply(self.format_question, axis=1)
        construct_df = pd.read_csv(os.path.join(self.data_path, "question-construct.csv"))
        question_df = question_df.merge(
            construct_df[["ConstructId", "ConstructName"]].drop_duplicates(),
            on="ConstructId",
            how="inner",
        )

        merged = data.merge(question_df, on="QuestionId", how="inner")
        merged = merged.sort_values(["UserId", "session_id", "DateAnswered"], ignore_index=True)

        # --- Build trajectories -----------------------------------------
        trajectories: list[list[dict]] = []
        identifiers: list[tuple[int, int]] = []

        grouped = merged.groupby(["UserId", "session_id"], sort=False)
        for (u_id, s_id), grp in grouped:
            trajectory = [
                {
                    **{k: row[self.mapping[k]] for k in self.contains},
                    "correct": row[self.mapping["correct"]],
                }
                for _, row in grp.iterrows()
            ]

            # Optional deterministic shuffle -----------------------------
            if self._rng is not None:
                self._rng.shuffle(trajectory)

            # Enforce trajectory length / count constraints --------------
            if self.max_trajectory_length is not None:
                trajectory = trajectory[: self.max_trajectory_length]

            if len(trajectory) >= self.min_trajectory_length:
                trajectories.append(trajectory)
                identifiers.append((u_id, s_id))

            if self.max_n_trajectories is not None and len(trajectories) >= self.max_n_trajectories:
                break

        return trajectories, identifiers


def example_usage():
    """Example usage of the EediFilteredLoader class."""
    loader = EediFilteredLoader(
        data_path=os.path.join(DATA_DIR, "eedi_private/data/"),
        limit_nrows=None,
        min_answer_seconds=5,
        session_gap_minutes=15,
        min_session_questions=65,
    )
    
    # Load and preprocess the data
    data = loader.load_data()
    trajectories, identifiers = loader.preprocess(data)
    len(trajectories), len(identifiers)

    # Print the first trajectory and its identifier
    print("First trajectory:", trajectories[0])
    print("Identifier:", identifiers[0])