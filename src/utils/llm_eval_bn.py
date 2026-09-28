"""Utilities for evaluating bottleneck quality using LLM-based annotation."""

from src.prompts.prompt_utils import load_prompt_template


def make_user_prompt(bottleneck: str, ground_truths_miscs: list[str], constructs_mastery: dict[str, int]) -> str:
    """Build an LLM evaluation prompt for a given bottleneck and ground-truth knowledge state.

    Args:
        bottleneck: The model-generated bottleneck text to evaluate.
        ground_truths_miscs: List of ground-truth misconception descriptions.
        constructs_mastery: Dict mapping construct name to mastery score (0 or 1).
    Returns:
        The formatted user prompt string.
    """
    user_prompt_template = load_prompt_template(filename="bottleneck_annotation.yaml", key="user_prompt_template")
    constructs_text = "\n".join(
        f"{construct}: {'mastered' if score == 1 else 'not mastered'}"
        for construct, score in constructs_mastery.items()
    )
    misconceptions_text = "\n".join(
        f"Misc {index}: {misc}" for index, misc in enumerate(ground_truths_miscs)
    ) or "None"
    return user_prompt_template.format(
        bottleneck=bottleneck,
        constructs=constructs_text,
        misconceptions=misconceptions_text,
    )
