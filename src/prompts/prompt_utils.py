from pathlib import Path
import yaml
from typing import Dict, Union, Optional
from src.utils.config import PROMPTS_DIR


def load_prompt_template(
    prompt_dir: str = PROMPTS_DIR, filename: str = "iterative_encoder.yaml", key: str = "template", return_dict: bool = False
) -> Union[str, Dict]:
    """
    Load a prompt template from a YAML file.

    Args:
        prompt_dir: Directory containing prompt templates
        filename: Name of the YAML file
        key: Key in the YAML file to retrieve (default: 'template')
        return_dict: If True, return the entire parsed YAML as a dict

    Returns:
        Either the template string (if return_dict=False) or the entire parsed YAML
    """
    path = Path(prompt_dir) / filename
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if return_dict:
        return data
    return data[key]


def build_iterative_encoder_prompt(previous_bottleneck: str, qa_block: str, use_cot: bool, max_bottleneck_tokens: int, template: str) -> str:
    cot_instruction = (
        "Think step by step and lay out your reasoning before you write the " "final summary. Then, enclose the final summary in <info>…</info>."
        if use_cot
        else "Enclose your entire summary in <info>…</info> and do not include " "anything else."
    )

    max_len_instruction = f"Make sure that your final summary is at most {int(0.75 * max_bottleneck_tokens)} words." if max_bottleneck_tokens else ""
    prompt = template.format(
        previous_bottleneck=(f"<info>{previous_bottleneck}</info>" if previous_bottleneck else "<info></info>"),
        qa_block=qa_block,
        cot_instruction=cot_instruction,
        max_len_instruction=max_len_instruction,
    )
    return prompt
