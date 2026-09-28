##############################################################################
# Convenience function for building an LBModel in one call
##############################################################################


def check_is_ground_truth(model_config):
    """
    Check if the model is using ground truth data.
    """
    if "ground-truth" in model_config['encoder']['name']:
        return True
    return False

def add_instructions_to_xprompt(x_prompt, use_cot, max_bottleneck_tokens=512, template=None):
    """
    Create a prompt for the encoder model using a template.
    
    Args:
        x_prompt: The input text containing student answers history
        use_cot: Whether to use chain-of-thought reasoning
        max_bottleneck_tokens: Maximum tokens for the bottleneck
        template: Optional template string to override the default
        
    Returns:
        The formatted encoder prompt
    """
    from src.prompts.prompt_utils import load_prompt_template
    
    # Load default template if none provided
    if template is None:
        template = load_prompt_template(filename="base_x.yaml")
    
    # Prepare CoT instruction
    if use_cot:
        cot_instruction = "Think step by step and lay out your thinking process carefully. Then, provide your summary in <info>...</info>."
    else:
        cot_instruction = "Enclose your entire summary in <info> and </info> and do not include anything else."
    
    # Calculate approximate word limit (0.75 tokens per word rule of thumb)
    max_words = int(0.75 * max_bottleneck_tokens)
    
    # Format the template
    encoder_prompt = template.format(
        input_text=x_prompt,
        cot_instruction=cot_instruction,
        max_words=max_words
    )
    
    return encoder_prompt

def add_instructions_to_yprompt(y_prompt, bottleneck, template=None):
    """
    Create a prompt for the decoder model using a template.
    
    Args:
        y_prompt: The prompt for the new question
        bottleneck: The encoded knowledge state from the encoder
        template: Optional template string to override the default
        
    Returns:
        The formatted decoder prompt
    """
    from src.prompts.prompt_utils import load_prompt_template
    
    # Load default template if none provided
    if template is None:
        template = load_prompt_template(filename="base_y.yaml")
    
    # Format the template
    decoder_prompt = template.format(
        bottleneck=bottleneck,
        y_prompt=y_prompt
    )
    
    return decoder_prompt
