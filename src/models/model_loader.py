"""Unified model loader: dispatches to API, HuggingFace, or LBM backends."""


def infer_model_type(model_config):
    """Infer model backend from the model name."""
    if "gpt" in model_config.name.lower():
        return "api"
    return "hf"


def load_any_model(model_config):
    """Instantiate a model from a Hydra/OmegaConf config dict.

    Supported ``type`` values: ``"api"`` (Azure OpenAI), ``"hf"`` (HuggingFace),
    ``"lbm"`` (Language Bottleneck Model with separate encoder/decoder).
    If ``type`` is omitted it is inferred from the model name.
    """
    model_type = model_config.get("type", None)
    if model_type is None:
        model_type = infer_model_type(model_config)

    if model_type == "api":
        from src.models.api_models import load_api_model
        return load_api_model(model_config)
    elif model_type == "hf":
        from src.models.hf_models import load_hf_model
        return load_hf_model(model_config)
    elif model_type == "lbm":
        from src.utils.lb_models import load_lb_model
        encoder_config, decoder_config = model_config.encoder, model_config.decoder
        return load_lb_model(encoder_config, decoder_config, **model_config)
    else:
        raise ValueError(f"Invalid model type '{model_type}'. Choose 'hf', 'api', or 'lbm'.")