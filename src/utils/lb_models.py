"""Factory function for instantiating Language Bottleneck Models."""

from src.models.lb_models.utils import check_is_ground_truth
from src.models.lb_models.lb_base import LBModel
from src.models.lb_models.lb_gt import LBModelGT
from src.models.lb_models.lb_steer import LBModelSteer


def load_lb_model(
    encoder_config,
    decoder_config,
    **model_config,
):
    """Instantiate an LBModel variant based on the config.

    Supported modes (via ``model_config["mode"]``):
      - ``"standard"`` (default) → :class:`LBModel`
      - ``"steer"`` → :class:`LBModelSteer`
      - ground-truth encoder name → :class:`LBModelGT`
    """
    if check_is_ground_truth(model_config):
        return LBModelGT(
            decoder_config=decoder_config,
            **model_config,
        )
    if model_config.get("mode", "standard") == "steer":
        return LBModelSteer(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            **model_config,
        )
    return LBModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        **model_config,
    )

