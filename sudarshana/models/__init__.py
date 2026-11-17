"""Model architectures and heads for Sudarshana."""

# Lazy imports — only load when torch is actually available.
# This lets data-pipeline and augmentation code import safely without torch.
try:
    from .token_head import SudarshanaTokenClassifier, TokenClassificationHead
    from .joint_model import SudarshanaJointModel

    __all__ = [
        "SudarshanaTokenClassifier",
        "TokenClassificationHead",
        "SudarshanaJointModel",
    ]
except (ImportError, AttributeError):  # pragma: no cover
    # torch / transformers not installed — model classes unavailable.
    __all__ = []
