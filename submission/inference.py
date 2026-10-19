"""Competition interface for FNR (Factor Neighborhood Ranker)."""
if __package__:
    from .fnr.inference.predictor import InferenceModel
else:
    from fnr.inference.predictor import InferenceModel

__all__ = ["InferenceModel"]
