"""
Star Map Quality Analysis and Evaluation
=========================================
Provides quantitative metrics for evaluating the quality of simulated
SWIR star maps, including SNR, centroid accuracy, and noise statistics.
"""

from .evaluation_metrics import StarMapEvaluator

__all__ = ["StarMapEvaluator"]
