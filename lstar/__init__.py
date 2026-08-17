"""
L-STAR: LLM-guided Spatial Transcriptomics Analysis and Ranking

A pipeline for LLM-based model comparison, consensus clustering, and biological
domain annotation for spatial transcriptomics data.
"""

from lstar.allwise import run_allwise_comparisons
from lstar.pairwise import run_pairwise_comparisons
from lstar.consensus import run_consensus_clustering
from lstar.domain_annotation import annotate_domain
from lstar.pipeline import l_star

__version__ = "0.8.0"
__all__ = [
    "run_pairwise_comparisons",
    "run_allwise_comparisons",
    "run_consensus_clustering",
    "l_star",
    "annotate_domain",
]
