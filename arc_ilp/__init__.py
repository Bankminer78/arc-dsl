"""
ILP System for ARC-DSL Program Synthesis

This module implements an Inductive Logic Programming system that searches
for ARC solutions by composing DSL primitives.
"""

from .types import BaseType, CallableType, Type
from .primitives import PrimitiveInfo, extract_primitives
from .hypothesis import Argument, PrimitiveCall, CallVariable, Hypothesis
from .generator import generate_hypotheses, generate_at_depth
from .evaluator import evaluate_hypothesis, execute_hypothesis
from .codegen import hypothesis_to_code
from .search import search, SearchResult

__all__ = [
    'BaseType',
    'CallableType',
    'Type',
    'PrimitiveInfo',
    'extract_primitives',
    'Argument',
    'PrimitiveCall',
    'CallVariable',
    'Hypothesis',
    'generate_hypotheses',
    'generate_at_depth',
    'evaluate_hypothesis',
    'execute_hypothesis',
    'hypothesis_to_code',
    'search',
    'SearchResult',
]
