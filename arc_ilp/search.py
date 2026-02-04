"""
Main search loop for finding hypotheses.

Implements iterative deepening search with optional timeout.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator

from .types import BaseType
from .primitives import extract_primitives, PrimitiveInfo
from .hypothesis import Hypothesis
from .generator import generate_at_depth, CONSTANTS
from .evaluator import evaluate_hypothesis
from .codegen import hypothesis_to_code


@dataclass
class SearchResult:
    """Result of a successful search"""
    hypothesis: Hypothesis
    code: str  # solvers.py format
    depth: int
    hypotheses_tried: int
    time_elapsed: float


def search(
    task_examples: List[Tuple[tuple, tuple]],
    task_id: str,
    dsl_module,
    constants_module,
    max_depth: Optional[int] = None,
    timeout: Optional[float] = None,
    verbose: bool = False
) -> Optional[SearchResult]:
    """
    Search for a hypothesis that solves all examples.

    Uses iterative deepening: tries all depth-1 hypotheses,
    then all depth-2, etc.

    Args:
        task_examples: List of (input_grid, expected_output) tuples
        task_id: Task ID for code generation
        dsl_module: The DSL module
        constants_module: The constants module
        max_depth: Maximum depth to search (None = no limit)
        timeout: Maximum time in seconds (None = no limit)
        verbose: Print progress information

    Returns:
        SearchResult if found, None otherwise

    Combinatorics (rough estimate for depth d):
    - ~130 primitives, ~30 constants, d variables
    - Branching factor per step: O(130 * (30 + d)^avg_arity)
    - Total hypotheses at depth d: O(branching_factor^d)

    Practical limits:
    - Depth 1-3: Fast (seconds)
    - Depth 4-5: Slow (minutes)
    - Depth 6+: Very slow (hours/days)
    """
    primitives = extract_primitives(dsl_module)

    start_time = time.time()
    total_tried = 0
    depth = 1

    while max_depth is None or depth <= max_depth:
        if verbose:
            print(f"Searching depth {depth}...")

        depth_tried = 0
        for hypothesis in generate_at_depth(depth, primitives):
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                if verbose:
                    print(f"Timeout after {total_tried + depth_tried} hypotheses")
                return None

            depth_tried += 1
            total_tried += 1

            if evaluate_hypothesis(hypothesis, task_examples, dsl_module, constants_module):
                elapsed = time.time() - start_time
                code = hypothesis_to_code(hypothesis, task_id)
                if verbose:
                    print(f"Found solution at depth {depth} after {total_tried} hypotheses ({elapsed:.2f}s)")
                return SearchResult(
                    hypothesis=hypothesis,
                    code=code,
                    depth=depth,
                    hypotheses_tried=total_tried,
                    time_elapsed=elapsed
                )

        if verbose:
            print(f"  Depth {depth}: {depth_tried} hypotheses tried")

        depth += 1

    if verbose:
        print(f"No solution found after {total_tried} hypotheses")
    return None


def search_with_callback(
    task_examples: List[Tuple[tuple, tuple]],
    task_id: str,
    dsl_module,
    constants_module,
    callback,
    max_depth: Optional[int] = None,
    timeout: Optional[float] = None
) -> Optional[SearchResult]:
    """
    Search with a callback for progress updates.

    Args:
        task_examples: List of (input_grid, expected_output) tuples
        task_id: Task ID for code generation
        dsl_module: The DSL module
        constants_module: The constants module
        callback: Function called with (depth, count, elapsed) periodically
        max_depth: Maximum depth to search
        timeout: Maximum time in seconds

    Returns:
        SearchResult if found, None otherwise
    """
    primitives = extract_primitives(dsl_module)

    start_time = time.time()
    total_tried = 0
    depth = 1
    last_callback = start_time

    while max_depth is None or depth <= max_depth:
        for hypothesis in generate_at_depth(depth, primitives):
            current_time = time.time()

            # Check timeout
            if timeout and (current_time - start_time) > timeout:
                return None

            # Call callback periodically (every 0.5 seconds)
            if current_time - last_callback > 0.5:
                callback(depth, total_tried, current_time - start_time)
                last_callback = current_time

            total_tried += 1

            if evaluate_hypothesis(hypothesis, task_examples, dsl_module, constants_module):
                elapsed = current_time - start_time
                code = hypothesis_to_code(hypothesis, task_id)
                return SearchResult(
                    hypothesis=hypothesis,
                    code=code,
                    depth=depth,
                    hypotheses_tried=total_tried,
                    time_elapsed=elapsed
                )

        depth += 1

    return None


def enumerate_solutions(
    task_examples: List[Tuple[tuple, tuple]],
    task_id: str,
    dsl_module,
    constants_module,
    max_depth: int = 5,
    max_solutions: int = 10
) -> Iterator[SearchResult]:
    """
    Enumerate multiple solutions for a task.

    Useful for finding alternative implementations.

    Args:
        task_examples: List of (input_grid, expected_output) tuples
        task_id: Task ID for code generation
        dsl_module: The DSL module
        constants_module: The constants module
        max_depth: Maximum depth to search
        max_solutions: Maximum number of solutions to return

    Yields:
        SearchResult for each solution found
    """
    primitives = extract_primitives(dsl_module)
    solutions_found = 0
    total_tried = 0
    start_time = time.time()

    for depth in range(1, max_depth + 1):
        for hypothesis in generate_at_depth(depth, primitives):
            total_tried += 1

            if evaluate_hypothesis(hypothesis, task_examples, dsl_module, constants_module):
                elapsed = time.time() - start_time
                code = hypothesis_to_code(hypothesis, task_id)
                yield SearchResult(
                    hypothesis=hypothesis,
                    code=code,
                    depth=depth,
                    hypotheses_tried=total_tried,
                    time_elapsed=elapsed
                )
                solutions_found += 1
                if solutions_found >= max_solutions:
                    return
