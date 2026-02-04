"""
Hypothesis evaluation against examples.

Tests hypotheses by executing them on input grids and comparing
to expected outputs.
"""

from typing import List, Tuple, Any, Optional

from .hypothesis import Hypothesis, PrimitiveCall, CallVariable, Argument


def resolve_arg(
    arg: Argument,
    env: dict,
    dsl_module,
    constants_module
) -> Any:
    """
    Resolve an argument to its actual value.

    Args:
        arg: The argument to resolve
        env: Runtime environment (variable bindings)
        dsl_module: The DSL module
        constants_module: The constants module

    Returns:
        The resolved value
    """
    if arg.kind == 'constant':
        return getattr(constants_module, arg.value)
    elif arg.kind == 'variable':
        return env[arg.value]
    elif arg.kind == 'input':
        return env['I']
    elif arg.kind == 'primitive':
        return getattr(dsl_module, arg.value)
    else:
        raise ValueError(f"Unknown argument kind: {arg.kind}")


def execute_hypothesis(
    hypothesis: Hypothesis,
    input_grid: tuple,
    dsl_module,
    constants_module
) -> Any:
    """
    Execute a hypothesis on an input to produce output.

    Args:
        hypothesis: The hypothesis to execute
        input_grid: The input grid
        dsl_module: The DSL module
        constants_module: The constants module

    Returns:
        The output (should be a Grid for valid hypotheses)

    Raises:
        Exception: If execution fails
    """
    env = {'I': input_grid}

    for step in hypothesis.steps:
        if isinstance(step, PrimitiveCall):
            func = getattr(dsl_module, step.primitive)
            args = [resolve_arg(arg, env, dsl_module, constants_module)
                    for arg in step.arguments]
            result = func(*args)
            env[step.output_var] = result
        elif isinstance(step, CallVariable):
            func = env[step.callee]
            args = [resolve_arg(arg, env, dsl_module, constants_module)
                    for arg in step.arguments]
            result = func(*args)
            env[step.output_var] = result
        else:
            raise ValueError(f"Unknown step type: {type(step)}")

    return env['O']


def evaluate_hypothesis(
    hypothesis: Hypothesis,
    examples: List[Tuple[tuple, tuple]],
    dsl_module,
    constants_module
) -> bool:
    """
    Test if hypothesis solves all examples.

    Args:
        hypothesis: The hypothesis to test
        examples: List of (input_grid, expected_output) tuples
        dsl_module: The DSL module
        constants_module: The constants module

    Returns:
        True iff all examples produce correct output
    """
    for input_grid, expected_output in examples:
        try:
            output = execute_hypothesis(
                hypothesis, input_grid, dsl_module, constants_module
            )
            if output != expected_output:
                return False
        except Exception:
            return False
    return True


def evaluate_hypothesis_partial(
    hypothesis: Hypothesis,
    examples: List[Tuple[tuple, tuple]],
    dsl_module,
    constants_module
) -> Tuple[int, int]:
    """
    Evaluate hypothesis and return (correct_count, total_count).

    Useful for debugging or partial scoring.

    Args:
        hypothesis: The hypothesis to test
        examples: List of (input_grid, expected_output) tuples
        dsl_module: The DSL module
        constants_module: The constants module

    Returns:
        Tuple of (number of correct examples, total examples)
    """
    correct = 0
    for input_grid, expected_output in examples:
        try:
            output = execute_hypothesis(
                hypothesis, input_grid, dsl_module, constants_module
            )
            if output == expected_output:
                correct += 1
        except Exception:
            pass
    return (correct, len(examples))
