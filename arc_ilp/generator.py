"""
Hypothesis enumeration using iterative deepening.

Generates all type-valid hypotheses up to a maximum depth.
"""

from typing import Dict, Iterator, List, Optional, Set
from itertools import product

from .types import BaseType, CallableType, Type, is_type_compatible
from .primitives import (
    PrimitiveInfo, HIGHER_ORDER_RETURNING_CALLABLE,
    FUNCTION_TAKING_PRIMITIVES
)
from .hypothesis import (
    Argument, PrimitiveCall, CallVariable, Hypothesis,
    TypeEnv, build_type_env, next_var_name
)


# Constants registry - map constant names to their types
CONSTANTS = {
    # Booleans
    'T': BaseType.BOOLEAN,
    'F': BaseType.BOOLEAN,

    # Integers
    'ZERO': BaseType.INTEGER,
    'ONE': BaseType.INTEGER,
    'TWO': BaseType.INTEGER,
    'THREE': BaseType.INTEGER,
    'FOUR': BaseType.INTEGER,
    'FIVE': BaseType.INTEGER,
    'SIX': BaseType.INTEGER,
    'SEVEN': BaseType.INTEGER,
    'EIGHT': BaseType.INTEGER,
    'NINE': BaseType.INTEGER,
    'TEN': BaseType.INTEGER,
    'NEG_ONE': BaseType.INTEGER,
    'NEG_TWO': BaseType.INTEGER,

    # IntegerTuples (direction vectors)
    'DOWN': BaseType.INTEGER_TUPLE,
    'RIGHT': BaseType.INTEGER_TUPLE,
    'UP': BaseType.INTEGER_TUPLE,
    'LEFT': BaseType.INTEGER_TUPLE,
    'ORIGIN': BaseType.INTEGER_TUPLE,
    'UNITY': BaseType.INTEGER_TUPLE,
    'NEG_UNITY': BaseType.INTEGER_TUPLE,
    'UP_RIGHT': BaseType.INTEGER_TUPLE,
    'DOWN_LEFT': BaseType.INTEGER_TUPLE,
    'ZERO_BY_TWO': BaseType.INTEGER_TUPLE,
    'TWO_BY_ZERO': BaseType.INTEGER_TUPLE,
    'TWO_BY_TWO': BaseType.INTEGER_TUPLE,
    'THREE_BY_THREE': BaseType.INTEGER_TUPLE,
}


def get_possible_arguments(
    param_type: Type,
    env: TypeEnv,
    primitives: Dict[str, PrimitiveInfo],
    include_primitives: bool = False
) -> List[Argument]:
    """
    Get all possible arguments for a parameter of given type.

    Args:
        param_type: The expected type of the parameter
        env: Current type environment with available variables
        primitives: Dictionary of primitive info
        include_primitives: Whether to include primitive names (for Callable params)

    Returns:
        List of possible Arguments
    """
    args = []

    # Input I (always Grid type)
    if is_type_compatible(BaseType.GRID, param_type):
        args.append(Argument(kind='input', value='I'))

    # Constants
    for const_name, const_type in CONSTANTS.items():
        if is_type_compatible(const_type, param_type):
            args.append(Argument(kind='constant', value=const_name))

    # Variables from environment
    for var_name, var_type in env.variables.items():
        if var_name == 'I':
            continue  # Already handled above
        if is_type_compatible(var_type, param_type):
            args.append(Argument(kind='variable', value=var_name))

    # Primitives as arguments (for higher-order functions)
    if include_primitives or isinstance(param_type, CallableType):
        for prim_name, prim_info in primitives.items():
            # Create a callable type for this primitive
            prim_callable = CallableType(
                input_types=tuple(prim_info.param_types),
                output_type=prim_info.return_type
            )
            if is_type_compatible(prim_callable, param_type):
                args.append(Argument(kind='primitive', value=prim_name))

    return args


def enumerate_bindings(
    prim_info: PrimitiveInfo,
    env: TypeEnv,
    primitives: Dict[str, PrimitiveInfo]
) -> Iterator[tuple]:
    """
    Enumerate all type-valid argument bindings for a primitive.

    Args:
        prim_info: Information about the primitive
        env: Current type environment
        primitives: Dictionary of primitive info

    Yields:
        Tuples of Arguments representing valid bindings
    """
    if len(prim_info.param_types) == 0:
        # Nullary function
        yield ()
        return

    # Get possible arguments for each parameter
    param_options = []
    for i, param_type in enumerate(prim_info.param_types):
        # For higher-order primitives, allow primitives as Callable arguments
        include_prims = (prim_info.name in FUNCTION_TAKING_PRIMITIVES and
                        isinstance(param_type, CallableType))
        options = get_possible_arguments(param_type, env, primitives, include_prims)
        if not options:
            # No valid arguments for this parameter - skip this primitive
            return
        param_options.append(options)

    # Generate all combinations
    for binding in product(*param_options):
        yield tuple(binding)


def enumerate_call_bindings(
    callable_type: CallableType,
    env: TypeEnv,
    primitives: Dict[str, PrimitiveInfo]
) -> Iterator[tuple]:
    """
    Enumerate argument bindings for calling a callable variable.

    Args:
        callable_type: The type of the callable
        env: Current type environment
        primitives: Dictionary of primitive info

    Yields:
        Tuples of Arguments for calling the callable
    """
    if len(callable_type.input_types) == 0:
        yield ()
        return

    param_options = []
    for param_type in callable_type.input_types:
        options = get_possible_arguments(param_type, env, primitives, False)
        if not options:
            return
        param_options.append(options)

    for binding in product(*param_options):
        yield tuple(binding)


def generate_single_step(
    primitives: Dict[str, PrimitiveInfo]
) -> Iterator[Hypothesis]:
    """
    Generate all single-step hypotheses.

    A single step must take Grid (I) and return Grid.
    """
    env = TypeEnv()  # Only has 'I': Grid

    for prim_name, prim_info in primitives.items():
        # Final output must be Grid
        if not is_type_compatible(prim_info.return_type, BaseType.GRID):
            continue

        for binding in enumerate_bindings(prim_info, env, primitives):
            step = PrimitiveCall(
                primitive=prim_name,
                arguments=binding,
                output_var='O'
            )
            yield Hypothesis(steps=(step,))


def extend_hypothesis(
    partial: Hypothesis,
    primitives: Dict[str, PrimitiveInfo],
    is_final: bool = False
) -> Iterator[Hypothesis]:
    """
    Add one more step to a partial hypothesis.

    Args:
        partial: The hypothesis to extend
        primitives: Dictionary of primitive info
        is_final: If True, only generate steps that output Grid (for final step)

    Yields:
        Extended hypotheses
    """
    env = build_type_env(partial, primitives)
    new_var = 'O' if is_final else next_var_name(partial)

    # Try adding a primitive call
    for prim_name, prim_info in primitives.items():
        # If final step, must return Grid
        if is_final and not is_type_compatible(prim_info.return_type, BaseType.GRID):
            continue

        for binding in enumerate_bindings(prim_info, env, primitives):
            new_step = PrimitiveCall(
                primitive=prim_name,
                arguments=binding,
                output_var=new_var
            )
            yield Hypothesis(steps=partial.steps + (new_step,))

    # Also try calling function-typed variables
    for var_name in env.get_callable_vars():
        var_type = env.get(var_name)
        if isinstance(var_type, CallableType):
            # If final step, must return Grid
            if is_final and not is_type_compatible(var_type.output_type, BaseType.GRID):
                continue

            for binding in enumerate_call_bindings(var_type, env, primitives):
                new_step = CallVariable(
                    callee=var_name,
                    arguments=binding,
                    output_var=new_var
                )
                yield Hypothesis(steps=partial.steps + (new_step,))


def generate_at_depth(
    depth: int,
    primitives: Dict[str, PrimitiveInfo]
) -> Iterator[Hypothesis]:
    """
    Generate all hypotheses with exactly `depth` steps.

    Uses iterative generation: builds from depth-1 hypotheses
    and adds the final step.
    """
    if depth == 1:
        yield from generate_single_step(primitives)
        return

    # For depth > 1, we need to generate partial hypotheses of depth-1
    # and then extend them with a final step
    if depth == 2:
        # Extend single-step partials
        for prim_name, prim_info in primitives.items():
            env = TypeEnv()  # Start with only I
            for binding in enumerate_bindings(prim_info, env, primitives):
                partial_step = PrimitiveCall(
                    primitive=prim_name,
                    arguments=binding,
                    output_var='x1'
                )
                partial = Hypothesis(steps=(partial_step,))
                yield from extend_hypothesis(partial, primitives, is_final=True)
    else:
        # For depth > 2, recursively generate depth-1 partials
        for partial in generate_partials_at_depth(depth - 1, primitives):
            yield from extend_hypothesis(partial, primitives, is_final=True)


def generate_partials_at_depth(
    depth: int,
    primitives: Dict[str, PrimitiveInfo]
) -> Iterator[Hypothesis]:
    """
    Generate partial hypotheses (not necessarily ending in Grid output).

    These are intermediate hypotheses used to build longer ones.
    """
    if depth == 1:
        # Single step partial - can output any type
        env = TypeEnv()
        for prim_name, prim_info in primitives.items():
            for binding in enumerate_bindings(prim_info, env, primitives):
                step = PrimitiveCall(
                    primitive=prim_name,
                    arguments=binding,
                    output_var='x1'
                )
                yield Hypothesis(steps=(step,))

        # Also include calling I with callable-returning primitives
        # (not applicable since we don't have callable vars yet at depth 1)
        return

    # For depth > 1, extend depth-1 partials
    for partial in generate_partials_at_depth(depth - 1, primitives):
        yield from extend_hypothesis(partial, primitives, is_final=False)


def generate_hypotheses(
    max_depth: int,
    primitives: Dict[str, PrimitiveInfo]
) -> Iterator[Hypothesis]:
    """
    Generate all type-valid hypotheses up to max_depth steps.

    Uses iterative deepening: generates all depth-1 hypotheses,
    then all depth-2, etc.

    Args:
        max_depth: Maximum number of steps
        primitives: Dictionary of primitive info

    Yields:
        Valid hypotheses (final output is Grid)
    """
    for depth in range(1, max_depth + 1):
        yield from generate_at_depth(depth, primitives)
