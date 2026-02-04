"""
Type system for ARC-DSL primitives.

Defines a type representation that mirrors arc_types.py but is usable for type checking
during hypothesis generation.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Union, FrozenSet, Optional


class BaseType(Enum):
    """Base types in the ARC-DSL"""
    BOOLEAN = auto()
    INTEGER = auto()
    INTEGER_TUPLE = auto()
    GRID = auto()
    CELL = auto()
    OBJECT = auto()
    OBJECTS = auto()
    INDICES = auto()
    INDICES_SET = auto()
    INTEGER_SET = auto()
    TUPLE = auto()
    TUPLE_TUPLE = auto()
    CONTAINER = auto()
    CONTAINER_CONTAINER = auto()
    FROZENSET = auto()
    ANY = auto()  # For polymorphic functions


@dataclass(frozen=True)
class CallableType:
    """Represents a function type"""
    input_types: tuple  # Tuple of Type, can be empty for nullary
    output_type: 'Type'

    def __hash__(self):
        return hash((self.input_types, self.output_type))


# A Type is either a BaseType or a CallableType
Type = Union[BaseType, CallableType]


# Union type sets for polymorphism
NUMERICAL = frozenset({BaseType.INTEGER, BaseType.INTEGER_TUPLE})
PATCH = frozenset({BaseType.OBJECT, BaseType.INDICES})
ELEMENT = frozenset({BaseType.OBJECT, BaseType.GRID})
PIECE = frozenset({BaseType.GRID, BaseType.OBJECT, BaseType.INDICES})
CONTAINER_TYPES = frozenset({
    BaseType.OBJECTS, BaseType.INDICES, BaseType.INTEGER_SET,
    BaseType.TUPLE, BaseType.GRID, BaseType.OBJECT, BaseType.CONTAINER
})
FROZENSET_TYPES = frozenset({
    BaseType.OBJECT, BaseType.OBJECTS, BaseType.INDICES,
    BaseType.INDICES_SET, BaseType.INTEGER_SET, BaseType.FROZENSET
})


def is_type_compatible(arg_type: Type, param_type: Type) -> bool:
    """
    Check if an argument type is compatible with a parameter type.

    Args:
        arg_type: The type of the argument being passed
        param_type: The expected parameter type

    Returns:
        True if arg_type can be used where param_type is expected
    """
    # ANY matches everything
    if param_type == BaseType.ANY:
        return True
    if arg_type == BaseType.ANY:
        return True

    # Direct match
    if arg_type == param_type:
        return True

    # Numerical union type
    if param_type == BaseType.INTEGER_TUPLE and arg_type == BaseType.INTEGER:
        # Integer can be used where IntegerTuple expected in some contexts
        return False

    # Container compatibility
    if param_type == BaseType.CONTAINER:
        return arg_type in CONTAINER_TYPES or arg_type == BaseType.CONTAINER

    if param_type == BaseType.CONTAINER_CONTAINER:
        return arg_type in {BaseType.OBJECTS, BaseType.INDICES_SET,
                           BaseType.TUPLE_TUPLE, BaseType.CONTAINER_CONTAINER}

    # FrozenSet compatibility
    if param_type == BaseType.FROZENSET:
        return arg_type in FROZENSET_TYPES

    # Patch union type (Object or Indices)
    if param_type in PATCH:
        if arg_type in PATCH:
            return True

    # Element union type (Object or Grid)
    if param_type in ELEMENT:
        if arg_type in ELEMENT:
            return True

    # Piece union type (Grid, Object, or Indices)
    if param_type in PIECE:
        if arg_type in PIECE:
            return True

    # Tuple compatibility
    if param_type == BaseType.TUPLE:
        return arg_type in {BaseType.TUPLE, BaseType.GRID, BaseType.INTEGER_TUPLE,
                           BaseType.TUPLE_TUPLE}

    # Callable type matching
    if isinstance(param_type, CallableType) and isinstance(arg_type, CallableType):
        # For callable matching, we're lenient - just check arity
        # Full type checking would require more complex inference
        return len(param_type.input_types) == len(arg_type.input_types)

    # If param expects Callable and arg is Callable
    if isinstance(param_type, CallableType):
        return isinstance(arg_type, CallableType)

    return False


def get_type_union(types: FrozenSet[BaseType]) -> Optional[BaseType]:
    """
    Get a common type that represents a union of types.
    Returns None if no common type exists.
    """
    if len(types) == 0:
        return None
    if len(types) == 1:
        return next(iter(types))

    # Check for known unions
    if types <= NUMERICAL:
        return BaseType.INTEGER_TUPLE  # Numerical supertype
    if types <= PATCH:
        return BaseType.OBJECT  # Use Object as Patch representative
    if types <= ELEMENT:
        return BaseType.GRID  # Use Grid as Element representative
    if types <= PIECE:
        return BaseType.GRID  # Use Grid as Piece representative
    if types <= CONTAINER_TYPES:
        return BaseType.CONTAINER

    return BaseType.ANY
