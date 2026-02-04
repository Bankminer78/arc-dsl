"""
Primitive metadata extraction from dsl.py.

Uses Python's inspect and typing modules to extract type signatures
from DSL functions.
"""

import inspect
from typing import Dict, List, get_type_hints, Callable, Any, Union, Tuple, FrozenSet
from dataclasses import dataclass

from .types import BaseType, CallableType, Type


@dataclass
class PrimitiveInfo:
    """Metadata about a DSL primitive"""
    name: str
    param_names: List[str]
    param_types: List[Type]
    return_type: Type
    is_higher_order: bool  # Takes Callable args
    returns_callable: bool  # Returns Callable


# Map Python type annotations to our type system
TYPE_MAP = {
    'Boolean': BaseType.BOOLEAN,
    'bool': BaseType.BOOLEAN,
    'Integer': BaseType.INTEGER,
    'int': BaseType.INTEGER,
    'IntegerTuple': BaseType.INTEGER_TUPLE,
    'Numerical': BaseType.INTEGER_TUPLE,  # Union, use broader type
    'Grid': BaseType.GRID,
    'Cell': BaseType.CELL,
    'Object': BaseType.OBJECT,
    'Objects': BaseType.OBJECTS,
    'Indices': BaseType.INDICES,
    'IndicesSet': BaseType.INDICES_SET,
    'IntegerSet': BaseType.INTEGER_SET,
    'Patch': BaseType.OBJECT,  # Union of Object/Indices
    'Element': BaseType.GRID,  # Union of Object/Grid
    'Piece': BaseType.GRID,  # Union of Grid/Patch
    'Tuple': BaseType.TUPLE,
    'TupleTuple': BaseType.TUPLE_TUPLE,
    'Container': BaseType.CONTAINER,
    'ContainerContainer': BaseType.CONTAINER_CONTAINER,
    'FrozenSet': BaseType.FROZENSET,
    'Any': BaseType.ANY,
}


def annotation_to_type(annotation) -> Type:
    """Convert a Python type annotation to our Type representation"""
    if annotation is None:
        return BaseType.ANY

    # Get the string representation
    if hasattr(annotation, '__name__'):
        type_str = annotation.__name__
    elif hasattr(annotation, '_name'):
        type_str = annotation._name
    else:
        type_str = str(annotation)

    # Handle Callable types
    if 'Callable' in type_str or (hasattr(annotation, '__origin__') and
                                   annotation.__origin__ is Callable):
        # Default to a generic callable
        return CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY)

    # Handle typing module types
    if hasattr(annotation, '__origin__'):
        origin = annotation.__origin__
        if origin is tuple or (hasattr(origin, '__name__') and origin.__name__ == 'Tuple'):
            args = getattr(annotation, '__args__', ())
            if args and len(args) == 2 and all(arg is int or
                (hasattr(arg, '__name__') and arg.__name__ in ('int', 'Integer')) for arg in args):
                return BaseType.INTEGER_TUPLE
            return BaseType.TUPLE
        if origin is frozenset or (hasattr(origin, '__name__') and origin.__name__ == 'FrozenSet'):
            return BaseType.FROZENSET
        if origin is Union:
            # Handle Union types
            args = getattr(annotation, '__args__', ())
            type_strs = []
            for arg in args:
                if hasattr(arg, '__name__'):
                    type_strs.append(arg.__name__)
            # Check for known unions
            if set(type_strs) == {'int', 'tuple'} or set(type_strs) == {'Integer', 'IntegerTuple'}:
                return BaseType.INTEGER_TUPLE  # Numerical
            if 'Object' in type_strs and 'Grid' in type_strs:
                return BaseType.GRID  # Element
            if 'Object' in type_strs and 'Indices' in type_strs:
                return BaseType.OBJECT  # Patch
            return BaseType.ANY

    # Strip generic args for lookup
    base_type_str = type_str.split('[')[0]

    # Direct lookup
    if base_type_str in TYPE_MAP:
        return TYPE_MAP[base_type_str]

    # Check for subscripted types
    for key in TYPE_MAP:
        if key in type_str:
            return TYPE_MAP[key]

    return BaseType.ANY


def extract_primitives(dsl_module) -> Dict[str, PrimitiveInfo]:
    """
    Extract type signatures from DSL module using inspect and typing.

    Args:
        dsl_module: The DSL module (typically imported as `dsl`)

    Returns:
        Dictionary mapping primitive names to their PrimitiveInfo
    """
    primitives = {}

    for name, obj in inspect.getmembers(dsl_module, inspect.isfunction):
        # Skip private functions
        if name.startswith('_'):
            continue

        try:
            sig = inspect.signature(obj)
            # Use raw annotations to preserve type aliases like Grid, Object, etc.
            # get_type_hints() resolves aliases which loses information
            raw_annotations = getattr(obj, '__annotations__', {})
        except (ValueError, TypeError):
            # Skip functions where we can't get signature
            continue

        param_names = []
        param_types = []

        for param_name, param in sig.parameters.items():
            param_names.append(param_name)
            if param_name in raw_annotations:
                param_type = annotation_to_type(raw_annotations[param_name])
            else:
                param_type = BaseType.ANY
            param_types.append(param_type)

        # Get return type
        if 'return' in raw_annotations:
            return_type = annotation_to_type(raw_annotations['return'])
        else:
            return_type = BaseType.ANY

        # Check if higher-order (takes Callable)
        is_higher_order = any(isinstance(pt, CallableType) for pt in param_types)

        # Check if returns Callable
        returns_callable = isinstance(return_type, CallableType)

        primitives[name] = PrimitiveInfo(
            name=name,
            param_names=param_names,
            param_types=param_types,
            return_type=return_type,
            is_higher_order=is_higher_order,
            returns_callable=returns_callable
        )

    return primitives


# Higher-order primitives that take functions and return functions
HIGHER_ORDER_RETURNING_CALLABLE = {
    'compose': CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY),
    'chain': CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY),
    'fork': CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY),
    'lbind': CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY),
    'rbind': CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY),
    'matcher': CallableType(input_types=(BaseType.ANY,), output_type=BaseType.BOOLEAN),
    'power': CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY),
}

# Primitives that take functions as arguments
FUNCTION_TAKING_PRIMITIVES = {
    'compose', 'chain', 'fork', 'lbind', 'rbind', 'matcher', 'power',
    'apply', 'mapply', 'sfilter', 'mfilter', 'argmax', 'argmin',
    'order', 'extract', 'valmax', 'valmin', 'rapply', 'papply', 'mpapply', 'prapply'
}


def get_callable_return_type(primitive_name: str, arg_types: List[Type]) -> Type:
    """
    Get the return type for a callable-returning primitive based on its arguments.

    For compose, chain, fork, lbind, rbind - the return type is a Callable.
    For branch - depends on the argument types.
    """
    if primitive_name in HIGHER_ORDER_RETURNING_CALLABLE:
        return HIGHER_ORDER_RETURNING_CALLABLE[primitive_name]

    if primitive_name == 'branch':
        # branch returns the type of its second/third argument
        if len(arg_types) >= 2:
            return arg_types[1]  # Return type of 'a' argument
        return BaseType.ANY

    return BaseType.ANY
