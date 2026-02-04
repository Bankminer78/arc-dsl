"""
Hypothesis data structures for representing candidate programs.

A hypothesis is a sequence of primitive calls that transforms an input Grid
to produce an output Grid.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional

from .types import BaseType, CallableType, Type


@dataclass(frozen=True)
class Argument:
    """An argument to a primitive call"""
    kind: str  # 'constant', 'variable', 'input', 'primitive'
    value: str  # Name of constant, variable (x1), 'I', or primitive name

    def __repr__(self):
        if self.kind == 'input':
            return 'I'
        elif self.kind == 'constant':
            return self.value
        elif self.kind == 'primitive':
            return self.value
        else:  # variable
            return self.value


@dataclass(frozen=True)
class PrimitiveCall:
    """A single primitive invocation"""
    primitive: str        # Name of primitive
    arguments: tuple      # Tuple of Arguments (immutable for hashing)
    output_var: str       # Variable name for output (x1, x2, etc., or O)

    def __repr__(self):
        args = ', '.join(str(a) for a in self.arguments)
        return f"{self.output_var} = {self.primitive}({args})"


@dataclass(frozen=True)
class CallVariable:
    """Call a function-typed variable: x2(arg1, arg2, ...)"""
    callee: str           # Variable name holding the function
    arguments: tuple      # Tuple of Arguments (immutable for hashing)
    output_var: str       # Variable name for output

    def __repr__(self):
        args = ', '.join(str(a) for a in self.arguments)
        return f"{self.output_var} = {self.callee}({args})"


# A step is either a primitive call or calling a variable
Step = Union[PrimitiveCall, CallVariable]


@dataclass(frozen=True)
class Hypothesis:
    """
    A candidate program as a sequence of steps.

    Invariant: last step's output_var is 'O' and has type Grid.
    """
    steps: tuple  # Tuple of Steps (immutable for hashing)

    def __repr__(self):
        lines = [str(step) for step in self.steps]
        return '\n'.join(lines)

    def __len__(self):
        return len(self.steps)

    @property
    def depth(self) -> int:
        """Number of steps in the hypothesis"""
        return len(self.steps)


@dataclass
class TypeEnv:
    """
    Track types of variables during hypothesis construction.

    This is a mutable helper class (not part of the hypothesis representation).
    """
    variables: Dict[str, Type] = field(default_factory=dict)

    def __init__(self):
        self.variables = {'I': BaseType.GRID}

    def copy(self) -> 'TypeEnv':
        """Create a copy of this type environment"""
        new_env = TypeEnv()
        new_env.variables = dict(self.variables)
        return new_env

    def add(self, name: str, typ: Type):
        """Add a variable binding"""
        self.variables[name] = typ

    def get(self, name: str) -> Optional[Type]:
        """Get the type of a variable"""
        return self.variables.get(name)

    def get_compatible(self, target_type: Type) -> List[str]:
        """Get all variable names compatible with target_type"""
        from .types import is_type_compatible
        return [name for name, typ in self.variables.items()
                if is_type_compatible(typ, target_type)]

    def get_callable_vars(self) -> List[str]:
        """Get all variable names that hold callable values"""
        return [name for name, typ in self.variables.items()
                if isinstance(typ, CallableType)]


def build_type_env(hypothesis: Hypothesis, primitives: Dict) -> TypeEnv:
    """Build a type environment from a partial hypothesis"""
    from .primitives import get_callable_return_type, HIGHER_ORDER_RETURNING_CALLABLE

    env = TypeEnv()

    for step in hypothesis.steps:
        if isinstance(step, PrimitiveCall):
            prim_info = primitives.get(step.primitive)
            if prim_info:
                # Check if this is a callable-returning primitive
                if step.primitive in HIGHER_ORDER_RETURNING_CALLABLE:
                    return_type = HIGHER_ORDER_RETURNING_CALLABLE[step.primitive]
                else:
                    return_type = prim_info.return_type
            else:
                return_type = BaseType.ANY
            env.add(step.output_var, return_type)
        else:  # CallVariable
            # When calling a callable variable, the return type comes from the callable
            callee_type = env.get(step.callee)
            if isinstance(callee_type, CallableType):
                return_type = callee_type.output_type
            else:
                return_type = BaseType.ANY
            env.add(step.output_var, return_type)

    return env


def next_var_name(hypothesis: Hypothesis) -> str:
    """Get the next variable name for a hypothesis"""
    return f"x{len(hypothesis.steps) + 1}"
