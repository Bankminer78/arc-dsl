"""
Code generation for hypotheses.

Converts hypotheses to solvers.py format code.
"""

from .hypothesis import Hypothesis, PrimitiveCall, CallVariable, Argument


def format_arg(arg: Argument) -> str:
    """Format an argument for code output"""
    if arg.kind == 'constant':
        return arg.value
    elif arg.kind == 'variable':
        return arg.value
    elif arg.kind == 'input':
        return 'I'
    elif arg.kind == 'primitive':
        return arg.value
    else:
        raise ValueError(f"Unknown argument kind: {arg.kind}")


def hypothesis_to_code(hypothesis: Hypothesis, task_id: str) -> str:
    """
    Convert hypothesis to solvers.py format.

    Args:
        hypothesis: The hypothesis to convert
        task_id: The task ID for the function name

    Returns:
        Python code string in solvers.py format
    """
    lines = [f"def solve_{task_id}(I):"]

    for step in hypothesis.steps:
        if isinstance(step, PrimitiveCall):
            args = ', '.join(format_arg(a) for a in step.arguments)
            lines.append(f"    {step.output_var} = {step.primitive}({args})")
        elif isinstance(step, CallVariable):
            args = ', '.join(format_arg(a) for a in step.arguments)
            lines.append(f"    {step.output_var} = {step.callee}({args})")

    lines.append("    return O")
    return '\n'.join(lines)


def hypothesis_to_lambda(hypothesis: Hypothesis) -> str:
    """
    Convert a single-step hypothesis to a lambda expression.

    Only works for depth-1 hypotheses.

    Args:
        hypothesis: The hypothesis to convert

    Returns:
        Lambda expression string
    """
    if len(hypothesis.steps) != 1:
        raise ValueError("Can only convert single-step hypotheses to lambda")

    step = hypothesis.steps[0]
    if isinstance(step, PrimitiveCall):
        args = ', '.join(format_arg(a) for a in step.arguments)
        return f"lambda I: {step.primitive}({args})"
    else:
        raise ValueError("Cannot convert variable call to lambda")


def hypothesis_repr(hypothesis: Hypothesis) -> str:
    """
    Get a compact string representation of a hypothesis.

    Useful for debugging and logging.

    Args:
        hypothesis: The hypothesis

    Returns:
        Compact string representation
    """
    steps = []
    for step in hypothesis.steps:
        if isinstance(step, PrimitiveCall):
            args = ', '.join(format_arg(a) for a in step.arguments)
            steps.append(f"{step.primitive}({args})")
        else:
            args = ', '.join(format_arg(a) for a in step.arguments)
            steps.append(f"{step.callee}({args})")
    return ' -> '.join(steps)
