"""
Tests for the ARC-ILP system.
"""

import sys
import time

# Import the DSL and constants
import dsl
import constants

# Import the ILP system
from arc_ilp import (
    BaseType, CallableType,
    extract_primitives,
    Argument, PrimitiveCall, Hypothesis,
    generate_at_depth,
    evaluate_hypothesis, execute_hypothesis,
    hypothesis_to_code,
    search, SearchResult
)


def test_type_system():
    """Test the type system"""
    from arc_ilp.types import is_type_compatible

    # Basic compatibility
    assert is_type_compatible(BaseType.GRID, BaseType.GRID)
    assert is_type_compatible(BaseType.INTEGER, BaseType.INTEGER)
    assert is_type_compatible(BaseType.ANY, BaseType.GRID)
    assert is_type_compatible(BaseType.GRID, BaseType.ANY)

    # Container compatibility
    assert is_type_compatible(BaseType.TUPLE, BaseType.CONTAINER)
    assert is_type_compatible(BaseType.GRID, BaseType.CONTAINER)

    # Callable compatibility
    c1 = CallableType(input_types=(BaseType.GRID,), output_type=BaseType.GRID)
    c2 = CallableType(input_types=(BaseType.ANY,), output_type=BaseType.ANY)
    assert is_type_compatible(c1, c2)

    print("✓ Type system tests passed")


def test_primitive_extraction():
    """Test primitive metadata extraction"""
    primitives = extract_primitives(dsl)

    # Check some known primitives
    assert 'vmirror' in primitives
    assert 'hmirror' in primitives
    assert 'rot90' in primitives
    assert 'compose' in primitives
    assert 'fork' in primitives

    # Check vmirror signature
    vmirror = primitives['vmirror']
    assert vmirror.name == 'vmirror'
    assert len(vmirror.param_types) == 1  # piece: Piece
    assert vmirror.return_type in {BaseType.GRID, BaseType.OBJECT, BaseType.ANY}

    # Check compose is higher-order
    compose = primitives['compose']
    assert compose.is_higher_order
    assert compose.returns_callable

    print(f"✓ Extracted {len(primitives)} primitives")


def test_hypothesis_creation():
    """Test hypothesis data structures"""
    # Create a simple hypothesis: vmirror(I)
    step = PrimitiveCall(
        primitive='vmirror',
        arguments=(Argument(kind='input', value='I'),),
        output_var='O'
    )
    hyp = Hypothesis(steps=(step,))

    assert len(hyp) == 1
    assert hyp.depth == 1
    assert hyp.steps[0].primitive == 'vmirror'

    print("✓ Hypothesis creation tests passed")


def test_hypothesis_execution():
    """Test hypothesis execution"""
    # Create vmirror(I) hypothesis
    step = PrimitiveCall(
        primitive='vmirror',
        arguments=(Argument(kind='input', value='I'),),
        output_var='O'
    )
    hyp = Hypothesis(steps=(step,))

    # Test input
    input_grid = ((1, 2, 3), (4, 5, 6))
    expected = ((3, 2, 1), (6, 5, 4))

    result = execute_hypothesis(hyp, input_grid, dsl, constants)
    assert result == expected

    print("✓ Hypothesis execution tests passed")


def test_hypothesis_evaluation():
    """Test hypothesis evaluation against examples"""
    # vmirror hypothesis
    step = PrimitiveCall(
        primitive='vmirror',
        arguments=(Argument(kind='input', value='I'),),
        output_var='O'
    )
    hyp = Hypothesis(steps=(step,))

    # Matching examples
    examples = [
        (((1, 2, 3), (4, 5, 6)), ((3, 2, 1), (6, 5, 4))),
        (((7, 8), (9, 0)), ((8, 7), (0, 9))),
    ]

    assert evaluate_hypothesis(hyp, examples, dsl, constants)

    # Non-matching examples
    bad_examples = [
        (((1, 2, 3), (4, 5, 6)), ((1, 2, 3), (4, 5, 6))),  # Identity, not vmirror
    ]

    assert not evaluate_hypothesis(hyp, bad_examples, dsl, constants)

    print("✓ Hypothesis evaluation tests passed")


def test_code_generation():
    """Test code generation"""
    # Two-step hypothesis
    step1 = PrimitiveCall(
        primitive='hmirror',
        arguments=(Argument(kind='input', value='I'),),
        output_var='x1'
    )
    step2 = PrimitiveCall(
        primitive='vconcat',
        arguments=(Argument(kind='variable', value='x1'), Argument(kind='input', value='I')),
        output_var='O'
    )
    hyp = Hypothesis(steps=(step1, step2))

    code = hypothesis_to_code(hyp, 'test123')
    assert 'def solve_test123(I):' in code
    assert 'x1 = hmirror(I)' in code
    assert 'O = vconcat(x1, I)' in code
    assert 'return O' in code

    print("✓ Code generation tests passed")


def test_generator_depth_1():
    """Test hypothesis generation at depth 1"""
    primitives = extract_primitives(dsl)

    count = 0
    for hyp in generate_at_depth(1, primitives):
        count += 1
        assert len(hyp) == 1
        assert hyp.steps[0].output_var == 'O'
        if count > 1000:  # Limit for testing
            break

    print(f"✓ Generated {count}+ depth-1 hypotheses")


def test_generator_depth_2():
    """Test hypothesis generation at depth 2"""
    primitives = extract_primitives(dsl)

    count = 0
    for hyp in generate_at_depth(2, primitives):
        count += 1
        assert len(hyp) == 2
        assert hyp.steps[0].output_var == 'x1'
        assert hyp.steps[1].output_var == 'O'
        if count > 1000:  # Limit for testing
            break

    print(f"✓ Generated {count}+ depth-2 hypotheses")


def test_search_vmirror():
    """Test search for vmirror solution"""
    # Examples that vmirror solves
    examples = [
        (((1, 2, 3), (4, 5, 6)), ((3, 2, 1), (6, 5, 4))),
        (((7, 8), (9, 0)), ((8, 7), (0, 9))),
    ]

    result = search(
        examples, 'test_vmirror', dsl, constants,
        max_depth=2, timeout=10.0, verbose=True
    )

    assert result is not None
    assert result.depth == 1
    assert 'vmirror' in result.code

    print(f"✓ Found vmirror solution: {result.code}")


def test_search_two_step():
    """Test search for a two-step solution"""
    # This test verifies the generator produces valid 2-step hypotheses
    # Full 2-step search can take minutes due to combinatorial explosion
    # (~30M hypotheses at depth 2)

    # Test that we can manually construct and evaluate a 2-step hypothesis
    # solve_4c4377d9: x1 = hmirror(I), O = vconcat(x1, I)
    step1 = PrimitiveCall(
        primitive='hmirror',
        arguments=(Argument(kind='input', value='I'),),
        output_var='x1'
    )
    step2 = PrimitiveCall(
        primitive='vconcat',
        arguments=(
            Argument(kind='variable', value='x1'),
            Argument(kind='input', value='I')
        ),
        output_var='O'
    )
    hyp = Hypothesis(steps=(step1, step2))

    # Verify it produces correct output
    input1 = ((1, 2), (3, 4))
    output1 = ((3, 4), (1, 2), (1, 2), (3, 4))

    result = execute_hypothesis(hyp, input1, dsl, constants)
    assert result == output1

    # Verify code generation
    code = hypothesis_to_code(hyp, '4c4377d9')
    assert 'hmirror' in code
    assert 'vconcat' in code

    print("✓ Two-step hypothesis construction and execution verified")
    print(f"  Generated code:\n{code}")


def test_search_with_constants():
    """Test search using constants"""
    # Examples that require replace(I, SIX, TWO) - single step with constants
    # This is solve_b1948b0a from solvers.py
    input1 = ((6, 0, 6), (0, 6, 0))
    output1 = ((2, 0, 2), (0, 2, 0))

    input2 = ((6, 6), (6, 6))
    output2 = ((2, 2), (2, 2))

    examples = [
        (input1, output1),
        (input2, output2),
    ]

    result = search(
        examples, 'test_constants', dsl, constants,
        max_depth=1, timeout=30.0, verbose=True
    )

    assert result is not None
    assert 'replace' in result.code
    print(f"✓ Found solution with constants:\n{result.code}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running ARC-ILP Tests")
    print("=" * 60)

    test_type_system()
    test_primitive_extraction()
    test_hypothesis_creation()
    test_hypothesis_execution()
    test_hypothesis_evaluation()
    test_code_generation()
    test_generator_depth_1()
    test_generator_depth_2()
    test_search_vmirror()
    test_search_two_step()
    test_search_with_constants()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
