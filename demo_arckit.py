"""
Demo script showing how to use arckit with the ARC-DSL solvers.

This script:
1. Loads ARC tasks using arckit
2. Tests existing DSL solvers against the tasks
3. Visualizes tasks in the terminal
"""
import arckit
import arckit.vis as vis
import solvers


def grid_to_tuple(grid):
    """Convert numpy array to tuple of tuples (DSL format)."""
    return tuple(tuple(int(x) for x in row) for row in grid)


def test_solver_on_task(task):
    """Test if we have a solver for this task and if it works."""
    solver_name = f"solve_{task.id}"

    if not hasattr(solvers, solver_name):
        return None, "no solver"

    solver = getattr(solvers, solver_name)

    # Test on all examples (train + test)
    all_examples = task.train + task.test

    for i, (input_grid, output_grid) in enumerate(all_examples):
        input_tuple = grid_to_tuple(input_grid)
        expected = grid_to_tuple(output_grid)

        try:
            result = solver(input_tuple)
            if result != expected:
                return False, f"wrong output on example {i}"
        except Exception as e:
            return False, f"error on example {i}: {e}"

    return True, "all correct"


def main():
    # Load ARC-AGI dataset
    print("Loading ARC-AGI dataset...")
    train_set, eval_set = arckit.load_data("arcagi")
    print(f"Train set: {train_set}")
    print(f"Eval set: {eval_set}")
    print()

    # Count how many tasks we have solvers for
    solved = 0
    no_solver = 0
    failed = 0

    print("Testing existing solvers against arckit data...")
    print("-" * 50)

    for task in train_set:
        success, msg = test_solver_on_task(task)
        if success is None:
            no_solver += 1
        elif success:
            solved += 1
        else:
            failed += 1
            print(f"FAILED: {task.id} - {msg}")

    print("-" * 50)
    print(f"Results: {solved} solved, {failed} failed, {no_solver} no solver")
    print()

    # Show a sample task
    print("Sample task visualization:")
    print("=" * 50)
    sample_task = train_set["00d62c1b"]  # Task from the README example
    sample_task.show()

    # Check if we can solve it
    success, msg = test_solver_on_task(sample_task)
    print(f"\nSolver status for {sample_task.id}: {msg}")


if __name__ == "__main__":
    main()
