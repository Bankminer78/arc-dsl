"""
Interactive ARC task explorer using arckit and the DSL.

Usage:
    python explore.py                  # Browse random unsolved tasks
    python explore.py 00d62c1b         # View specific task
    python explore.py --eval           # Browse evaluation set
    python explore.py --failed         # Show failing solvers
"""
import sys
import random
import arckit
import solvers
from dsl import *
from constants import *


def grid_to_tuple(grid):
    """Convert numpy array to tuple of tuples (DSL format)."""
    return tuple(tuple(int(x) for x in row) for row in grid)


def tuple_to_list(grid):
    """Convert tuple of tuples back to list of lists for display."""
    return [list(row) for row in grid]


def get_solver(task_id):
    """Get solver function if it exists."""
    solver_name = f"solve_{task_id}"
    if hasattr(solvers, solver_name):
        return getattr(solvers, solver_name)
    return None


def test_solver(task, solver):
    """Test solver on task, return (success, message)."""
    all_examples = task.train + task.test
    for i, (input_grid, output_grid) in enumerate(all_examples):
        input_tuple = grid_to_tuple(input_grid)
        expected = grid_to_tuple(output_grid)
        try:
            result = solver(input_tuple)
            if result != expected:
                return False, f"wrong output on example {i}"
        except Exception as e:
            return False, f"error: {e}"
    return True, "correct"


def show_task(task):
    """Display a task with solver status."""
    print(f"\n{'='*60}")
    print(f"Task: {task.id}")
    print(f"Train examples: {len(task.train)}, Test examples: {len(task.test)}")
    print("="*60)
    task.show()

    solver = get_solver(task.id)
    if solver:
        success, msg = test_solver(task, solver)
        status = "✓ SOLVED" if success else f"✗ FAILED: {msg}"
        print(f"\nSolver status: {status}")

        # Show the solver code
        import inspect
        print(f"\nSolver code:")
        print("-" * 40)
        print(inspect.getsource(solver))
    else:
        print(f"\n⚠ No solver exists for this task")
        print("You can create one in solvers.py:")
        print(f"\ndef solve_{task.id}(I):")
        print("    # Your solution here")
        print("    O = I  # placeholder")
        print("    return O")


def find_failed_solvers(train_set):
    """Find tasks where solvers exist but fail."""
    failed = []
    for task in train_set:
        solver = get_solver(task.id)
        if solver:
            success, msg = test_solver(task, solver)
            if not success:
                failed.append((task, msg))
    return failed


def main():
    # Load dataset
    dataset = "arcagi"
    if "--eval" in sys.argv:
        use_eval = True
        sys.argv.remove("--eval")
    else:
        use_eval = False

    train_set, eval_set = arckit.load_data(dataset)
    task_set = eval_set if use_eval else train_set

    print(f"Loaded {task_set} from {dataset}")

    # Handle --failed flag
    if "--failed" in sys.argv:
        print("\nSearching for failed solvers...")
        failed = find_failed_solvers(train_set)
        if failed:
            print(f"\nFound {len(failed)} failed solver(s):")
            for task, msg in failed:
                print(f"  - {task.id}: {msg}")
                show_task(task)
        else:
            print("All solvers pass!")
        return

    # Handle specific task ID
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        try:
            task = task_set[task_id]
            show_task(task)
        except (KeyError, IndexError):
            print(f"Task '{task_id}' not found in {'eval' if use_eval else 'train'} set")
        return

    # Interactive mode - show random tasks without solvers (from eval set)
    print("\nShowing random tasks from evaluation set (no solvers exist)...")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            task = random.choice(list(eval_set))
            show_task(task)
            input("\nPress Enter for next task...")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
