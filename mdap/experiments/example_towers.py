"""
Example script demonstrating MAKER on Towers of Hanoi.

This example shows how to:
1. Set up an LLM function
2. Create a MAKER agent
3. Run MAKER to solve a problem
4. Verify the solution
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mdap.core.agent import TowersOfHanoiAgent
from mdap.core.maker import MAKER
from mdap.domains.towers_of_hanoi import (
    create_initial_state,
    verify_solution,
    get_optimal_num_steps
)


def create_mock_llm():
    """
    Create a mock LLM that solves Towers of Hanoi perfectly.

    This is for demonstration purposes only. In practice, you would
    use a real LLM API like OpenAI, Anthropic, or open-source models.

    The mock LLM precomputes the optimal solution and returns steps from it.
    """
    from mdap.domains.towers_of_hanoi import apply_move, is_valid_move

    def generate_optimal_solution(n_disks):
        """Generate optimal ToH solution moving from peg 0 to peg 2."""
        moves = []

        def hanoi_recursive(n, source, target, auxiliary):
            if n == 1:
                moves.append([1, source, target])
            else:
                hanoi_recursive(n - 1, source, auxiliary, target)
                moves.append([n, source, target])
                hanoi_recursive(n - 1, auxiliary, target, source)

        hanoi_recursive(n_disks, 0, 2, 1)
        return moves

    # Cache solutions for different disk counts
    solutions_cache = {}

    def solve_single_step(prev_move, current_state):
        """Find the next move by looking up in optimal solution."""
        n_disks = sum(len(peg) for peg in current_state)

        if n_disks not in solutions_cache:
            solutions_cache[n_disks] = generate_optimal_solution(n_disks)

        optimal_moves = solutions_cache[n_disks]

        # Find current position in solution by replaying from start
        test_state = [[n_disks - i for i in range(n_disks)], [], []]
        for i, move in enumerate(optimal_moves):
            if test_state == current_state:
                # Found current state, return next move
                next_move = optimal_moves[i]
                next_state = apply_move(current_state, next_move)
                return next_move, next_state

            test_state = apply_move(test_state, move)

        # If we're at the end
        if test_state == current_state:
            raise ValueError("Already at goal state")

        raise ValueError("Could not find current state in solution")

    def mock_llm(system_prompt, user_prompt, kwargs):
        """Mock LLM function."""
        import re

        # Extract previous move and current state from prompt
        prev_match = re.search(r"Previous move: (\[.*?\])", user_prompt)
        state_match = re.search(r"Current State: (\[\[.*?\]\])", user_prompt)

        if not prev_match or not state_match:
            raise ValueError("Could not parse prompt")

        prev_move = eval(prev_match.group(1))
        current_state = eval(state_match.group(1))

        # Solve the step
        move, next_state = solve_single_step(prev_move, current_state)

        # Format response
        response = f"""
Let me solve this step by step.

Previous move: {prev_move}
Current state: {current_state}

Following the standard procedure, the next move is:

move = {move}
next_state = {next_state}
"""
        return response

    return mock_llm


def create_openai_llm(api_key: str, model: str = "gpt-4"):
    """
    Create an OpenAI LLM function.

    Args:
        api_key: OpenAI API key
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")

    Returns:
        LLM function compatible with MAKER
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    client = OpenAI(api_key=api_key)

    def llm_fn(system_prompt, user_prompt, kwargs):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=kwargs.get("max_tokens", 750),
            temperature=kwargs.get("temperature", 0.1)
        )
        return response.choices[0].message.content

    return llm_fn


def main():
    """Run MAKER on a small Towers of Hanoi problem."""

    # Configuration
    N_DISKS = 10  # Testing with 10 disks = 1,023 steps
    K = 3  # Vote margin (using paper's k=3 for larger problems)

    print("=" * 70)
    print("MAKER: Solving Towers of Hanoi with Maximal Agentic Decomposition")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Number of disks: {N_DISKS}")
    print(f"  Vote margin (k): {K}")
    print(f"  Expected steps: {get_optimal_num_steps(N_DISKS)}")
    print()

    # Create LLM function
    # Option 1: Mock LLM (for demonstration)
    llm_fn = create_mock_llm()
    print("Using: Mock LLM (perfect solver)")

    # Option 2: OpenAI (uncomment to use)
    # api_key = os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("Set OPENAI_API_KEY environment variable")
    # llm_fn = create_openai_llm(api_key, model="gpt-4")
    # print("Using: OpenAI GPT-4")

    print()

    # Create agent
    agent = TowersOfHanoiAgent(
        llm_fn=llm_fn,
        n_disks=N_DISKS,
        max_tokens=750,
        temperature=0.1
    )

    # Create MAKER system
    maker = MAKER(
        agent=agent,
        k=K,
        voting_method="first-to-ahead-by-k",
        verbose=True
    )

    # Initial state
    initial_state = create_initial_state(N_DISKS)
    num_steps = get_optimal_num_steps(N_DISKS)

    # Generate solution
    print("Starting MAKER...\n")
    solution = maker.generate_solution(
        initial_state=initial_state,
        num_steps=num_steps,
        initial_move=[0, 0, 1]  # Placeholder
    )

    # Verify solution
    print("\n" + "=" * 70)
    print("Verifying solution...")
    print("=" * 70)
    is_valid, message, trajectory = verify_solution(initial_state, solution, N_DISKS)

    if is_valid:
        print("✓ Solution is VALID!")
        print(f"✓ Zero errors across {len(solution)} steps")
    else:
        print(f"✗ Solution is INVALID: {message}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    stats = maker.get_stats()
    print(f"Steps completed: {stats['steps_completed']}")
    print(f"Total LLM calls: {stats['total_samples']}")
    print(f"Valid votes: {stats['total_valid_votes']}")
    print(f"Red-flagged: {stats['total_invalid_votes']}")
    print(f"Average samples per step: {stats['total_samples'] / num_steps:.2f}")
    print(f"Red-flag rate: {100 * stats['total_invalid_votes'] / stats['total_samples']:.1f}%")

    # Show first few moves
    print("\n" + "=" * 70)
    print("Solution (first 10 moves):")
    print("=" * 70)
    for i, move in enumerate(solution[:10]):
        print(f"  Step {i+1}: Move disk {move[0]} from peg {move[1]} to peg {move[2]}")

    if len(solution) > 10:
        print(f"  ... ({len(solution) - 10} more moves)")

    print("\n" + "=" * 70)
    print("MAKER demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
