"""
MAKER: Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging.

Implements the main MAKER system as described in Algorithm 1.
"""

from typing import List, Dict, Any, Callable, Optional
from .agent import TowersOfHanoiAgent
from .voting import do_voting, first_to_k_voting


class MAKER:
    """
    Main MAKER system for solving Towers of Hanoi with zero errors.

    Combines three key components:
    1. Maximal Agentic Decomposition (MAD): Each agent handles one step
    2. First-to-ahead-by-k voting: Error correction at each step
    3. Red-flagging: Discard unreliable responses
    """

    def __init__(
        self,
        agent: TowersOfHanoiAgent,
        k: int = 3,
        voting_method: str = "first-to-ahead-by-k",
        verbose: bool = False
    ):
        """
        Initialize MAKER system.

        Args:
            agent: TowersOfHanoiAgent instance
            k: Vote margin required (default 3, as used in Section 4.4)
            voting_method: "first-to-ahead-by-k" or "first-to-k"
            verbose: Print progress information
        """
        self.agent = agent
        self.k = k
        self.voting_method = voting_method
        self.verbose = verbose

        # Select voting function
        if voting_method == "first-to-ahead-by-k":
            self.voting_fn = do_voting
        elif voting_method == "first-to-k":
            self.voting_fn = first_to_k_voting
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")

        # Statistics tracking
        self.stats = {
            "steps_completed": 0,
            "total_samples": 0,
            "total_valid_votes": 0,
            "total_invalid_votes": 0,
            "step_stats": []
        }

    def generate_solution(
        self,
        initial_state: List[List[int]],
        num_steps: int,
        initial_move: Optional[List[int]] = None
    ) -> List[List[int]]:
        """
        Generate a solution for the Towers of Hanoi problem.

        Implements Algorithm 1 (generate_solution) from the paper.

        Args:
            initial_state: Initial disk configuration [[peg0], [peg1], [peg2]]
            num_steps: Number of steps to execute (2^n - 1 for n disks)
            initial_move: Optional initial move (default [0, 0, 1] for starting)

        Returns:
            List of all moves in the solution

        The algorithm proceeds as follows:
        1. For each step:
           a. Call do_voting to get the next move via first-to-ahead-by-k voting
           b. Update the current state
           c. Track statistics
        2. Return the complete action sequence
        """
        action_list = []
        current_state = [peg[:] for peg in initial_state]  # Deep copy
        previous_move = initial_move or [0, 0, 1]  # Placeholder initial move

        if self.verbose:
            print(f"MAKER starting: {num_steps} steps with k={self.k}, method={self.voting_method}")
            print(f"Initial state: {initial_state}")

        for step in range(num_steps):
            if self.verbose and (step + 1) % 1000 == 0:
                print(f"Step {step + 1}/{num_steps} - Samples: {self.stats['total_samples']}")

            # Algorithm 1, line 5: Call do_voting
            move, next_state, vote_stats = self.voting_fn(
                self.agent,
                current_state,
                previous_move,
                self.k
            )

            # Update statistics
            self.stats["steps_completed"] += 1
            self.stats["total_samples"] += vote_stats["total_samples"]
            self.stats["total_valid_votes"] += vote_stats["valid_samples"]
            self.stats["total_invalid_votes"] += vote_stats["invalid_samples"]
            self.stats["step_stats"].append({
                "step": step,
                "move": move,
                "samples": vote_stats["total_samples"],
                "valid_votes": vote_stats["valid_samples"],
                "invalid_votes": vote_stats["invalid_samples"],
                "winner_votes": vote_stats["winner_votes"],
                "num_candidates": vote_stats["num_candidates"]
            })

            # Algorithm 1, line 6: Append action
            action_list.append(move)

            # Update state for next iteration
            current_state = next_state
            previous_move = move

        if self.verbose:
            print(f"\nMAKER completed successfully!")
            print(f"Total steps: {self.stats['steps_completed']}")
            print(f"Total LLM calls: {self.stats['total_samples']}")
            print(f"Valid votes: {self.stats['total_valid_votes']}")
            print(f"Invalid (red-flagged): {self.stats['total_invalid_votes']}")
            print(f"Average samples per step: {self.stats['total_samples'] / num_steps:.2f}")

        return action_list

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the solution process."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "steps_completed": 0,
            "total_samples": 0,
            "total_valid_votes": 0,
            "total_invalid_votes": 0,
            "step_stats": []
        }
