"""
Voting mechanisms for error correction.

Implements first-to-ahead-by-k voting as described in Section 3.2 and Algorithm 2.
"""

from typing import Dict, Tuple, List, Callable, Any
from collections import defaultdict
import json


def serialize_vote(move: List[int], state: List[List[int]]) -> str:
    """Serialize a vote (move, state) pair to a hashable string."""
    return json.dumps({"move": move, "state": state}, sort_keys=True)


def deserialize_vote(vote_str: str) -> Tuple[List[int], List[List[int]]]:
    """Deserialize a vote string back to (move, state) pair."""
    data = json.loads(vote_str)
    return data["move"], data["state"]


def do_voting(
    agent,
    current_state: List[List[int]],
    previous_move: List[int],
    k: int,
    max_attempts: int = 10000
) -> Tuple[List[int], List[List[int]], Dict[str, Any]]:
    """
    Execute first-to-ahead-by-k voting on a single step.

    Implements Algorithm 2 (do_voting) from the paper.

    This continues sampling votes until one candidate is ahead of all others by at least k votes.
    This is a generalization of the gambler's ruin problem and is related to sequential
    probability ratio testing (SPRT).

    Args:
        agent: Agent object with get_vote() method
        current_state: Current disk configuration
        previous_move: Previous move
        k: Vote margin required to win
        max_attempts: Maximum number of sampling attempts

    Returns:
        Tuple of (winning_move, winning_state, stats_dict)
        stats_dict contains:
            - total_samples: Total samples attempted
            - valid_samples: Valid votes (not red-flagged)
            - invalid_samples: Red-flagged samples
            - vote_counts: Dict of vote counts
            - winner_votes: Number of votes for winner

    Raises:
        RuntimeError: If max_attempts is exceeded
    """
    vote_counts = defaultdict(int)  # Vote counts: V in the algorithm
    total_samples = 0
    valid_samples = 0
    invalid_samples = 0

    # Use temperature 0 for first vote to get best guess
    first_vote = True

    while total_samples < max_attempts:
        total_samples += 1

        try:
            # Get a vote (Algorithm 3: get_vote)
            # Uses temperature 0 for first, 0.1 for rest (as in Section 4.4)
            temp = 0.0 if first_vote else None
            move, state = agent.get_vote(current_state, previous_move, temperature=temp)
            first_vote = False

            # Serialize vote for counting
            vote_key = serialize_vote(move, state)
            vote_counts[vote_key] += 1
            valid_samples += 1

            # Check if this vote is ahead by k (Algorithm 2, line 6)
            current_count = vote_counts[vote_key]
            max_other_count = max([v for k, v in vote_counts.items() if k != vote_key], default=0)

            if current_count >= k + max_other_count:
                # Winner found!
                winning_move, winning_state = deserialize_vote(vote_key)
                stats = {
                    "total_samples": total_samples,
                    "valid_samples": valid_samples,
                    "invalid_samples": invalid_samples,
                    "vote_counts": {k: v for k, v in vote_counts.items()},
                    "winner_votes": current_count,
                    "num_candidates": len(vote_counts)
                }
                return winning_move, winning_state, stats

        except ValueError as e:
            # Red-flagged response (Section 3.3)
            invalid_samples += 1
            continue

    # Should not reach here in practice
    raise RuntimeError(
        f"Voting did not converge after {max_attempts} attempts. "
        f"Valid: {valid_samples}, Invalid: {invalid_samples}, "
        f"Top votes: {sorted(vote_counts.values(), reverse=True)[:5]}"
    )


def first_to_k_voting(
    agent,
    current_state: List[List[int]],
    previous_move: List[int],
    k: int,
    max_attempts: int = 10000
) -> Tuple[List[int], List[List[int]], Dict[str, Any]]:
    """
    Simpler first-to-k voting (first candidate to reach k votes wins).

    This is less statistically powerful than first-to-ahead-by-k but
    still provides error correction. Mentioned in Section 4.4.

    Args:
        agent: Agent object with get_vote() method
        current_state: Current disk configuration
        previous_move: Previous move
        k: Number of votes required to win
        max_attempts: Maximum number of sampling attempts

    Returns:
        Tuple of (winning_move, winning_state, stats_dict)
    """
    vote_counts = defaultdict(int)
    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    first_vote = True

    while total_samples < max_attempts:
        total_samples += 1

        try:
            temp = 0.0 if first_vote else None
            move, state = agent.get_vote(current_state, previous_move, temperature=temp)
            first_vote = False

            vote_key = serialize_vote(move, state)
            vote_counts[vote_key] += 1
            valid_samples += 1

            # Check if any candidate has reached k votes
            if vote_counts[vote_key] >= k:
                winning_move, winning_state = deserialize_vote(vote_key)
                stats = {
                    "total_samples": total_samples,
                    "valid_samples": valid_samples,
                    "invalid_samples": invalid_samples,
                    "vote_counts": {k: v for k, v in vote_counts.items()},
                    "winner_votes": vote_counts[vote_key],
                    "num_candidates": len(vote_counts)
                }
                return winning_move, winning_state, stats

        except ValueError:
            invalid_samples += 1
            continue

    raise RuntimeError(f"Voting did not converge after {max_attempts} attempts.")
