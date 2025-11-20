"""
Parsers for extracting and validating LLM responses.

Implements red-flagging parsers as described in Section 3.3 of the paper.
Red-flagging discards responses that show signs of unreliability:
1. Overly long responses
2. Incorrectly formatted responses
"""

import re
import ast
from typing import Tuple, List


def _validate_move(move):
    """Validate that a move is a list of exactly 3 integers."""
    if not isinstance(move, list) or len(move) != 3 or not all(isinstance(x, int) for x in move):
        raise ValueError("'move' must be a list of exactly 3 integers.")
    return move


def _validate_state(state, n_disks=20):
    """
    Validate that a state is a list of three lists containing all disks 1..n_disks exactly once.

    Args:
        state: The state to validate (list of 3 lists)
        n_disks: Number of disks in the problem
    """
    if not (isinstance(state, list) and len(state) == 3 and all(isinstance(t, list) for t in state)):
        raise ValueError("'next_state' must be a list of three lists.")

    flat = [x for t in state for x in t]
    if not all(isinstance(x, int) for x in flat):
        raise ValueError("All entries in 'next_state' must be integers.")

    if len(flat) != n_disks or set(flat) != set(range(1, n_disks + 1)):
        missing = sorted(set(range(1, n_disks + 1)) - set(flat))
        extra = sorted(set(flat) - set(range(1, n_disks + 1)))
        raise ValueError(
            f"State must contain 1..{n_disks} exactly once. "
            f"Missing: {missing or '[]'}, Extras: {extra or '[]'}"
        )

    return state


def parse_move_state_flag(response_text: str, n_disks: int = 20) -> Tuple[List[int], List[List[int]]]:
    """
    Red-flagging parser that strictly enforces format requirements.

    This parser will raise ValueError if:
    - The move or next_state cannot be found
    - The format is incorrect
    - The move is not [disk_id, from_peg, to_peg]
    - The state doesn't contain all disks exactly once

    Args:
        response_text: The LLM's response text
        n_disks: Number of disks in the problem (default 20)

    Returns:
        Tuple of (move, next_state)

    Raises:
        ValueError: If the response is malformed or invalid
    """
    # Match square brackets for move
    move_pat = re.compile(r"(?is)\bmove\b\s*=\s*(\[[^\[\]]*\])")
    # Match nested square brackets for state
    state_pat = re.compile(
        r"(?is)\bnext_state\b\s*=\s*(\[\s*\[[^\[\]]*\]\s*,\s*\[[^\[\]]*\]\s*,\s*\[[^\[\]]*\]\s*\])"
    )

    move_matches = list(move_pat.finditer(response_text))
    if not move_matches:
        raise ValueError("No 'move = [...]' found.")
    move_str = move_matches[-1].group(1)  # last 'move'

    state_matches = list(state_pat.finditer(response_text))
    if not state_matches:
        raise ValueError("No 'next_state = [[...],[...],[...]]' found.")
    state_str = state_matches[-1].group(1)  # last 'next_state'

    try:
        move = ast.literal_eval(move_str)
    except Exception as e:
        raise ValueError("Could not parse 'move' as a Python list.") from e

    try:
        next_state = ast.literal_eval(state_str)
    except Exception as e:
        raise ValueError("Could not parse 'next_state' as Python lists.") from e

    return _validate_move(move), _validate_state(next_state, n_disks)
