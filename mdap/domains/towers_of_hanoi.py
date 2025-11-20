"""
Towers of Hanoi domain utilities.

Implements state management, validation, and solution verification
for the Towers of Hanoi problem.
"""

from typing import List, Tuple


def create_initial_state(n_disks: int) -> List[List[int]]:
    """
    Create the initial state for Towers of Hanoi with n disks.

    Args:
        n_disks: Number of disks

    Returns:
        Initial state with all disks on peg 0: [[n, n-1, ..., 2, 1], [], []]
    """
    return [[n_disks - i for i in range(n_disks)], [], []]


def is_valid_state(state: List[List[int]], n_disks: int) -> Tuple[bool, str]:
    """
    Check if a state is valid.

    A valid state must:
    1. Be a list of 3 lists (3 pegs)
    2. Contain all disks 1..n_disks exactly once
    3. Have disks on each peg in descending order (larger at bottom)

    Args:
        state: State to check
        n_disks: Expected number of disks

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check structure
    if not isinstance(state, list) or len(state) != 3:
        return False, "State must be a list of 3 lists"

    if not all(isinstance(peg, list) for peg in state):
        return False, "Each peg must be a list"

    # Check all disks present
    all_disks = []
    for peg in state:
        all_disks.extend(peg)

    if sorted(all_disks) != list(range(1, n_disks + 1)):
        missing = set(range(1, n_disks + 1)) - set(all_disks)
        extra = set(all_disks) - set(range(1, n_disks + 1))
        return False, f"Invalid disk set. Missing: {missing}, Extra: {extra}"

    # Check each peg has disks in descending order (larger at bottom)
    for i, peg in enumerate(state):
        for j in range(len(peg) - 1):
            if peg[j] < peg[j + 1]:  # Larger disk on top of smaller
                return False, f"Peg {i} has invalid ordering: {peg}"

    return True, ""


def is_valid_move(
    move: List[int],
    state: List[List[int]],
    n_disks: int
) -> Tuple[bool, str]:
    """
    Check if a move is valid given the current state.

    A valid move must:
    1. Be [disk_id, from_peg, to_peg] with valid disk_id and peg indices
    2. The disk must be the top disk on the from_peg
    3. The disk can be placed on the to_peg (smaller than top disk or empty)

    Args:
        move: Move to check [disk_id, from_peg, to_peg]
        state: Current state
        n_disks: Number of disks

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(move) != 3:
        return False, f"Move must have 3 elements, got {len(move)}"

    disk_id, from_peg, to_peg = move

    # Check indices
    if not (1 <= disk_id <= n_disks):
        return False, f"Invalid disk_id: {disk_id}"

    if not (0 <= from_peg <= 2):
        return False, f"Invalid from_peg: {from_peg}"

    if not (0 <= to_peg <= 2):
        return False, f"Invalid to_peg: {to_peg}"

    if from_peg == to_peg:
        return False, "Cannot move disk to same peg"

    # Check disk is on top of from_peg
    if not state[from_peg]:
        return False, f"Peg {from_peg} is empty"

    if state[from_peg][-1] != disk_id:
        return False, f"Disk {disk_id} is not on top of peg {from_peg}"

    # Check disk can be placed on to_peg
    if state[to_peg] and state[to_peg][-1] < disk_id:
        return False, f"Cannot place disk {disk_id} on smaller disk {state[to_peg][-1]}"

    return True, ""


def apply_move(state: List[List[int]], move: List[int]) -> List[List[int]]:
    """
    Apply a move to a state and return the new state.

    Args:
        state: Current state
        move: Move to apply [disk_id, from_peg, to_peg]

    Returns:
        New state after applying the move
    """
    disk_id, from_peg, to_peg = move

    # Create new state (deep copy)
    new_state = [peg[:] for peg in state]

    # Remove disk from from_peg
    removed_disk = new_state[from_peg].pop()

    # Verify it's the correct disk
    assert removed_disk == disk_id, f"Expected disk {disk_id}, got {removed_disk}"

    # Add disk to to_peg
    new_state[to_peg].append(disk_id)

    return new_state


def is_goal_state(state: List[List[int]], n_disks: int) -> bool:
    """
    Check if the state is the goal state (all disks on peg 2).

    Args:
        state: State to check
        n_disks: Number of disks

    Returns:
        True if goal state, False otherwise
    """
    return (
        state[0] == [] and
        state[1] == [] and
        state[2] == [n_disks - i for i in range(n_disks)]
    )


def verify_solution(
    initial_state: List[List[int]],
    moves: List[List[int]],
    n_disks: int
) -> Tuple[bool, str, List[List[List[int]]]]:
    """
    Verify that a sequence of moves solves the problem.

    Args:
        initial_state: Starting state
        moves: List of moves
        n_disks: Number of disks

    Returns:
        Tuple of (is_valid, error_message, state_trajectory)
        state_trajectory contains all states including initial and final
    """
    # Check initial state
    valid, msg = is_valid_state(initial_state, n_disks)
    if not valid:
        return False, f"Invalid initial state: {msg}", []

    state_trajectory = [initial_state]
    current_state = initial_state

    # Apply each move
    for i, move in enumerate(moves):
        # Check move validity
        valid, msg = is_valid_move(move, current_state, n_disks)
        if not valid:
            return False, f"Invalid move {i} {move}: {msg}", state_trajectory

        # Apply move
        current_state = apply_move(current_state, move)

        # Check resulting state
        valid, msg = is_valid_state(current_state, n_disks)
        if not valid:
            return False, f"Invalid state after move {i}: {msg}", state_trajectory

        state_trajectory.append(current_state)

    # Check if goal is reached
    if not is_goal_state(current_state, n_disks):
        return False, f"Did not reach goal state. Final: {current_state}", state_trajectory

    return True, "Solution is valid!", state_trajectory


def get_optimal_num_steps(n_disks: int) -> int:
    """
    Get the optimal number of steps to solve Towers of Hanoi with n disks.

    Args:
        n_disks: Number of disks

    Returns:
        Optimal number of steps (2^n - 1)
    """
    return (2 ** n_disks) - 1
