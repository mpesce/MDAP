"""
Agent module for LLM interaction.

Implements the agent templating function φ and extractors ψ_a and ψ_x
as described in Section 3.1 of the paper.
"""

from typing import Dict, Any, Tuple, List, Optional, Callable
from .parser import parse_move_state_flag


# Prompt templates from Appendix C of the paper
SYSTEM_PROMPT = """
You are a helpful assistant. Solve this puzzle for me.

There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
another stack.
3. A larger disk may not be placed on top of a smaller disk.

The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2,
1], [], []], and a solution might be:
moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]
This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.

Requirements:
- The positions are 0-indexed (the leftmost peg is 0).
- Ensure your answer includes a single next move in this EXACT FORMAT:
'''move = [disk id, from peg, to peg]'''
- Ensure your answer includes the next state resulting from applying the move to the current
state in this EXACT FORMAT:
'''next_state = [[...], [...], [...]]'''
"""

USER_TEMPLATE = """
Rules:
- Only one disk can be moved at a time.
- Only the top disk from any stack can be moved.
- A larger disk may not be placed on top of a smaller disk.

For all moves, follow the standard Tower of Hanoi procedure:
If the previous move did not move disk 1, move disk 1 clockwise one peg (0 -> 1 -> 2 -> 0).
If the previous move did move disk 1, make the only legal move that does not involve moving
disk1.

Use these clear steps to find the next move given the previous move and current state.

Previous move: {previous_move}
Current State: {current_state}

Based on the previous move and current state, find the single next move that follows the
procedure and the resulting next state.
"""


class TowersOfHanoiAgent:
    """
    Agent for solving Towers of Hanoi one step at a time.

    Implements Maximal Agentic Decomposition (MAD) where each agent
    handles exactly one step (m=1 in the paper's notation).
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str, Dict[str, Any]], str],
        n_disks: int = 20,
        max_tokens: int = 750,
        temperature: float = 0.1,
        parser_fn: Optional[Callable] = None
    ):
        """
        Initialize the agent.

        Args:
            llm_fn: Function that takes (system_prompt, user_prompt, kwargs) and returns LLM response
            n_disks: Number of disks in the problem
            max_tokens: Maximum output tokens (red flag threshold from Section 4.4)
            temperature: Sampling temperature
            parser_fn: Optional custom parser function (default: parse_move_state_flag)
        """
        self.llm_fn = llm_fn
        self.n_disks = n_disks
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.parser_fn = parser_fn or parse_move_state_flag

    def get_vote(
        self,
        current_state: List[List[int]],
        previous_move: List[int],
        temperature: Optional[float] = None
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Get a single vote (sample) for the next move.

        Implements Algorithm 3 (get_vote) from the paper.

        Args:
            current_state: Current disk configuration [[peg0], [peg1], [peg2]]
            previous_move: Previous move [disk_id, from_peg, to_peg]
            temperature: Optional override for sampling temperature

        Returns:
            Tuple of (next_move, next_state)

        Raises:
            ValueError: If response cannot be parsed (red-flagged)
        """
        # Template the prompt (function φ in the paper)
        user_prompt = USER_TEMPLATE.format(
            previous_move=previous_move,
            current_state=current_state
        )

        # Sample from LLM (M ◦ φ in the paper)
        temp = temperature if temperature is not None else self.temperature
        response = self.llm_fn(
            SYSTEM_PROMPT,
            user_prompt,
            {
                "max_tokens": self.max_tokens,
                "temperature": temp
            }
        )

        # Check for red flags (overly long responses already handled by max_tokens)
        # Parse and validate (ψ_a and ψ_x in the paper)
        try:
            move, next_state = self.parser_fn(response, self.n_disks)
        except ValueError as e:
            # Red flag: incorrectly formatted response
            raise ValueError(f"Red-flagged response: {e}") from e

        return move, next_state
