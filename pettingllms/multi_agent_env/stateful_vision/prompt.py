# prompt.py
from __future__ import annotations
import json
from typing import Any, List

def _truncate(s, length=300):
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= length else s[: length // 2] + "...(truncated)..." + s[-length // 2 :]


def prompt_plan_path(turn_idx: int, state: Any) -> str:
    """Generate prompt for plan path task"""
    grid = getattr(state, "grid", None) or []
    start = getattr(state, "start", None)
    goal = getattr(state, "goal", None)

    grid_str = grid
    return (
            "You are a path planner (grid walker).\n\n"
            f"The current state is shown in the provided image.\n"
            f"Grid ('.' free, '#' blocked):\n```\n{grid_str}\n```\n"
            f"Start: {start},  Goal: {goal}\n"
            "Rules: moves are U/D/L/R; stay within bounds; do not cross '#'.\n"
            "If already at goal, return an empty list.\n\n"
        )


def prompt_sokoban(turn_idx: int, state: Any) -> str:
    """Generate prompt for sokoban task"""
    observation = getattr(state, "observation", "") or getattr(state, "grid", "")
    boxes = getattr(state, "boxes", set())
    goals = getattr(state, "goals", set())
    boxes_on_goals = len(boxes & goals) if boxes and goals else 0
    total_boxes = len(boxes) if boxes else 0
    method_hints = (
        "Hints: \n"
        "- Enumerate pushable moves: to push a box at (br,bc) by (dr,dc), ensure player can reach (br-dr,bc-dc) and the destination (br+dr,bc+dc) is empty and not a wall/box.\n"
        "- Deadlock avoidance: avoid pushing boxes into corners or along walls where no goal exists; avoid locking two boxes side-by-side against walls.\n"
        "- If cannot solve fully, return a short valid push sequence.\n\n"
    )
    # Note: For sokoban, observation is now an image, but we still provide grid text for reference
    grid_text = ""
    if hasattr(state, 'grid') and state.grid:
        grid_text = f"Grid representation (for reference):\n```\n{state.grid}\n```\n"

    return (
        "You are solving a Sokoban puzzle (push boxes onto goals).\n"
        "The current state is shown in the provided image.\n"
        "Legend: '#' wall, ' ' floor, '.' goal, '$' box, '@' player, '*' box on goal, '+' player on goal.\n"
        "Moves: U/D/L/R (cannot walk through walls; to move a box, walk into it with empty space behind).\n"
        "Success: all boxes are on goals. If already solved, return an empty list.\n\n"
        f"Boxes on goals: {boxes_on_goals}/{total_boxes}\n"
        + grid_text
        + method_hints
    )


def prompt_code_sokoban(turn_idx: int, state: Any) -> str:
    """Generate code-based prompt for sokoban"""
    boxes = getattr(state, "boxes", set())
    goals = getattr(state, "goals", set())
    boxes_on_goals = len(boxes & goals) if boxes and goals else 0
    total_boxes = len(boxes) if boxes else 0

    # Get grid text for code generation
    grid_text = ""
    if hasattr(state, 'grid') and state.grid:
        grid_text = state.grid

    return (
        "Programming task: Implement a Sokoban solver.\n"
        "The current state is shown in the provided image.\n"
        "Implement solve_sokoban(observation: str) -> List[str] that returns a move sequence using only 'U','D','L','R'.\n\n"
        "Input:\n"
        "- observation: the level grid as a multiline string.\n"
        "  Legend: '#' wall, ' ' floor, '.' goal, '$' box, '@' player, '*' box on goal, '+' player on goal.\n"
        f"- Progress: {boxes_on_goals}/{total_boxes} boxes on goals.\n"
        "- Current state grid:\n```\n"
        f"{grid_text}\n"
        "```\n\n"
        "Your code must:\n"
        "1) print the final result.\n"
        "2) Define solve_sokoban(observation: str) -> List[str].\n"
        "3) Compute the actions from the given observation.\n"
        "4) Print the final result using EXACTLY one of these formats:\n"
        "   - **Actions List**: [\"U\",\"R\",\"D\",\"L\"]\n"
        "Rules:\n"
        "- The player moves U/D/L/R and cannot pass through walls; to push a box, you must move into it and the destination cell behind must be empty or a goal.\n"
        "- Success: all boxes are on goals.\n\n"
        "- Deadlock avoidance: avoid non-goal corners and bad wall locks.\n"
        "- If you cannot fully solve it, still print a short valid sequence that makes progress.\n\n"
        "template:\n"
        "```python\n"
        "```\n"
    )


PROMPT_BUILDERS = {
    "plan_path": prompt_plan_path,
    "sokoban": prompt_sokoban,
}

PROMPT_TOOL_CALL_BUILDERS = {
    "plan_path": prompt_plan_path,
    "sokoban": prompt_code_sokoban,
}

def build_plan_prompt(benchmark: str, turn_idx: int, state: Any) -> str:
    if benchmark not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(PROMPT_BUILDERS.keys())}")
    return PROMPT_BUILDERS[benchmark](turn_idx, state)

def build_tool_prompt(benchmark: str, turn_idx: int, state: Any) -> str:
    if benchmark not in PROMPT_TOOL_CALL_BUILDERS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(PROMPT_TOOL_CALL_BUILDERS.keys())}")
    return PROMPT_TOOL_CALL_BUILDERS[benchmark](turn_idx, state)
