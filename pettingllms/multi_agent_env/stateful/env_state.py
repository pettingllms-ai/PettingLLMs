from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, ClassVar
import copy
import math
import random
import numpy as np
from collections import deque
from pettingllms.multi_agent_env.stateful.utils import generate_room

# =========================================================
# 1) Eight Queens (N-Queens) EnvState
# =========================================================

@dataclass
class EnvStateBase:
    tool_action: List[str] = field(default_factory=list, init=False)
    tool_code: str = field(default="", init=False)
    tool_execution_output: str = field(default="", init=False)
    plan_action: List[str] = field(default_factory=list, init=False)
    observation: str = field(default="", init=False)

    # History records for multi-turn interactions
    code_history: List[str] = field(default_factory=list, init=False)
    execution_history: List[str] = field(default_factory=list, init=False)
    plan_reasoning_history: List[str] = field(default_factory=list, init=False)
    plan_action_history: List[List[str]] = field(default_factory=list, init=False)
    tool_action_history: List[List[str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        if not hasattr(self, 'tool_action'):
            self.tool_action = []
        if not hasattr(self, 'tool_code'):
            self.tool_code = ""
        if not hasattr(self, 'tool_execution_output'):
            self.tool_execution_output = ""
        if not hasattr(self, 'plan_action'):
            self.plan_action = []
        if not hasattr(self, 'observation'):
            self.observation = ""
        # Initialize history lists
        if not hasattr(self, 'code_history'):
            self.code_history = []
        if not hasattr(self, 'execution_history'):
            self.execution_history = []
        if not hasattr(self, 'plan_reasoning_history'):
            self.plan_reasoning_history = []
        if not hasattr(self, 'plan_action_history'):
            self.plan_action_history = []
        if not hasattr(self, 'tool_action_history'):
            self.tool_action_history = []
    
    def __str__(self) -> str:
        return (
            f"tool_action: {self.tool_action}\n"
            f"tool_code: {self.tool_code}\n"
            f"tool_execution_output: {self.tool_execution_output}\n"
            f"plan_action: {self.plan_action}\n"
            f"observation: {self.observation}"
        )
    
    def __repr__(self) -> str:
        return self.__str__()

    @property
    def status(self) -> str:
        """Unified status view based on 'done'.
        Returns "done" when the environment is finished, otherwise "in_progress".
        """
        return "done" if getattr(self, 'done', False) else "in_progress"

    def to_dict_compact(self, agent_name: str = None) -> Dict[str, Any]:
        """Compact, logging-friendly snapshot across all env states.
        Always includes a unified 'status' and 'done' flag; conditionally includes
        a few common progress/reward fields when present.
        """
        compact: Dict[str, Any] = {
            "status": self.status,
            "done": getattr(self, 'done', False),
        }
        # Common optional fields across different envs
        for key in [
            "reward", "total_reward", "step_count", "steps", "invalid_count","tool_action", "tool_code", "tool_execution_output", "plan_action", "observation","tool_reward",
            "boxes_on_goals",
        ]:
            if hasattr(self, key):
                compact[key] = getattr(self, key)
        return compact

    def to_dict(self) -> Dict[str, Any]:
        """Broad snapshot of state with best-effort serialization.
        This method attempts to serialize fields into JSON-friendly types.
        """
        import json
        result: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            # Skip private/cached internals if any
            if k.startswith('_'):
                continue
            try:
                json.dumps(v)
                result[k] = v
            except Exception:
                # Numpy arrays or other non-serializable objects
                try:
                    import numpy as _np  # local import to avoid top-level dependency assumption
                    if isinstance(v, _np.ndarray):
                        result[k] = v.tolist()
                    else:
                        result[k] = str(v)
                except Exception:
                    result[k] = str(v)

        # Add unified view
        result["status"] = self.status
        result["done"] = getattr(self, 'done', False)
        return result

@dataclass
class EightQueensEnvState(EnvStateBase):
    """N-Queens problem: Place N queens on NxN board without attacking each other"""
    
    N: int = 8
    cols: List[int] = field(default_factory=lambda: [-1] * 8)
    positions: List[int] = field(default_factory=lambda: [-1] * 8)
    done: bool = False
    step_count: int = 0
    reward: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.cols = [-1] * self.N
        self.positions = [-1] * self.N
        self.reset()

    def reset(self):
        """Reset environment"""
        self.cols = [-1] * self.N
        self.positions = self.cols[:]
        self.done = False
        self.step_count = 0
        self.reward = 0.0
        self.observation = self.text_observation()
    
    def text_observation(self) -> str:
        """Text observation"""
        board = []
        for r in range(self.N):
            row = ['.'] * self.N
            if self.cols[r] >= 0:
                row[self.cols[r]] = 'Q'
            board.append(''.join(row))
        return '\n'.join(board)
    
    def available_actions(self) -> List[Tuple[int, int]]:
        """Available actions: (row, col), col=-1 means clear the row"""
        actions = []
        for r in range(self.N):
            for c in range(-1, self.N):
                actions.append((r, c))
        return actions
    
    def _conflicts(self, cols: List[int]) -> int:
        """Calculate number of conflicts"""
        count = 0
        for r1 in range(len(cols)):
            c1 = cols[r1]
            if c1 < 0:
                continue
            for r2 in range(r1 + 1, len(cols)):
                c2 = cols[r2]
                if c2 < 0:
                    continue
                # Same column or diagonal
                if c1 == c2 or abs(c1 - c2) == abs(r1 - r2):
                    count += 1
        return count
    
    def _is_solved(self) -> bool:
        """Check if solved"""
        return all(c >= 0 for c in self.cols) and self._conflicts(self.cols) == 0

    def step(self, action):
        """Execute action and update environment state. Action format: [col1, col2, ..., colN] JSON array"""
        if self.done:
            self.reward = 0.0
            return

        # Parse action: expect list with N column indices
        if not isinstance(action, list) or len(action) != self.N:
            self.reward = 0.0  # Invalid action format - failure
            return

        # Check all column indices are valid
        for col in action:
            if not isinstance(col, int) or not (0 <= col < self.N):
                self.reward = 0.0  # Invalid column index - failure
                return

        # Record previous state
        prev_conflicts = self._conflicts(self.cols)

        # Set new queen positions
        self.cols = list(action)
        self.positions = self.cols[:]

        # Calculate reward - binary: 0 for failure, 1 for success
        self.reward = 0.0  # Default to failure

        # Check completion
        if self._is_solved():
            self.reward = 1.0  # Success
            self.done = True

        self.step_count += 1
        self.observation = self.text_observation()



# =========================================================
# 2) Blocksworld EnvState
# =========================================================

@dataclass
class BlocksworldEnvState(EnvStateBase):
    """Blocksworld: Move blocks to specified stack configuration"""
    
    init_stacks: List[List[str]]
    goal_stacks: List[List[str]]
    stacks: List[List[str]] = field(default_factory=list)
    current_stacks: List[List[str]] = field(default_factory=list)
    all_blocks: Set[str] = field(default_factory=set)
    done: bool = False
    step_count: int = 0
    reward: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.init_stacks = [list(s) for s in self.init_stacks]
        self.goal_stacks = [list(s) for s in self.goal_stacks]
        self.all_blocks = set(sum(self.init_stacks, []))
        self.reset()

    def reset(self):
        """Reset environment"""
        self.stacks = [list(s) for s in self.init_stacks]
        self.current_stacks = [list(s) for s in self.stacks]
        self.done = False
        self.step_count = 0
        self.reward = 0.0
        self.observation = self.text_observation()
    
    def text_observation(self) -> str:
        """Text observation"""
        obs = []
        for i, stack in enumerate(self.stacks):
            if stack:
                obs.append(f"Stack {i}: {' -> '.join(stack)}")
            else:
                obs.append(f"Stack {i}: empty")
        obs.append(f"Goal: {self.goal_stacks}")
        return '\n'.join(obs)
    
    def available_actions(self) -> List[Tuple[str, str]]:
        """Available actions: (block, destination), destination can be block name or 'table'"""
        actions = []
        # Find all movable blocks (top blocks)
        clear_blocks = []
        for stack in self.stacks:
            if stack:
                clear_blocks.append(stack[-1])
        
        for block in clear_blocks:
            # Move to table
            actions.append((block, "table"))
            # Move to other blocks
            for other_block in clear_blocks:
                if other_block != block:
                    actions.append((block, other_block))
        return actions
    
    def _is_clear(self, block: str) -> bool:
        """Check if block is on top of stack"""
        for stack in self.stacks:
            if stack and stack[-1] == block:
                return True
        return False
    
    def _find_block(self, block: str) -> Tuple[int, int]:
        """Find block position: (stack_index, position_in_stack)"""
        for si, stack in enumerate(self.stacks):
            for pi, b in enumerate(stack):
                if b == block:
                    return si, pi
        raise ValueError(f"Block {block} not found")
    
    def _is_goal_reached(self) -> bool:
        """Check if goal is reached"""
        # Simple comparison, ignore empty stacks
        current = [stack for stack in self.stacks if stack]
        goal = [stack for stack in self.goal_stacks if stack]
        return sorted([tuple(s) for s in current]) == sorted([tuple(s) for s in goal])
    
    def _calculate_goal_similarity(self) -> float:
        """Calculate similarity between current and goal configuration (0-1)"""
        total_blocks = len(self.all_blocks)
        correct_positions = 0
        
        # Check if each block is in correct relative position
        for block in self.all_blocks:
            current_pos = self._get_block_context(block, self.stacks)
            goal_pos = self._get_block_context(block, self.goal_stacks)
            
            if current_pos == goal_pos:
                correct_positions += 1
        
        return correct_positions / total_blocks if total_blocks > 0 else 0.0
    
    def _get_block_context(self, block: str, stacks: List[List[str]]) -> Tuple:
        """Get block context: (block_below, block_above)"""
        for stack in stacks:
            if block in stack:
                idx = stack.index(block)
                below = stack[idx-1] if idx > 0 else None
                above = stack[idx+1] if idx < len(stack)-1 else None
                return (below, above)
        return (None, None)

    def step(self, action):
        """Execute action and update environment state. Action format: [{"move": ["B","table"]}, {"move": ["C","B"]}] JSON array"""
        if self.done:
            self.reward = 0.0
            return

        # Parse action: expect list of move operation dictionaries
        if not isinstance(action, list):
            self.reward = 0.0  # Invalid action format - failure
            return

        # Execute each action
        action_valid = True
        for move_action in action:
            if not isinstance(move_action, dict) or "move" not in move_action:
                action_valid = False
                break

            move = move_action["move"]
            if not isinstance(move, list) or len(move) != 2:
                action_valid = False
                break

            block, dest = move

            # Check if block is movable (on top of stack)
            if not self._is_clear(block):
                action_valid = False
                break

            # Execute move
            try:
                stack_idx, pos = self._find_block(block)

                # Check if destination is valid
                if dest != "table" and (dest not in self.all_blocks or not self._is_clear(dest)):
                    action_valid = False
                    break

                # Execute move
                self.stacks[stack_idx].pop()  # Remove block

                if dest == "table":
                    # Move to table (create new stack)
                    self.stacks.append([block])
                else:
                    # Move to other block
                    dest_stack_idx, _ = self._find_block(dest)
                    self.stacks[dest_stack_idx].append(block)

                # Update current_stacks attribute
                self.current_stacks = [list(s) for s in self.stacks]

            except ValueError:
                action_valid = False
                break

            self.step_count += 1

        # Binary reward: 0 for failure, 1 for success
        if not action_valid:
            self.reward = 0.0  # Invalid action - failure
        elif self._is_goal_reached():
            self.reward = 1.0  # Success
            self.done = True
        else:
            self.reward = 0.0  # Not yet solved - failure

        self.current_stacks = [list(s) for s in self.stacks]  # Update current_stacks attribute
        self.observation = self.text_observation()


# =========================================================
# 3) Sudoku 4x4 EnvState
# =========================================================

@dataclass
class SudukuEnvState(EnvStateBase):
    """Dynamic size Sudoku: Fill NxN grid satisfying row, column and sub-grid constraints (keep 4x4 name for compatibility)"""
    
    puzzle: Optional[List[List[int]]] = None
    seed: Optional[int] = None
    size: int = 4  # Sudoku size, default 4x4
    config: Optional[dict] = None
    init_grid: List[List[int]] = field(default_factory=list)
    grid: List[List[int]] = field(default_factory=list)
    done: bool = False
    step_count: int = 0
    reward: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        
        # Read map_size parameter from config if exists
        if self.config and hasattr(self.config, 'env') and hasattr(self.config.env, 'map_size'):
            try:
                self.size = int(self.config.env.map_size)
            except Exception:
                self.size = self.config.env.map_size
        elif self.config and isinstance(self.config, dict) and 'env' in self.config and 'map_size' in self.config['env']:
            try:
                self.size = int(self.config['env']['map_size'])
            except Exception:
                self.size = self.config['env']['map_size']
        if self.size is None:
            self.size = 16
        
        # If puzzle provided, use directly
        if self.puzzle is not None:
            assert len(self.puzzle) == self.size and all(len(row) == self.size for row in self.puzzle), f"Must be {self.size}x{self.size} grid"
            self.init_grid = [row[:] for row in self.puzzle]
        # If seed provided, generate puzzle based on seed
        elif self.seed is not None:
            self.puzzle = self._generate_puzzle_from_seed(self.seed, self.size)
            self.init_grid = [row[:] for row in self.puzzle]
        else:
            # Use default puzzle
            self.puzzle = self._get_default_puzzle(self.size)
            self.init_grid = [row[:] for row in self.puzzle]
            
        self.puzzle = [row[:] for row in self.puzzle]  # Update puzzle attribute
        self.reset()
    
    def _generate_puzzle_from_seed(self, seed: int, size: int) -> List[List[int]]:
        """Generate NxN Sudoku puzzle from pre-generated JSON file based on seed"""
        import json
        import os
        import random
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Build JSON file path
        json_file_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "..", 
            "datasets", "sudoku_environments", 
            f"sudoku_{size}x{size}.json"
        )
        
        try:
            # Read pre-generated environments
            with open(json_file_path, 'r', encoding='utf-8') as f:
                environments = json.load(f)
            
            if not environments:
                print(f"[WARN] JSON file {json_file_path} is empty, using default puzzle")
                return self._get_default_puzzle(size)
            
            # Select environment based on seed (ensure reproducibility)
            env_index = seed % len(environments)
            selected_env = environments[env_index]
            
            # Return selected puzzle
            puzzle = selected_env["puzzle"]
            
            # Validate puzzle size
            if len(puzzle) != size or any(len(row) != size for row in puzzle):
                print(f"[WARN] Selected puzzle size mismatch, expected {size}x{size}, actual {len(puzzle)}x{len(puzzle[0]) if puzzle else 0}")
                return self._get_default_puzzle(size)
            
            return puzzle
            
        except FileNotFoundError:
            print(f"[WARN] Pre-generated environment file not found: {json_file_path}")
            print(f"[INFO] Using default puzzle instead")
            return self._get_default_puzzle(size)
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON file parsing error: {e}")
            return self._get_default_puzzle(size)
            
        except Exception as e:
            print(f"[ERROR] Error reading environment file: {e}")
            return self._get_default_puzzle(size)
    
    
    def _get_default_puzzle(self, size: int) -> List[List[int]]:
        """Get default NxN Sudoku puzzle"""
        # Use different default puzzles for different sizes
        if size == 4:
            return [[1, 0, 0, 4],
                    [0, 0, 1, 0], 
                    [0, 4, 0, 0],
                    [4, 0, 0, 1]]
        elif size == 9:
            return [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9]
            ]
        elif size == 16:
            # Default 16x16 Sudoku puzzle (simplified)
            puzzle = [[0 for _ in range(16)] for _ in range(16)]
            # Fill some basic numbers
            puzzle[0] = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            puzzle[1] = [0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
            return puzzle
        else:
            # For other sizes, create a basic empty puzzle
            puzzle = [[0 for _ in range(size)] for _ in range(size)]
            # Fill first row with some numbers to make it solvable
            for i in range(min(size, 4)):
                puzzle[0][i] = i + 1
            return puzzle

    def reset(self):
        """Reset environment"""
        self.grid = [row[:] for row in self.init_grid]
        self.puzzle = [row[:] for row in self.init_grid]  # Update puzzle attribute
        self.done = False
        self.step_count = 0
        self.reward = 0.0
        self.observation = self.text_observation()
    
    def text_observation(self) -> str:
        """Text observation"""
        obs = []
        for row in self.grid:
            obs.append(' '.join(str(x) if x != 0 else '.' for x in row))
        return '\n'.join(obs)
    
    def available_actions(self) -> List[Tuple[int, int, int]]:
        """Available actions: (row, col, value)"""
        actions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == 0:  # Only fill empty cells
                    for v in range(1, self.size + 1):
                        actions.append((r, c, v))
        return actions
    
    def _is_valid_placement(self, r: int, c: int, v: int) -> bool:
        """Check if placing v at (r,c) is valid"""
        # Type conversion and validation
        try:
            r = int(r)
            c = int(c) 
            v = int(v)
        except (ValueError, TypeError):
            return False
            
        if not (0 <= r < self.size and 0 <= c < self.size and 1 <= v <= self.size):
            return False
        
        if self.grid[r][c] != 0:  # Non-empty cell
            return False
        
        # Check row
        if v in self.grid[r]:
            return False
        
        # Check column
        if any(self.grid[rr][c] == v for rr in range(self.size)):
            return False
        
        # Check sub-grid
        box_size = int(self.size ** 0.5)
        box_r, box_c = (r // box_size) * box_size, (c // box_size) * box_size
        for rr in range(box_r, box_r + box_size):
            for cc in range(box_c, box_c + box_size):
                if self.grid[rr][cc] == v:
                    return False
        
        return True
    
    def _is_solved(self) -> bool:
        """Check if solved"""
        # Check if filled
        for row in self.grid:
            if 0 in row:
                return False
        
        # Check rules
        for r in range(self.size):
            for c in range(self.size):
                v = self.grid[r][c]
                # Temporarily clear to check uniqueness
                self.grid[r][c] = 0
                valid = self._is_valid_placement(r, c, v)
                self.grid[r][c] = v
                if not valid:
                    return False
        return True
    
    def _calculate_progress(self) -> float:
        """Calculate solving progress (0-1)"""
        total_cells = self.size * self.size
        filled_cells = sum(1 for r in range(self.size) for c in range(self.size) if self.grid[r][c] != 0)
        return filled_cells / total_cells
    
    def _count_constraints_satisfied(self) -> int:
        """Count satisfied constraints"""
        satisfied = 0
        box_size = int(self.size ** 0.5)
        
        # Check row constraints
        for r in range(self.size):
            values = [self.grid[r][c] for c in range(self.size) if self.grid[r][c] != 0]
            if len(values) == len(set(values)):  # No duplicates
                satisfied += len(values) - 1 if len(values) > 1 else 0
        
        # Check column constraints  
        for c in range(self.size):
            values = [self.grid[r][c] for r in range(self.size) if self.grid[r][c] != 0]
            if len(values) == len(set(values)):  # No duplicates
                satisfied += len(values) - 1 if len(values) > 1 else 0
        
        # Check sub-grid constraints
        for box_r in range(0, self.size, box_size):
            for box_c in range(0, self.size, box_size):
                values = []
                for r in range(box_r, box_r + box_size):
                    for c in range(box_c, box_c + box_size):
                        if self.grid[r][c] != 0:
                            values.append(self.grid[r][c])
                if len(values) == len(set(values)):  # No duplicates
                    satisfied += len(values) - 1 if len(values) > 1 else 0
        
        return satisfied
    
    def _get_possible_values(self, r: int, c: int) -> Set[int]:
        """Get possible values for position (r,c)"""
        if self.grid[r][c] != 0:
            return set()
        
        possible = set(range(1, self.size + 1))
        box_size = int(self.size ** 0.5)
        
        # Exclude values in same row
        for cc in range(self.size):
            if self.grid[r][cc] in possible:
                possible.remove(self.grid[r][cc])
        
        # Exclude values in same column
        for rr in range(self.size):
            if self.grid[rr][c] in possible:
                possible.remove(self.grid[rr][c])
        
        # Exclude values in same sub-grid
        box_r, box_c = (r // box_size) * box_size, (c // box_size) * box_size
        for rr in range(box_r, box_r + box_size):
            for cc in range(box_c, box_c + box_size):
                if self.grid[rr][cc] in possible:
                    possible.remove(self.grid[rr][cc])
        
        return possible

    def step(self, action):
        """Execute action and update environment state. Action format: complete NxN grid or fill step list [[r,c,v],...]"""
        if self.done:
            self.reward = 0.0
            return

        # Check action format
        if isinstance(action, list) and len(action) == self.size and all(isinstance(row, list) and len(row) == self.size for row in action):
            # Format 1: Complete NxN grid [[1,2,3,4],[3,4,1,2],...]
            new_grid = action

            # Validate grid format and perform type conversion
            try:
                converted_grid = []
                for row in new_grid:
                    converted_row = []
                    for val in row:
                        int_val = int(val)
                        if not (1 <= int_val <= self.size):
                            self.reward = 0.0  # Invalid grid value - failure
                            return
                        converted_row.append(int_val)
                    converted_grid.append(converted_row)
                new_grid = converted_grid
            except (ValueError, TypeError):
                self.reward = 0.0  # Invalid grid value - failure
                return

            # Update grid
            self.grid = [row[:] for row in new_grid]
            self.puzzle = [row[:] for row in self.grid]  # Update puzzle attribute

            # Binary reward: 0 for failure, 1 for success
            if self._is_solved():
                self.reward = 1.0  # Success
                self.done = True
            else:
                self.reward = 0.0  # Not solved - failure

        elif isinstance(action, list) and all(isinstance(step, list) and len(step) == 3 for step in action):
            # Format 2: Fill step list [[r,c,v], [r,c,v], ...]
            action_valid = True

            for step in action:
                try:
                    r, c, v = step
                    # Ensure correct types
                    r, c, v = int(r), int(c), int(v)
                except (ValueError, TypeError, IndexError):
                    # Skip invalid steps
                    continue

                if self._is_valid_placement(r, c, v):
                    self.grid[r][c] = v
                else:
                    action_valid = False
                    break

                self.step_count += 1

            # Binary reward: 0 for failure, 1 for success
            if not action_valid:
                self.reward = 0.0  # Invalid action - failure
            elif self._is_solved():
                self.reward = 1.0  # Success
                self.done = True
            else:
                self.reward = 0.0  # Not yet solved - failure

        else:
            self.reward = 0.0  # Invalid action format - failure
            return
        
        self.step_count += 1
        self.puzzle = [row[:] for row in self.grid]  # Update puzzle attribute
        self.observation = self.text_observation()
    
    def _is_grid_valid(self) -> bool:
        """Check if current grid violates Sudoku rules"""
        box_size = int(self.size ** 0.5)
        
        # Check rows
        for r in range(self.size):
            values = [self.grid[r][c] for c in range(self.size) if self.grid[r][c] != 0]
            if len(values) != len(set(values)):
                return False
        
        # Check columns
        for c in range(self.size):
            values = [self.grid[r][c] for r in range(self.size) if self.grid[r][c] != 0]
            if len(values) != len(set(values)):
                return False
        
        # Check sub-grids
        for box_r in range(0, self.size, box_size):
            for box_c in range(0, self.size, box_size):
                values = []
                for r in range(box_r, box_r + box_size):
                    for c in range(box_c, box_c + box_size):
                        if self.grid[r][c] != 0:
                            values.append(self.grid[r][c])
                if len(values) != len(set(values)):
                    return False
        
        return True


@dataclass
class PlanPathGridEnvState(EnvStateBase):
    """
    2D grid path planning worker (BFS baseline) + action/reward interface.
    - Grid: '.' passable, '#' impassable
    - Actions: U/D/L/R (4-neighborhood)
    - Usage: step-by-step interaction: reset_agent() -> step(action_list) ... -> done
    - Action format: action sequence ["R", "R", "D", "D"]
    """
    
    seed: int
    grid_h: int = 10
    grid_w: int = grid_h
    block_ratio: float = 0.22
    r_step: Optional[float] = None
    r_invalid: Optional[float] = None
    r_goal: Optional[float] = None
    r_opt: Optional[float] = None
    r_fail: Optional[float] = None
    gamma: Optional[float] = None
    lambda_pot: Optional[float] = None
    max_steps: Optional[int] = None
    config: Optional[dict] = None
    
    # Environment state attributes
    grid: str = ""
    grid_list: List[str] = field(default_factory=list)
    h: int = 0
    w: int = 0
    start: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    goal: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    _shortest_path_cache: Optional[List[Tuple[int, int]]] = None
    
    # Step-by-step interaction state
    pos: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    done: bool = False
    steps: int = 0
    step_count: int = 0
    invalid_count: int = 0
    total_reward: float = 0.0
    reward: float = 0.0
    _last_phi: float = 0.0

    # ====== Default reward coefficients (can be overridden in __init__) ======
    DEFAULT_R_STEP: ClassVar[float] = -0.01   # Light penalty per step
    DEFAULT_R_INVALID: ClassVar[float] = -0.10   # Illegal action penalty (out of bounds/hit wall/non-adjacent)
    DEFAULT_R_GOAL: ClassVar[float] = +1.00   # Reach goal reward
    DEFAULT_R_OPT: ClassVar[float] = +0.50   # Shortest path bonus (if optimal)
    DEFAULT_R_FAIL: ClassVar[float] = -1.00   # Failure (terminated but not reached)/infeasible
    DEFAULT_GAMMA: ClassVar[float] = 0.99    # Discount only for shaping
    DEFAULT_LAMBDA_POT: ClassVar[float] = 1.00    # Shaping coefficient
    DEFAULT_MAX_STEPS: ClassVar[int] = 10_000  # Upper limit (prevent infinite loops)

    ACTIONS: ClassVar[Dict[str, Tuple[int,int]]] = {
        "U": (-1, 0),
        "D": (+1, 0),
        "L": ( 0,-1),
        "R": ( 0,+1),
    }

    def __post_init__(self):
        super().__post_init__()
        # Read map_size parameter from config if exists
        if self.config and hasattr(self.config, 'env') and hasattr(self.config.env, 'map_size'):
            try:
                self.grid_h = int(self.config.env.map_size)
                self.grid_w = int(self.config.env.map_size)
            except Exception:
                self.grid_h = self.config.env.map_size
                self.grid_w = self.config.env.map_size
        elif self.config and isinstance(self.config, dict) and 'env' in self.config and 'map_size' in self.config['env']:
            try:
                self.grid_h = int(self.config['env']['map_size'])
                self.grid_w = int(self.config['env']['map_size'])
            except Exception:
                self.grid_h = self.config['env']['map_size']
                self.grid_w = self.config['env']['map_size']
        
        # Generate random environment based on seed
        grid, start, goal = self._generate_random_environment(self.seed, self.grid_h, self.grid_w, self.block_ratio)
        
        # Map/basic
        self.grid = '\n'.join(grid)  # String format for prompt
        self.grid_list = grid  # Keep list format for internal use
        self.h = len(grid)
        self.w = len(grid[0]) if self.h > 0 else 0
        self.start = tuple(start)
        self.goal = tuple(goal)
        self._shortest_path_cache = None

        # Reward parameters
        self.r_step     = self.DEFAULT_R_STEP     if self.r_step     is None else self.r_step
        self.r_invalid  = self.DEFAULT_R_INVALID  if self.r_invalid  is None else self.r_invalid
        self.r_goal     = self.DEFAULT_R_GOAL     if self.r_goal     is None else self.r_goal
        self.r_opt      = self.DEFAULT_R_OPT      if self.r_opt      is None else self.r_opt
        self.r_fail     = self.DEFAULT_R_FAIL     if self.r_fail     is None else self.r_fail
        self.gamma      = self.DEFAULT_GAMMA      if self.gamma      is None else self.gamma
        self.lambda_pot = self.DEFAULT_LAMBDA_POT if self.lambda_pot is None else self.lambda_pot
        self.max_steps  = self.DEFAULT_MAX_STEPS  if self.max_steps  is None else self.max_steps

        # Step-by-step interaction state
        self.reset_agent()
        
        # Add attributes for new step method
        self.reward = 0.0
        self.done = False
        self.step_count = 0
        self.observation = self.text_observation()
    
    def _generate_random_environment(self, seed: int, grid_h: int, grid_w: int, block_ratio: float) -> Tuple[List[str], Tuple[int, int], Tuple[int, int]]:
        """Generate random grid, start and goal based on seed"""
        # Set random seed for reproducibility
        rng = random.Random(seed)
        np.random.seed(seed)
        
        max_trials = max(2000, 50)  # Maximum number of attempts
        for _ in range(max_trials):
            # Generate random grid (0=passable, 1=obstacle)
            grid_array = (np.random.rand(grid_h, grid_w) < block_ratio).astype(int)
            
            # Find all passable positions
            free_positions = [(r, c) for r in range(grid_h) for c in range(grid_w) if grid_array[r, c] == 0]
            
            if len(free_positions) < 2:  # Need at least two passable positions
                continue
                
            # Randomly select start and goal
            start = rng.choice(free_positions)
            goal = rng.choice(free_positions)
            while goal == start:  # Ensure start and goal are different
                goal = rng.choice(free_positions)
            
            # Check if path exists from start to goal
            if self._bfs_check_reachable(grid_array, start, goal):
                # Convert numpy array to string list format
                grid_str = []
                for row in grid_array:
                    row_str = ''.join('.' if cell == 0 else '#' for cell in row)
                    grid_str.append(row_str)
                
                return grid_str, start, goal
        
        # If unable to generate valid environment, create simple default environment
        print(f"[WARN] Unable to generate valid environment for seed {seed}, using default environment")
        return self._create_default_environment(grid_h, grid_w)
    
    def _bfs_check_reachable(self, grid_array: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Use BFS to check if goal is reachable from start"""
        h, w = grid_array.shape
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
                
            # Check four-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and 
                    grid_array[nr, nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return False
    
    def _create_default_environment(self, grid_h: int, grid_w: int) -> Tuple[List[str], Tuple[int, int], Tuple[int, int]]:
        """Create simple default environment (all passable)"""
        grid = ['.' * grid_w for _ in range(grid_h)]
        start = (0, 0)
        goal = (grid_h - 1, grid_w - 1)
        return grid, start, goal
    
    def text_observation(self) -> str:
        """Text observation"""
        obs_lines = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                if (r, c) == self.pos:
                    row += "S"  # Current position
                elif (r, c) == self.goal:
                    row += "G"  # Goal position
                else:
                    row += self.grid_list[r][c]
            obs_lines.append(row)
        return "\n".join(obs_lines)

    # ============== Geometry/Graph Search ==============
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def passable(self, r: int, c: int) -> bool:
        return self.grid_list[r][c] != '#'

    def neighbors(self, r: int, c: int):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.passable(nr, nc):
                yield (nr, nc)

    def shortest_path(self) -> Optional[List[Tuple[int, int]]]:
        """BFS find shortest path (including start and goal); return None if unreachable"""
        if self._shortest_path_cache is not None:
            return self._shortest_path_cache
        from collections import deque
        q = deque([self.start])
        prev: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {self.start: None}
        while q:
            cur = q.popleft()
            if cur == self.goal:
                path = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = prev[node]
                path.reverse()
                self._shortest_path_cache = path
                return path
            for nxt in self.neighbors(*cur):
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        return None


    # ============== Representation/Description ==============

    def describe(self) -> str:
        return (
            "PlanPathGridWorker: 2D grid shortest-path (BFS). "
            "'.' passable, '#' blocked; moves: U/D/L/R (4-neighborhood)."
        )

    # ============== Action Interface (Step-by-step Interaction) ===============
    def reset_agent(self):
        """Reset step-by-step interaction state"""
        self.pos: Tuple[int,int] = self.start
        self.done: bool = False
        self.steps: int = 0
        self.step_count: int = 0
        self.invalid_count: int = 0
        self.total_reward: float = 0.0
        self.reward: float = 0.0
        self._last_phi: float = self._potential(self.pos)
        self.observation = self.text_observation()

    def get_valid_actions(self, pos: Optional[Tuple[int,int]] = None) -> List[str]:
        """Return valid action set for current position (excluding out of bounds/hit wall)"""
        if pos is None: pos = self.pos
        valid = []
        for a, (dr, dc) in self.ACTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if self.in_bounds(nr, nc) and self.passable(nr, nc):
                valid.append(a)
        return valid

    def _potential(self, pos: Tuple[int,int]) -> float:
        """Potential: negative Manhattan distance (closer to goal = higher value)"""
        return - (abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1]))

    def _apply_action(self, pos: Tuple[int,int], action: str) -> Tuple[Tuple[int,int], bool]:
        """Try to apply action; return (next_pos, is_valid)"""
        if action not in self.ACTIONS:
            return pos, False
        dr, dc = self.ACTIONS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not self.in_bounds(nr, nc) or not self.passable(nr, nc):
            return pos, False
        return (nr, nc), True

    def step(self, action):
        """
        Execute action and update environment state. Action format:
        Action sequence: ["R", "R", "D", "D"]
        """
        if self.done:
            self.reward = 0.0
            return

        # Check action format
        if isinstance(action, list) and all(isinstance(item, str) for item in action):
            # Action sequence ["R", "R", "D", "D"]
            self._execute_action_sequence(action)
        else:
            self.reward = 0.0  # Invalid action format - failure

    def _execute_action_sequence(self, actions: List[str]):
        """Execute action sequence - binary reward"""
        for action in actions:
            pos, _, done, _ = self.step_single(action)
            if done:
                break
        # Binary reward: 0 for failure, 1 for success
        if self.done and self.pos == self.goal:
            self.reward = 1.0  # Success
        else:
            self.reward = 0.0  # Failure


    def step_single(self, action: str) -> Tuple[Tuple[int,int], float, bool, Dict[str,Any]]:
        """
        Execute single action step and return:
          next_pos, reward, done, info
        Reward = base step penalty/illegal penalty + potential-based shaping (+ termination reward/optimal bonus)
        """
        if self.done:
            return self.pos, 0.0, True, {"msg": "episode already done"}
        prev_pos = self.pos
        next_pos, valid = self._apply_action(prev_pos, action)

        # Base reward
        reward = 0.0
        if valid:
            reward += self.r_step
        else:
            reward += self.r_invalid
            self.invalid_count += 1

        # Shaping (doesn't change optimal policy)
        cur_phi = self._last_phi
        nxt_phi = self._potential(next_pos)
        shaping = self.lambda_pot * (self.gamma * nxt_phi - cur_phi)
        reward += shaping

        # State update
        self.pos = next_pos if valid else prev_pos
        self._last_phi = self._potential(self.pos)
        self.steps += 1

        # Termination check
        if self.pos == self.goal:
            # Reach goal
            reward += self.r_goal
            # Optimal bonus (if reachable)
            sp = self.shortest_path()
            if sp is not None:
                if self.steps == len(sp) - 1:  # Note: steps is "action steps", sp is "node count"
                    reward += self.r_opt
            self.done = True
        elif self.steps >= self.max_steps:
            # Timeout failure
            reward += self.r_fail
            self.done = True

        self.total_reward += reward
        info = {
            "valid": valid,
            "steps": self.steps,
            "invalid_count": self.invalid_count,
            "shaping": shaping,
            "pos": self.pos,
            "goal": self.goal,
            "done": self.done,
        }
        return self.pos, reward, self.done, info# =========================================================
# State Registry - Initialize before any register calls
# =========================================================

@dataclass
class SokobanGridEnvState(EnvStateBase):
    """
    Sokoban push boxes environment - push all boxes to target positions
    - Symbols:
      '#' wall (impassable)
      ' ' empty space (passable)
      '.' target position (boxes need to be pushed here)
      '$' box
      '@' player
      '*' box on target position
      '+' player on target position
    - Actions: U/D/L/R (4-neighborhood)
    - Win condition: all boxes are on target positions
    """
    
    seed: int
    level: int = 1  # Level number for selecting predefined levels
    r_step: Optional[float] = None
    r_invalid: Optional[float] = None
    r_box_on_goal: Optional[float] = None
    r_box_off_goal: Optional[float] = None
    r_win: Optional[float] = None
    r_fail: Optional[float] = None
    gamma: Optional[float] = None
    lambda_pot: Optional[float] = None
    max_steps: Optional[int] = None
    config: Optional[dict] = None
    
    # Environment state attributes
    grid: str = ""
    grid_list: List[str] = field(default_factory=list)
    h: int = 0
    w: int = 0
    player_pos: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    boxes: Set[Tuple[int, int]] = field(default_factory=set)  # Box positions
    goals: Set[Tuple[int, int]] = field(default_factory=set)  # Target positions
    walls: Set[Tuple[int, int]] = field(default_factory=set)  # Wall positions
    # Low-level state generated by utils.py (numerical grid)
    room_structure: Optional[np.ndarray] = field(default=None, init=False)  # Fixed structure: 0 wall/1 empty/2 target
    room_state: Optional[np.ndarray] = field(default=None, init=False)      # Variable state: contains player/boxes
    box_mapping: Dict[Tuple[int,int], Tuple[int,int]] = field(default_factory=dict, init=False)
    action_sequence: List[int] = field(default_factory=list, init=False)
    
    # Step-by-step interaction state
    done: bool = False
    steps: int = 0
    step_count: int = 0
    invalid_count: int = 0
    total_reward: float = 0.0
    reward: float = 0.0
    boxes_on_goals: int = 0  # Number of boxes on target positions
    _last_phi: float = 0.0

    # ====== Default reward coefficients ======
    DEFAULT_R_STEP: ClassVar[float] = -0.01      # Light penalty per step
    DEFAULT_R_INVALID: ClassVar[float] = -0.05   # Illegal action penalty
    DEFAULT_R_BOX_ON_GOAL: ClassVar[float] = +1.0   # Push box to target position
    DEFAULT_R_BOX_OFF_GOAL: ClassVar[float] = -0.5  # Box leaves target position
    DEFAULT_R_WIN: ClassVar[float] = +10.0       # Win reward
    DEFAULT_R_FAIL: ClassVar[float] = -5.0       # Failure penalty
    DEFAULT_GAMMA: ClassVar[float] = 0.99        # Discount factor
    DEFAULT_LAMBDA_POT: ClassVar[float] = 0.1    # Shaping coefficient
    DEFAULT_MAX_STEPS: ClassVar[int] = 200       # Maximum steps

    ACTIONS: ClassVar[Dict[str, Tuple[int,int]]] = {
        "U": (-1, 0),
        "D": (+1, 0),
        "L": ( 0,-1),
        "R": ( 0,+1),
    }
    
    # Remove old predefined levels, use utils.generate_room uniformly

    def __post_init__(self):
        super().__post_init__()
         # Read map_size parameter from config if exists
        self.size = None
        if self.config and hasattr(self.config, 'env') and hasattr(self.config.env, 'map_size'):
            try:
                self.size = int(self.config.env.map_size)
            except Exception:
                self.size = self.config.env.map_size
        elif self.config and isinstance(self.config, dict) and 'env' in self.config and 'map_size' in self.config['env']:
            try:
                self.size = int(self.config['env']['map_size'])
            except Exception:
                self.size = self.config['env']['map_size']
        
        if self.size is None:
            self.size = 16
        
        # ========== Generation logic based on utils.py: fixed 6x6 + 1 box ==========
        assert generate_room is not None, "Cannot import generate_room, please confirm utils.py is available"

        room_structure, room_state, box_mapping, action_sequence = generate_room(
            dim=(self.size, self.size),
            p_change_directions=0.35,
            num_steps=25,
            num_boxes=1,
            tries=20,  # Increased from 4 to 20 for better success rate
            second_player=False,
            search_depth=200,  # Increased from 100 to 200 for better initial state generation
            min_box_distance=2,  # Minimum distance between boxes and targets
            min_difficulty_score=1,  # Lower difficulty requirement to ease generation
            seed=self.seed,  # Pass seed to ensure reproducible room generation
        )

        # Save low-level grid and search information
        self.room_structure = room_structure
        self.room_state = room_state
        self.box_mapping = {tuple(k): tuple(v) for k, v in box_mapping.items()}
        self.action_sequence = list(action_sequence)

        # Parse numerical grid to sets/ASCII
        self.h, self.w = room_structure.shape
        self.walls = set(map(tuple, np.argwhere(room_structure == 0)))
        self.goals = set(map(tuple, np.argwhere(room_structure == 2)))

        player_pos_arr = np.argwhere(room_state == 5)
        if player_pos_arr.size == 0:
            empty_cells = np.argwhere(room_structure == 1)
            self.player_pos = tuple(empty_cells[0]) if empty_cells.size > 0 else (0, 0)
        else:
            self.player_pos = tuple(player_pos_arr[0])

        boxes_not_on_goal = set(map(tuple, np.argwhere(room_state == 4)))
        boxes_on_goal = set(map(tuple, np.argwhere(room_state == 3)))
        self.boxes = boxes_not_on_goal | boxes_on_goal

        grid_rows: List[str] = []
        for r in range(self.h):
            row_chars: List[str] = []
            for c in range(self.w):
                pos = (r, c)
                if pos in self.walls:
                    row_chars.append('#')
                elif pos == self.player_pos:
                    row_chars.append('+' if pos in self.goals else '@')
                elif pos in self.boxes:
                    row_chars.append('*' if pos in self.goals else '$')
                elif pos in self.goals:
                    row_chars.append('.')
                else:
                    row_chars.append(' ')
            grid_rows.append(''.join(row_chars))
        self.grid_list = grid_rows
        self.grid = '\n'.join(grid_rows)

        self.boxes_on_goals = len(self.boxes & self.goals)

        # Reward parameters
        self.r_step     = self.DEFAULT_R_STEP     if self.r_step     is None else self.r_step
        self.r_invalid  = self.DEFAULT_R_INVALID  if self.r_invalid  is None else self.r_invalid
        self.r_box_on_goal  = self.DEFAULT_R_BOX_ON_GOAL  if self.r_box_on_goal  is None else self.r_box_on_goal
        self.r_box_off_goal = self.DEFAULT_R_BOX_OFF_GOAL if self.r_box_off_goal is None else self.r_box_off_goal
        self.r_win      = self.DEFAULT_R_WIN      if self.r_win      is None else self.r_win
        self.r_fail     = self.DEFAULT_R_FAIL     if self.r_fail     is None else self.r_fail
        self.gamma      = self.DEFAULT_GAMMA      if self.gamma      is None else self.gamma
        self.lambda_pot = self.DEFAULT_LAMBDA_POT if self.lambda_pot is None else self.lambda_pot
        self.max_steps  = self.DEFAULT_MAX_STEPS  if self.max_steps  is None else self.max_steps

        # Step-by-step interaction initial state (don't regenerate level)
        self.done = False
        self.steps = 0
        self.step_count = 0
        self.invalid_count = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self._last_phi = self._potential()
        self.observation = self.text_observation()
    
    # Remove old level string parsing and generation methods (unified through utils.generate_room)

    def text_observation(self) -> str:
        """Text observation"""
        obs_lines = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                pos = (r, c)
                if pos in self.walls:
                    row += "#"
                elif pos == self.player_pos:
                    if pos in self.goals:
                        row += "+"  # Player on target position
                    else:
                        row += "@"  # Player
                elif pos in self.boxes:
                    if pos in self.goals:
                        row += "*"  # Box on target position
                    else:
                        row += "$"  # Box
                elif pos in self.goals:
                    row += "."  # Empty target position
                else:
                    row += " "  # Empty space
            obs_lines.append(row)
        return "\n".join(obs_lines)

    def describe(self) -> str:
        return (
            "SokobanGridWorker: Push boxes ($) to goal positions (.). "
            "Player (@) moves with U/D/L/R. Win when all boxes are on goals (*)."
        )

    def reset_agent(self):
        """Reset step-by-step interaction state to initial state generated by utils.generate_room"""
        assert self.room_structure is not None and self.room_state is not None

        # Restore sets/ASCII based on saved numerical grid
        room_structure = self.room_structure
        room_state = self.room_state

        self.h, self.w = room_structure.shape
        self.walls = set(map(tuple, np.argwhere(room_structure == 0)))
        self.goals = set(map(tuple, np.argwhere(room_structure == 2)))

        player_pos_arr = np.argwhere(room_state == 5)
        if player_pos_arr.size == 0:
            empty_cells = np.argwhere(room_structure == 1)
            self.player_pos = tuple(empty_cells[0]) if empty_cells.size > 0 else (0, 0)
        else:
            self.player_pos = tuple(player_pos_arr[0])

        boxes_not_on_goal = set(map(tuple, np.argwhere(room_state == 4)))
        boxes_on_goal = set(map(tuple, np.argwhere(room_state == 3)))
        self.boxes = boxes_not_on_goal | boxes_on_goal

        grid_rows: List[str] = []
        for r in range(self.h):
            row_chars: List[str] = []
            for c in range(self.w):
                pos = (r, c)
                if pos in self.walls:
                    row_chars.append('#')
                elif pos == self.player_pos:
                    row_chars.append('+' if pos in self.goals else '@')
                elif pos in self.boxes:
                    row_chars.append('*' if pos in self.goals else '$')
                elif pos in self.goals:
                    row_chars.append('.')
                else:
                    row_chars.append(' ')
            grid_rows.append(''.join(row_chars))
        self.grid_list = grid_rows
        self.grid = '\n'.join(grid_rows)

        self.done = False
        self.steps = 0
        self.step_count = 0
        self.invalid_count = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.boxes_on_goals = len(self.boxes & self.goals)
        self._last_phi = self._potential()
        self.observation = self.text_observation()

    def get_valid_actions(self, pos: Optional[Tuple[int,int]] = None) -> List[str]:
        """Return valid action set for current position"""
        if pos is None: 
            pos = self.player_pos
        valid = []
        for action, (dr, dc) in self.ACTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            next_pos = (nr, nc)
            
            # Check if out of bounds or hit wall
            if not self.in_bounds(nr, nc) or next_pos in self.walls:
                continue
                
            # Check if pushing box
            if next_pos in self.boxes:
                # Calculate box's next position
                box_nr, box_nc = nr + dr, nc + dc
                box_next_pos = (box_nr, box_nc)
                
                # Box cannot be pushed to wall or another box
                if (not self.in_bounds(box_nr, box_nc) or 
                    box_next_pos in self.walls or 
                    box_next_pos in self.boxes):
                    continue
            
            valid.append(action)
        return valid

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def _potential(self) -> float:
        """Potential function: based on distance from boxes to nearest goal"""
        if not self.boxes or not self.goals:
            return 0.0
        
        total_distance = 0.0
        for box in self.boxes:
            min_dist = min(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) 
                          for goal in self.goals)
            total_distance += min_dist
        
        # Return negative distance (smaller distance = higher potential)
        return -total_distance

    def _is_won(self) -> bool:
        """Check if won (all boxes are on target positions)"""
        return len(self.boxes & self.goals) == len(self.boxes)

    def _apply_action(self, action: str) -> Tuple[Tuple[int,int], bool, Dict[str,Any]]:
        """Try to apply action; return (next_player_pos, is_valid, info)"""
        if action not in self.ACTIONS:
            return self.player_pos, False, {"msg": "invalid action"}
        
        dr, dc = self.ACTIONS[action]
        nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
        next_pos = (nr, nc)
        
        # Check if out of bounds or hit wall
        if not self.in_bounds(nr, nc) or next_pos in self.walls:
            return self.player_pos, False, {"msg": "hit wall or out of bounds"}
        
        info = {"pushed_box": False, "box_on_goal_change": 0}
        
        # Check if pushing box
        if next_pos in self.boxes:
            # Calculate box's next position
            box_nr, box_nc = nr + dr, nc + dc
            box_next_pos = (box_nr, box_nc)
            
            # Box cannot be pushed to wall or another box
            if (not self.in_bounds(box_nr, box_nc) or 
                box_next_pos in self.walls or 
                box_next_pos in self.boxes):
                return self.player_pos, False, {"msg": "cannot push box"}
            
            # Push box
            self.boxes.remove(next_pos)
            self.boxes.add(box_next_pos)
            info["pushed_box"] = True
            
            # Check box on goal position change
            old_on_goal = next_pos in self.goals
            new_on_goal = box_next_pos in self.goals
            
            if old_on_goal and not new_on_goal:
                info["box_on_goal_change"] = -1  # Box leaves target position
            elif not old_on_goal and new_on_goal:
                info["box_on_goal_change"] = +1  # Box reaches target position
        
        return next_pos, True, info

    def step(self, action):
        """
        Execute action and update environment state. Action format:
        Action sequence: ["R", "R", "D", "D"]
        """
        if self.done:
            self.reward = 0.0
            return

        # Check action format
        if isinstance(action, list) and all(isinstance(item, str) for item in action):
            # Action sequence ["R", "R", "D", "D"]
            self._execute_action_sequence(action)
        else:
            self.reward = 0.0  # Invalid action format - failure

    def _execute_action_sequence(self, actions: List[str]):
        """Execute action sequence - binary reward"""
        for action in actions:
            pos, _, done, _ = self.step_single(action)
            if done:
                break
        # Binary reward: 0 for failure, 1 for success
        if self.done and self._is_won():
            self.reward = 1.0  # Success
        else:
            self.reward = 0.0  # Failure
    
    def step_single(self, action: str) -> Tuple[Tuple[int,int], float, bool, Dict[str,Any]]:
        """
        Execute single action step and return:
          next_pos, reward, done, info
        """
        if self.done:
            return self.player_pos, 0.0, True, {"msg": "episode already done"}
        
        prev_pos = self.player_pos
        prev_boxes_on_goals = len(self.boxes & self.goals)
        
        next_pos, valid, action_info = self._apply_action(action)

        # Base reward
        reward = 0.0
        if valid:
            reward += self.r_step
        else:
            reward += self.r_invalid
            self.invalid_count += 1

        # Box on goal position change reward
        if valid and action_info["pushed_box"]:
            if action_info["box_on_goal_change"] > 0:
                reward += self.r_box_on_goal
            elif action_info["box_on_goal_change"] < 0:
                reward += self.r_box_off_goal

        # Potential-based shaping
        cur_phi = self._last_phi
        nxt_phi = self._potential()
        shaping = self.lambda_pot * (self.gamma * nxt_phi - cur_phi)
        reward += shaping

        # State update
        if valid:
            self.player_pos = next_pos
        self._last_phi = self._potential()
        self.steps += 1
        
        # Update number of boxes on target positions
        self.boxes_on_goals = len(self.boxes & self.goals)

        # Termination check
        if self._is_won():
            # Win
            reward += self.r_win
            self.done = True
        elif self.steps >= self.max_steps:
            # Timeout failure
            reward += self.r_fail
            self.done = True

        self.total_reward += reward
        self.observation = self.text_observation()
        
        info = {
            "valid": valid,
            "steps": self.steps,
            "invalid_count": self.invalid_count,
            "shaping": shaping,
            "player_pos": self.player_pos,
            "boxes_on_goals": self.boxes_on_goals,
            "total_boxes": len(self.boxes),
            "done": self.done,
            "won": self._is_won() if self.done else False,
            **action_info
        }
        return self.player_pos, reward, self.done, info

STATE_REGISTRY = {
    "EightQueens": EightQueensEnvState,
    "Blocksworld": BlocksworldEnvState, 
    "suduku": SudukuEnvState,
    "plan_path": PlanPathGridEnvState,
    "sokoban": SokobanGridEnvState,  
}

def get_state_class_by_benchmark(benchmark_name: str):
    if benchmark_name not in STATE_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available benchmarks: {list(STATE_REGISTRY.keys())}"
        )
    return STATE_REGISTRY[benchmark_name]

