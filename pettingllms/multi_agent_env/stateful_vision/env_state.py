from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, ClassVar
import copy
import math
import random
import numpy as np
from collections import deque
from pettingllms.multi_agent_env.stateful_vision.utils import generate_room
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
from pathlib import Path

# =========================================================
# Image Path Helper Functions
# =========================================================

def get_image_save_path(
    config: Optional[Any] = None,
    experiment_name: Optional[str] = None,
    step: Optional[int] = None,
    rollout_idx: Optional[int] = None,
    turn_idx: Optional[int] = None,
    agent_name: Optional[str] = None,
    base_dir: str = "tmp_image"
) -> Path:
    """
    Generate image save path following the structure:
    tmp_image/date/experiment_name/step/rolloutidx/turn_agent.png

    Args:
        config: Configuration object (to read base_dir if available)
        experiment_name: Name of the experiment
        step: Training step number
        rollout_idx: Rollout index
        turn_idx: Turn index
        agent_name: Agent name
        base_dir: Base directory for images (default: "tmp_image")

    Returns:
        Path object for the image file
    """
    # Read base_dir from config if available
    if config is not None:
        if hasattr(config, 'env') and hasattr(config.env, 'image_save_dir'):
            base_dir = config.env.image_save_dir
        elif hasattr(config, 'training') and hasattr(config.training, 'image_save_dir'):
            base_dir = config.training.image_save_dir

    # Get current date
    date_str = datetime.now().strftime("%m%d")

    # Build path components
    path_parts = [base_dir, date_str]

    if experiment_name is not None:
        path_parts.append(experiment_name)

    if step is not None:
        path_parts.append(f"step_{step}")

    if rollout_idx is not None:
        path_parts.append(f"rollout_{rollout_idx}")

    # Create directory path
    dir_path = Path(*path_parts)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename_parts = []
    if turn_idx is not None:
        filename_parts.append(f"turn_{turn_idx}")
    if agent_name is not None:
        filename_parts.append(agent_name)

    filename = "_".join(filename_parts) if filename_parts else "image"
    filename += ".png"

    return dir_path / filename


# =========================================================
# Base EnvState with Multimodal Support
# =========================================================

@dataclass
class EnvStateBase:
    tool_action: List[str] = field(default_factory=list, init=False)
    tool_code: str = field(default="", init=False)
    tool_execution_output: str = field(default="", init=False)
    plan_action: List[str] = field(default_factory=list, init=False)
    observation: Any = field(default=None, init=False)  # Changed from str to Any to support images

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
            self.observation = None

    def __str__(self) -> str:
        obs_type = type(self.observation).__name__ if self.observation is not None else "None"
        return (
            f"tool_action: {self.tool_action}\n"
            f"tool_code: {self.tool_code}\n"
            f"tool_execution_output: {self.tool_execution_output}\n"
            f"plan_action: {self.plan_action}\n"
            f"observation: <{obs_type}>"
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
            "reward", "total_reward", "step_count", "steps", "invalid_count",
            "tool_action", "tool_code", "tool_execution_output", "plan_action",
            "tool_reward", "boxes_on_goals",
        ]:
            if hasattr(self, key):
                compact[key] = getattr(self, key)

        # Add observation info (but not the actual image to avoid bloat)
        if hasattr(self, 'observation') and self.observation is not None:
            if isinstance(self.observation, Image.Image):
                compact["observation_type"] = "PIL.Image"
                compact["observation_size"] = self.observation.size
            else:
                compact["observation_type"] = type(self.observation).__name__

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
            # Skip observation since it might be an image
            if k == 'observation':
                if isinstance(v, Image.Image):
                    result[k] = f"<PIL.Image {v.size}>"
                else:
                    result[k] = str(v) if v is not None else None
                continue
            try:
                json.dumps(v)
                result[k] = v
            except Exception:
                # Numpy arrays or other non-serializable objects
                try:
                    import numpy as _np
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

    def render_to_image(self, save_path: Optional[Path] = None) -> Optional[Image.Image]:
        """
        Render state to PIL Image. Override in subclasses.

        Args:
            save_path: Optional path to save the image

        Returns:
            PIL Image object
        """
        return None

    def save_observation_image(
        self,
        config: Optional[Any] = None,
        experiment_name: Optional[str] = None,
        step: Optional[int] = None,
        rollout_idx: Optional[int] = None,
        turn_idx: Optional[int] = None,
        agent_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Save the current observation image to disk.

        Args:
            config: Configuration object
            experiment_name: Experiment name
            step: Training step
            rollout_idx: Rollout index
            turn_idx: Turn index
            agent_name: Agent name

        Returns:
            Path where the image was saved, or None if no image
        """
        if self.observation is None:
            return None

        if not isinstance(self.observation, Image.Image):
            return None

        # Generate save path
        save_path = get_image_save_path(
            config=config,
            experiment_name=experiment_name,
            step=step,
            rollout_idx=rollout_idx,
            turn_idx=turn_idx,
            agent_name=agent_name
        )

        # Save image
        self.observation.save(save_path)
        return save_path


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
    DEFAULT_R_STEP: ClassVar[float] = -0.01
    DEFAULT_R_INVALID: ClassVar[float] = -0.10
    DEFAULT_R_GOAL: ClassVar[float] = +1.00
    DEFAULT_R_OPT: ClassVar[float] = +0.50
    DEFAULT_R_FAIL: ClassVar[float] = -1.00
    DEFAULT_GAMMA: ClassVar[float] = 0.99
    DEFAULT_LAMBDA_POT: ClassVar[float] = 1.00
    DEFAULT_MAX_STEPS: ClassVar[int] = 10_000

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
        self.grid = '\n'.join(grid)
        self.grid_list = grid
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
        self.observation = self.render_to_image()

    def _generate_random_environment(self, seed: int, grid_h: int, grid_w: int, block_ratio: float) -> Tuple[List[str], Tuple[int, int], Tuple[int, int]]:
        """Generate random grid, start and goal based on seed"""
        rng = random.Random(seed)
        np.random.seed(seed)

        max_trials = max(2000, 50)
        for _ in range(max_trials):
            grid_array = (np.random.rand(grid_h, grid_w) < block_ratio).astype(int)

            free_positions = [(r, c) for r in range(grid_h) for c in range(grid_w) if grid_array[r, c] == 0]

            if len(free_positions) < 2:
                continue

            start = rng.choice(free_positions)
            goal = rng.choice(free_positions)
            while goal == start:
                goal = rng.choice(free_positions)

            if self._bfs_check_reachable(grid_array, start, goal):
                grid_str = []
                for row in grid_array:
                    row_str = ''.join('.' if cell == 0 else '#' for cell in row)
                    grid_str.append(row_str)

                return grid_str, start, goal

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

    def render_to_image(self, cell_size: int = 30, save_path: Optional[Path] = None) -> Image.Image:
        """
        Render grid state as PIL Image

        Args:
            cell_size: Size of each grid cell in pixels
            save_path: Optional path to save the image

        Returns:
            PIL Image object
        """
        img_width = self.w * cell_size
        img_height = self.h * cell_size

        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)

        # Colors
        wall_color = (60, 60, 60)        # Dark gray
        empty_color = (240, 240, 240)    # Light gray
        player_color = (100, 150, 255)   # Blue
        goal_color = (255, 200, 50)      # Gold

        for r in range(self.h):
            for c in range(self.w):
                x0 = c * cell_size
                y0 = r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                pos = (r, c)

                # Draw background
                if self.grid_list[r][c] == '#':
                    color = wall_color
                else:
                    color = empty_color

                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(200, 200, 200))

                # Draw special markers
                if pos == self.pos:
                    # Player position (circle)
                    margin = cell_size // 4
                    draw.ellipse([x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                                fill=player_color)
                elif pos == self.goal:
                    # Goal position (star-like shape)
                    margin = cell_size // 4
                    draw.ellipse([x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                                fill=goal_color)

        # Save image if path is provided
        if save_path is not None:
            img.save(save_path)

        return img

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
        self.observation = self.render_to_image()

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

        if isinstance(action, list) and all(isinstance(item, str) for item in action):
            self._execute_action_sequence(action)
        else:
            self.reward = -1.0

    def _execute_action_sequence(self, actions: List[str]):
        """Execute action sequence"""
        total_reward = 0.0
        for action in actions:
            pos, reward, done, _ = self.step_single(action)
            total_reward += reward
            if done:
                break
        self.reward = total_reward
        self.observation = self.render_to_image()

    def step_single(self, action: str) -> Tuple[Tuple[int,int], float, bool, Dict[str,Any]]:
        """
        Execute single action step and return:
          next_pos, reward, done, info
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

        # Shaping
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
            reward += self.r_goal
            sp = self.shortest_path()
            if sp is not None:
                if self.steps == len(sp) - 1:
                    reward += self.r_opt
            self.done = True
        elif self.steps >= self.max_steps:
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
        return self.pos, reward, self.done, info


@dataclass
class SokobanGridEnvState(EnvStateBase):
    """
    Sokoban push boxes environment - push all boxes to target positions
    - Symbols:
      '#' wall, ' ' empty space, '.' target, '$' box, '@' player, '*' box on target, '+' player on target
    - Actions: U/D/L/R (4-neighborhood)
    - Win condition: all boxes are on target positions
    """

    seed: int
    level: int = 1
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
    boxes: Set[Tuple[int, int]] = field(default_factory=set)
    goals: Set[Tuple[int, int]] = field(default_factory=set)
    walls: Set[Tuple[int, int]] = field(default_factory=set)
    room_structure: Optional[np.ndarray] = field(default=None, init=False)
    room_state: Optional[np.ndarray] = field(default=None, init=False)
    box_mapping: Dict[Tuple[int,int], Tuple[int,int]] = field(default_factory=dict, init=False)
    action_sequence: List[int] = field(default_factory=list, init=False)

    # Step-by-step interaction state
    done: bool = False
    steps: int = 0
    step_count: int = 0
    invalid_count: int = 0
    total_reward: float = 0.0
    reward: float = 0.0
    boxes_on_goals: int = 0
    _last_phi: float = 0.0

    # ====== Default reward coefficients ======
    DEFAULT_R_STEP: ClassVar[float] = -0.01
    DEFAULT_R_INVALID: ClassVar[float] = -0.05
    DEFAULT_R_BOX_ON_GOAL: ClassVar[float] = +1.0
    DEFAULT_R_BOX_OFF_GOAL: ClassVar[float] = -0.5
    DEFAULT_R_WIN: ClassVar[float] = +10.0
    DEFAULT_R_FAIL: ClassVar[float] = -5.0
    DEFAULT_GAMMA: ClassVar[float] = 0.99
    DEFAULT_LAMBDA_POT: ClassVar[float] = 0.1
    DEFAULT_MAX_STEPS: ClassVar[int] = 200

    ACTIONS: ClassVar[Dict[str, Tuple[int,int]]] = {
        "U": (-1, 0),
        "D": (+1, 0),
        "L": ( 0,-1),
        "R": ( 0,+1),
    }

    def __post_init__(self):
        super().__post_init__()
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

        assert generate_room is not None, "Cannot import generate_room"

        room_structure, room_state, box_mapping, action_sequence = generate_room(
            dim=(self.size, self.size),
            p_change_directions=0.35,
            num_steps=25,
            num_boxes=1,
            tries=20,
            second_player=False,
            search_depth=200,
            min_box_distance=2,
            min_difficulty_score=1,
            seed=self.seed,
        )

        self.room_structure = room_structure
        self.room_state = room_state
        self.box_mapping = {tuple(k): tuple(v) for k, v in box_mapping.items()}
        self.action_sequence = list(action_sequence)

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

        self.done = False
        self.steps = 0
        self.step_count = 0
        self.invalid_count = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self._last_phi = self._potential()
        self.observation = self.render_to_image()

    def render_to_image(self, cell_size: int = 30, save_path: Optional[Path] = None) -> Image.Image:
        """
        Render Sokoban state as PIL Image

        Args:
            cell_size: Size of each grid cell in pixels
            save_path: Optional path to save the image

        Returns:
            PIL Image object
        """
        img_width = self.w * cell_size
        img_height = self.h * cell_size

        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)

        # Colors
        wall_color = (80, 60, 40)         # Brown
        floor_color = (240, 230, 220)     # Light beige
        goal_color = (255, 220, 150)      # Yellow-orange
        box_color = (180, 100, 50)        # Brown box
        box_on_goal_color = (100, 200, 100)  # Green
        player_color = (100, 150, 255)    # Blue

        for r in range(self.h):
            for c in range(self.w):
                x0 = c * cell_size
                y0 = r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                pos = (r, c)

                # Draw background
                if pos in self.walls:
                    color = wall_color
                elif pos in self.goals:
                    color = goal_color
                else:
                    color = floor_color

                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(150, 150, 150))

                # Draw boxes
                if pos in self.boxes:
                    margin = cell_size // 5
                    if pos in self.goals:
                        draw.rectangle([x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                                     fill=box_on_goal_color, outline=(50, 50, 50), width=2)
                    else:
                        draw.rectangle([x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                                     fill=box_color, outline=(50, 50, 50), width=2)

                # Draw player
                if pos == self.player_pos:
                    margin = cell_size // 4
                    draw.ellipse([x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                               fill=player_color, outline=(50, 50, 150), width=2)

        # Save image if path is provided
        if save_path is not None:
            img.save(save_path)

        return img

    def describe(self) -> str:
        return (
            "SokobanGridWorker: Push boxes ($) to goal positions (.). "
            "Player (@) moves with U/D/L/R. Win when all boxes are on goals (*)."
        )

    def reset_agent(self):
        """Reset step-by-step interaction state to initial state"""
        assert self.room_structure is not None and self.room_state is not None

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
        self.observation = self.render_to_image()

    def get_valid_actions(self, pos: Optional[Tuple[int,int]] = None) -> List[str]:
        """Return valid action set for current position"""
        if pos is None:
            pos = self.player_pos
        valid = []
        for action, (dr, dc) in self.ACTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            next_pos = (nr, nc)

            if not self.in_bounds(nr, nc) or next_pos in self.walls:
                continue

            if next_pos in self.boxes:
                box_nr, box_nc = nr + dr, nc + dc
                box_next_pos = (box_nr, box_nc)

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

        if not self.in_bounds(nr, nc) or next_pos in self.walls:
            return self.player_pos, False, {"msg": "hit wall or out of bounds"}

        info = {"pushed_box": False, "box_on_goal_change": 0}

        if next_pos in self.boxes:
            box_nr, box_nc = nr + dr, nc + dc
            box_next_pos = (box_nr, box_nc)

            if (not self.in_bounds(box_nr, box_nc) or
                box_next_pos in self.walls or
                box_next_pos in self.boxes):
                return self.player_pos, False, {"msg": "cannot push box"}

            self.boxes.remove(next_pos)
            self.boxes.add(box_next_pos)
            info["pushed_box"] = True

            old_on_goal = next_pos in self.goals
            new_on_goal = box_next_pos in self.goals

            if old_on_goal and not new_on_goal:
                info["box_on_goal_change"] = -1
            elif not old_on_goal and new_on_goal:
                info["box_on_goal_change"] = +1

        return next_pos, True, info

    def step(self, action):
        """Execute action and update environment state"""
        if self.done:
            self.reward = 0.0
            return

        if isinstance(action, list) and all(isinstance(item, str) for item in action):
            self._execute_action_sequence(action)
        else:
            self.reward = -1.0

    def _execute_action_sequence(self, actions: List[str]):
        """Execute action sequence"""
        total_reward = 0.0
        for action in actions:
            pos, reward, done, _ = self.step_single(action)
            total_reward += reward
            if done:
                break
        self.reward = total_reward
        self.observation = self.render_to_image()

    def step_single(self, action: str) -> Tuple[Tuple[int,int], float, bool, Dict[str,Any]]:
        """Execute single action step"""
        if self.done:
            return self.player_pos, 0.0, True, {"msg": "episode already done"}

        prev_pos = self.player_pos
        prev_boxes_on_goals = len(self.boxes & self.goals)

        next_pos, valid, action_info = self._apply_action(action)

        reward = 0.0
        if valid:
            reward += self.r_step
        else:
            reward += self.r_invalid
            self.invalid_count += 1

        if valid and action_info["pushed_box"]:
            if action_info["box_on_goal_change"] > 0:
                reward += self.r_box_on_goal
            elif action_info["box_on_goal_change"] < 0:
                reward += self.r_box_off_goal

        cur_phi = self._last_phi
        nxt_phi = self._potential()
        shaping = self.lambda_pot * (self.gamma * nxt_phi - cur_phi)
        reward += shaping

        if valid:
            self.player_pos = next_pos
        self._last_phi = self._potential()
        self.steps += 1

        self.boxes_on_goals = len(self.boxes & self.goals)

        if self._is_won():
            reward += self.r_win
            self.done = True
        elif self.steps >= self.max_steps:
            reward += self.r_fail
            self.done = True

        self.total_reward += reward

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


# =========================================================
# State Registry
# =========================================================

STATE_REGISTRY = {
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
