#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for action extraction, code parsing, and Sokoban environment generation
"""

import json
import random
import re
import ast
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import marshal
from collections import deque


# ============================================================
# 1. Action and Code Extraction Functions
# ============================================================

def extract_final_action(text: str, benchmark: str = "plan_path") -> List | None:
    """
    Extract the final action from text that appears on the last line starting with '#### '.
    Supports different formats for different benchmarks.
    """
    if text is None or not isinstance(text, str):
        return None
    
    pattern = re.compile(r'(?m)^\s*####\s+(.+)$', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        return None
    
    action_str = matches[-1].strip()
    
    try:
        if action_str.startswith('[') and action_str.endswith(']'):
            parsed = ast.literal_eval(action_str)
            if isinstance(parsed, list):
                return parsed
    except (ValueError, SyntaxError):
        pass
    
    try:
        if action_str.startswith('[') and action_str.endswith(']'):
            return json.loads(action_str)
    except json.JSONDecodeError:
        pass
    
    try:
        if benchmark in ("plan_path", "sokoban") and action_str.startswith('[') and action_str.endswith(']'):
            inner = action_str[1:-1].strip()
            if inner:
                actions = [item.strip().strip('"\'') for item in inner.split(',')]
                actions = [action for action in actions if action]
                if all(a in ['U','D','L','R'] for a in actions):
                    return actions
            else:
                return []
    except Exception:
        pass
    
    return None


def extract_code_from_response(response: str) -> str:
    """
    Extract code blocks from response (supports ```python or ``` format)
    """
    if response is None or not isinstance(response, str):
        return ""
    
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    incomplete_python_pattern = r'```python\s*(.*?)$'
    matches = re.findall(incomplete_python_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    incomplete_code_pattern = r'```\s*(.*?)$'
    matches = re.findall(incomplete_code_pattern, response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if any(keyword in code for keyword in ['def ', 'import ', 'from ', '=', 'print(', 'return', 'if ', 'for ', 'while ']):
            return code
    
    return response.strip()


def extract_actions_from_code_output(output: str, benchmark: str = "plan_path") -> Optional[List]:
    """
    Extract action list from code execution output
    Supported formats:
    - **Actions List**: [...]
    - Actions: [...]
    - Direct list output
    """
    if output is None or not isinstance(output, str) or output.startswith("error:"):
        return None
    
    try:
        actions_pattern = r'\*\*Actions List\*\*:\s*(\[.*?\])'
        matches = re.findall(actions_pattern, output, re.DOTALL)
        
        if matches:
            actions_str = matches[-1]
            try:
                actions = eval(actions_str)
                if isinstance(actions, list):
                    if benchmark == "plan_path":
                        if all(isinstance(action, str) and action in ['U', 'D', 'L', 'R'] for action in actions):
                            return actions
                    elif benchmark == "suduku":
                        if (len(actions) > 0 and isinstance(actions[0], list) and 
                            all(isinstance(row, list) and len(row) > 0 for row in actions)):
                            return actions
                        elif all(isinstance(step, list) and len(step) == 3 for step in actions):
                            return actions
                    else:
                        return actions
            except:
                pass
        
        actions_pattern2 = r'Actions:\s*(\[.*?\])'
        matches2 = re.findall(actions_pattern2, output, re.DOTALL)
        
        if matches2:
            actions_str = matches2[-1]
            try:
                actions = eval(actions_str)
                if isinstance(actions, list):
                    if benchmark == "plan_path":
                        if all(isinstance(action, str) and action in ['U', 'D', 'L', 'R'] for action in actions):
                            return actions
                    elif benchmark == "suduku":
                        if (len(actions) > 0 and isinstance(actions[0], list)):
                            return actions
                    else:
                        return actions
            except:
                pass
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    parsed = eval(line)
                    if isinstance(parsed, list):
                        if benchmark == "plan_path":
                            if all(isinstance(item, str) and item in ['U', 'D', 'L', 'R'] for item in parsed):
                                return parsed
                        elif benchmark == "suduku":
                            if (len(parsed) > 0 and isinstance(parsed[0], list)):
                                return parsed
                        else:
                            return parsed
                except:
                    continue
        
    except Exception:
        pass
    
    return None


def truncatefn(s, length=300):
    """Truncate string for logging output"""
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= length else s[: length // 2] + "...(truncated)..." + s[-length // 2 :]


# ============================================================
# 2. Problem Loading Functions (for training/validation datasets)
# ============================================================

def load_plan_path_problem_batch(
    env_indices: List[int],
    dataset_name: str = "train", 
    split: str = "train",
    mode: str = "train",
    config: dict = None,
    benchmark_name: str = "PlanPath"
) -> List[Dict[str, Any]]:
    """
    Load problem batch based on benchmark name
    
    Args:
        env_indices: List of environment indices
        dataset_name: Dataset name
        split: Dataset split
        mode: "train" or "validate"
        config: Configuration dictionary
        benchmark_name: Benchmark name
        
    Returns:
        List of problem dictionaries for the benchmark
    """
    if benchmark_name == "EightQueens":
        problems = []
        for i in range(len(env_indices)):
            n = 4 + (i % 5)
            problems.append({"N": n})
        return problems
    
    elif benchmark_name == "Blocksworld":
        problems = []
        block_configs = [
            {
                "init_stacks": [["A", "B"], ["C"]],
                "goal_stacks": [["C", "A"], ["B"]]
            },
            {
                "init_stacks": [["A"], ["B", "C"], ["D"]],
                "goal_stacks": [["D", "C", "B", "A"], [], []]
            },
            {
                "init_stacks": [["A", "B", "C"], []],
                "goal_stacks": [[], ["C", "B", "A"]]
            },
        ]
        for i in range(len(env_indices)):
            config_idx = i % len(block_configs)
            problems.append(block_configs[config_idx])
        return problems
    
    elif benchmark_name == "suduku":
        problems = []
        for i in range(len(env_indices)):
            seed = env_indices[i] if i < len(env_indices) else i
            problems.append({"seed": seed})
        return problems
    
    elif benchmark_name in ("plan_path", "sokoban"):
        problems = []
        for i in range(len(env_indices)):
            seed = env_indices[i] if i < len(env_indices) else i
            problems.append({"seed": seed})
        return problems
    
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}. Supported: EightQueens, Blocksworld, suduku, plan_path, sokoban")


# ============================================================
# 3. Sokoban Environment Generation Functions
# ============================================================

def get_shortest_action_path(room_fixed, room_state, MAX_DEPTH=100):
    """
    Use BFS to find the shortest action sequence to push all boxes to target positions
    
    Parameters:
        room_state (np.ndarray): Room state
            - 0: wall
            - 1: empty space
            - 2: target position
            - 3: box on target
            - 4: box not on target
            - 5: player
        room_fixed (np.ndarray): Fixed room structure
            - 0: wall
            - 1: empty space
            - 2: target position
        MAX_DEPTH (int): Maximum search depth
        
    Returns:
        action_sequence (list): Action sequence
    """
    queue = deque([(np.copy(room_state), [])])
    explored_states = set()
    
    moves = [(-1,0), (1,0), (0,-1), (0,1)]
    actions = [1, 2, 3, 4]
    
    while queue:
        room_state, path = queue.popleft()
        if len(path) > MAX_DEPTH:
            return []

        state_tohash = marshal.dumps(room_state)
        if state_tohash in explored_states:
            continue
        explored_states.add(state_tohash)

        player_pos = tuple(np.argwhere(room_state == 5)[0])
        boxes_on_target = set(map(tuple, np.argwhere((room_state == 3))))
        boxes_not_on_target = set(map(tuple, np.argwhere((room_state == 4))))
        boxes = boxes_on_target | boxes_not_on_target

        if not boxes_not_on_target:
            return path
            
        for move, action in zip(moves, actions):
            new_room_state = np.copy(room_state)
            new_player_pos = (player_pos[0] + move[0], player_pos[1] + move[1])
            
            if new_player_pos[0] < 0 or new_player_pos[0] >= room_fixed.shape[0] \
                or new_player_pos[1] < 0 or new_player_pos[1] >= room_fixed.shape[1] \
                or room_fixed[new_player_pos] == 0:
                continue
                
            if new_player_pos in boxes:
                box_pos = new_player_pos
                new_box_pos = (new_player_pos[0] + move[0], new_player_pos[1] + move[1])
                
                if room_fixed[new_box_pos] == 0 or new_box_pos in boxes \
                    or new_box_pos[0] < 0 or new_box_pos[0] >= room_fixed.shape[0] \
                    or new_box_pos[1] < 0 or new_box_pos[1] >= room_fixed.shape[1]:
                    continue
                    
                new_room_state[box_pos] = room_fixed[box_pos]
                if room_fixed[new_box_pos] == 2:
                    new_room_state[new_box_pos] = 3
                else:
                    new_room_state[new_box_pos] = 4
            
            new_room_state[player_pos] = room_fixed[player_pos]
            new_room_state[new_player_pos] = 5
            queue.append((new_room_state, path + [action]))
                    
    return []


def add_random_player_movement(room_state, room_structure, move_probability=0.5, continue_probability=0.5, max_steps=3):
    """
    Randomly move the player after reverse_playing to make the level more challenging
    
    Parameters:
        room_state: Current room state
        room_structure: Fixed room structure
        move_probability: Probability of moving the player
        continue_probability: Probability of continuing to move
        max_steps: Maximum number of steps
        
    Returns:
        Updated room state
    """
    if random.random() > move_probability:
        return room_state
    
    player_pos = np.where(room_state == 5)
    player_pos = np.array([player_pos[0][0], player_pos[1][0]])
    
    previous_positions = [tuple(player_pos)]
    
    steps_taken = 0
    while steps_taken < max_steps:
        valid_moves = []
        for action in range(4):
            change = CHANGE_COORDINATES[action]
            next_pos = player_pos + change
            
            if (room_state[next_pos[0], next_pos[1]] in [1, 2] and 
                tuple(next_pos) not in previous_positions):
                valid_moves.append((action, next_pos))
        
        if not valid_moves:
            break
        
        chosen_action, next_pos = random.choice(valid_moves)
        
        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        room_state[next_pos[0], next_pos[1]] = 5
        
        player_pos = next_pos
        previous_positions.append(tuple(player_pos))
        
        steps_taken += 1
        
        if steps_taken >= max_steps or random.random() > continue_probability:
            break
    
    return room_state


def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False, search_depth=100, min_box_distance=2, min_difficulty_score=None, seed=None):
    """
    Generate Sokoban room
    
    Args:
        dim: Room dimensions
        p_change_directions: Probability of changing direction in topology generation
        num_steps: Number of steps in topology generation
        num_boxes: Number of boxes to place
        tries: Maximum number of generation attempts
        second_player: Whether to add a second player
        search_depth: Maximum search depth for reverse playing
        min_box_distance: Minimum Manhattan distance between each box and its target (default: 2)
        min_difficulty_score: Minimum required total displacement score (default: num_boxes * min_box_distance)
        seed: Random seed for reproducible room generation (default: None)
    
    Returns:
        room_structure: Fixed structure (walls/empty/targets)
        room_state: Current state (including player/boxes)
        box_mapping: Box mapping
        action_sequence: Action sequence
    """
    # Set random seed to ensure reproducibility with the same seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)
    
    # Set minimum difficulty score based on number of boxes and minimum distance
    if min_difficulty_score is None:
        min_difficulty_score = num_boxes * min_box_distance

    for t in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        room_state = room.copy()
        room_state[room_state == 2] = 4

        room_state, box_mapping, action_sequence = reverse_playing(room_state, room_structure, search_depth, min_box_distance)
        room_state[room_state == 3] = 4

        score = box_displacement_score(box_mapping, min_box_distance)
        if score >= min_difficulty_score:
            break

    final_score = box_displacement_score(box_mapping, min_box_distance)
    if final_score == 0:
        raise RuntimeWarning(f'Generated Model with score == 0 (boxes too close to targets, min_distance={min_box_distance})')
    
    if final_score < min_difficulty_score:
        raise RuntimeWarning(f'Generated Model with insufficient difficulty: score={final_score}, required={min_difficulty_score}')

    # Adjust random movement based on difficulty
    if final_score <= min_difficulty_score:
        move_probability = 0.9  # Add more randomness to increase difficulty
    elif final_score <= min_difficulty_score * 1.5:
        move_probability = 0.7
    else:
        move_probability = 0.5
        
    room_state = add_random_player_movement(
        room_state, 
        room_structure,
        move_probability=move_probability,
        continue_probability=0.5,
        max_steps=3
    )

    return room_structure, room_state, box_mapping, action_sequence


def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15):
    """Generate room topology (empty spaces and walls)"""
    dim_x, dim_y = dim

    masks = [
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
    ]

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = random.sample(directions, 1)[0]

    position = np.array([
        random.randint(1, dim_x - 1),
        random.randint(1, dim_y - 1)]
    )

    level = np.zeros(dim, dtype=int)

    for s in range(num_steps):
        if random.random() < p_change_directions:
            direction = random.sample(directions, 1)[0]

        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        mask = random.sample(masks, 1)[0]
        mask_start = position - 1
        level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level


def place_boxes_and_player(room, num_boxes, second_player):
    """Place player and boxes in the room"""
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
            num_possible_positions,
            num_players,
            num_boxes)
        )

    ind = np.random.randint(num_possible_positions)
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    room[player_position] = 5

    if second_player:
        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

    for n in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = np.random.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        room[box_position] = 2

    return room


# Global variables for reverse playing
explored_states = set()
num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None


def reverse_playing(room_state, room_structure, search_depth=100, min_box_distance=2):
    """
    Reverse play Sokoban where the player can pull boxes
    Ensures a solvable level with all boxes not on target positions
    
    Args:
        room_state: Current room state
        room_structure: Fixed room structure
        search_depth: Maximum search depth
        min_box_distance: Minimum required distance between boxes and targets
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_action_sequence

    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box

    explored_states = set()
    best_room_score = -1
    best_room = None
    best_box_mapping = box_mapping
    best_action_sequence = []

    depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=search_depth, action_sequence=[], min_box_distance=min_box_distance)

    return best_room, best_box_mapping, best_action_sequence


def depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300, action_sequence=[], min_box_distance=2):
    """
    Depth-first search through all possible room states
    
    Args:
        room_state: Current room state
        room_structure: Fixed room structure
        box_mapping: Mapping from target positions to current box positions
        box_swaps: Number of box movements
        last_pull: Last pulled box position
        ttl: Time to live (remaining search depth)
        action_sequence: Sequence of actions taken
        min_box_distance: Minimum required distance between boxes and targets
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_action_sequence

    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 300000:
        return

    state_tohash = marshal.dumps(room_state)

    if not (state_tohash in explored_states):
        # Calculate score with minimum distance requirement
        displacement_score = box_displacement_score(box_mapping, min_box_distance)
        room_score = box_swaps * displacement_score
        
        # Ensure all boxes are off targets (not on position 2)
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = room_state.copy()
            best_room_score = room_score
            best_box_mapping = box_mapping.copy()
            best_action_sequence = action_sequence.copy()

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            if action >= 4:
                continue

            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()

            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1
            
            action_sequence_next = action_sequence + [action]
            depth_first_search(room_state_next, room_structure, box_mapping_next, box_swaps_next, last_pull_next, ttl, action_sequence_next, min_box_distance)


def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    """Execute reverse action (pull box)"""
    player_position = np.where(room_state == 5)
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change

    if room_state[next_position[0], next_position[1]] in [1, 2]:
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = 5

        if action < 4:
            possible_box_location = change[0] * -1, change[1] * -1
            possible_box_location += player_position

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]]

                for k in box_mapping.keys():
                    if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                        box_mapping[k] = (player_position[0], player_position[1])
                        last_pull = k

    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping, min_distance=2):
    """
    Calculate the sum of Manhattan distances between all boxes and their target positions
    
    Args:
        box_mapping: Dictionary mapping target positions to current box positions
        min_distance: Minimum required distance for each box (default: 2)
    
    Returns:
        score: Sum of distances, or 0 if any box is closer than min_distance
    """
    score = 0
    
    for box_target in box_mapping.keys():
        box_location = np.array(box_mapping[box_target])
        box_target = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target))
        
        # If any box is too close to its target, return 0 (invalid)
        if dist < min_distance:
            return 0
        
        score += dist

    return score


# ============================================================
# Constants
# ============================================================

TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
