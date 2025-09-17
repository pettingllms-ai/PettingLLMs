import logging
import re
from typing import Any, List, Tuple, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.plan_path.utils import (
    _extract_actions, _extract_path, _actions_to_path, _format_grid
)

from pettingllms.multi_agent_env.math.math_utils import get_code_execution_output
from pettingllms.multi_agent_env.plan_path.prompt import build_tool_prompt
import asyncio
logger = logging.getLogger(__name__)
import copy

def truncatefn(s, length=300):
    """截断字符串到指定长度"""
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[:length] + "..."


def extract_code_from_response(response: str) -> str:
    """
    从智能体响应中提取代码块。
    
    Args:
        response: 智能体响应字符串
        
    Returns:
        提取的代码字符串
    """
    # 优先寻找完整的 Python 代码块
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()  # 返回最后一个代码块
    
    # 寻找完整的通用代码块
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    # 寻找不完整的 Python 代码块（只有开始标记）
    incomplete_python_pattern = r'```python\s*(.*?)$'
    matches = re.findall(incomplete_python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    # 寻找不完整的通用代码块（只有开始标记）
    incomplete_code_pattern = r'```\s*(.*?)$'
    matches = re.findall(incomplete_code_pattern, response, re.DOTALL)
    
    if matches:
        code = matches[-1].strip()
        # 检查是否看起来像代码（包含常见的Python关键字或语法）
        if any(keyword in code for keyword in ['def ', 'import ', 'from ', '=', 'print(', 'return', 'if ', 'for ', 'while ']):
            return code
    
    # 如果没有找到代码块，返回整个响应
    return response.strip()


def extract_actions_from_code_output(output: str) -> Optional[List[str]]:
    """
    从代码执行输出中提取动作序列。
    
    Args:
        output: 代码执行的输出字符串
        
    Returns:
        动作序列列表，如 ["U", "R", "D"] 或 None
    """
    if not output or output.startswith("error:"):
        return None
    
    try:
        # 优先寻找 "Actions: [...]" 格式
        actions_with_label_pattern = r'Actions:\s*\[\s*["\']?[UDLR]["\']?\s*(?:,\s*["\']?[UDLR]["\']?\s*)*\]'
        matches = re.findall(actions_with_label_pattern, output)
        
        if matches:
            # 提取动作列表部分
            actions_str = re.search(r'\[.*\]', matches[-1]).group()
            actions = eval(actions_str)
            if isinstance(actions, list) and all(
                isinstance(action, str) and action in ['U', 'D', 'L', 'R']
                for action in actions
            ):
                return actions
        
        # 尝试解析为纯动作序列 ["U", "D", "L", "R"]
        actions_pattern = r'\[\s*["\']?[UDLR]["\']?\s*(?:,\s*["\']?[UDLR]["\']?\s*)*\]'
        matches = re.findall(actions_pattern, output)
        
        if matches:
            actions_str = matches[-1]
            actions = eval(actions_str)
            if isinstance(actions, list) and all(
                isinstance(action, str) and action in ['U', 'D', 'L', 'R']
                for action in actions
            ):
                return actions
        
        # 尝试从输出的每一行中提取动作序列
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            # 检查是否包含 Actions: 标签
            if "Actions:" in line:
                # 提取 Actions: 后面的列表
                actions_part = line.split("Actions:")[-1].strip()
                if actions_part.startswith('[') and actions_part.endswith(']'):
                    try:
                        actions = eval(actions_part)
                        if isinstance(actions, list) and all(
                            isinstance(action, str) and action in ['U', 'D', 'L', 'R']
                            for action in actions
                        ):
                            return actions
                    except:
                        continue
            # 检查纯列表格式
            elif line.startswith('[') and line.endswith(']'):
                try:
                    parsed = eval(line)
                    if isinstance(parsed, list) and all(
                        isinstance(item, str) and item in ['U', 'D', 'L', 'R'] 
                        for item in parsed
                    ):
                        return parsed
                except:
                    continue
        
    except Exception as e:
        logger.warning(f"Failed to extract actions from code output: {e}")
    
    return None


class ToolAgent(Agent):
    """
    Code-generation style planning agent.
    - 仅首轮/后续 prompt 由 benchmark 决定（已外部化）。
    - 其余逻辑（执行、解析、评分、写回、done 判定）保持不变。
    """

    def __init__(self, rollout_idx: int | None = None, benchmark: str = "plan_path", **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.benchmark = benchmark  # 关键：切换不同任务的 prompt
        self.agent_reward_history = []
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)

    def update_from_env(self, turn_idx: int, env_data: Env):
        self.env_data = env_data
        state = getattr(env_data, "state", None)
        formatted_prompt = "You are an AI assistant specialized in solving planning problems through code generation. Your task is to analyze the given scenario and generate Python code that produces a sequence of actions to solve the problem.\n\nInstructions:\n1. Write Python code enclosed in ```python and ``` tags\n2. Your code should output an action sequence using print() function\n3. Actions should be represented as a list of strings: ['U', 'D', 'L', 'R'] (Up, Down, Left, Right)\n4. You may return either the complete action sequence to reach the goal, or a partial sequence if you're uncertain\n5. If your algorithm produces numerical results, convert them using: action_map = {0:'U', 1:'D', 2:'L', 3:'R'}\n6. Ensure your code is executable and produces clear output\n\n"
        formatted_prompt+= build_tool_prompt(self.benchmark, turn_idx, state)
        formatted_prompt+= f"Important: Your code must output the final action sequence in this exact format:\n**Actions List**: Example: [\"U\", \"R\", \"D\", \"L\"](just an example, do not print this directly) \n\nNote: If your algorithm produces numerical results, convert them using action_map = {{0:'U', 1:'D', 2:'L', 3:'R'}} before outputting.\n"
        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        self.current_code = extract_code_from_response(response)
        return self.current_code

    async def step(self, env_data: Env, env_worker: Any = None):
        # === 以下保持你的原始实现（执行代码 -> 解析 -> 评分 -> 回写） ===
        
        generated_code = self.current_code or ""
        env_data.state.code_generated_action = generated_code

        code_execution_output = None
        try:
            code_execution_output = await get_code_execution_output(
                generated_code,
                timeout=20.0,
                ray_actor=env_worker,
            )
            env_data.state.code_execution_output = code_execution_output
        except Exception as e:
            code_execution_output = f"error: {e}"
            env_data.state.code_execution_output = code_execution_output
        env_data.state.tool_execution_output = code_execution_output
        env_data.state.tool_code = generated_code
        self.current_action = extract_actions_from_code_output(code_execution_output or [])
        env_data.state.tool_action = self.current_action
        state = copy.deepcopy(env_data.state)
        state.step(self.current_action)
        if self.current_action is None:
            self.agent_reward = -5
        else:
            self.agent_reward = 1+ state.reward
        self.agent_reward_history.append(self.agent_reward)
        
        # 检查是否成功完成任务
        if hasattr(state, 'done') and env_data.state.done:
            # 根据不同的 benchmark 检查成功条件
            if self.benchmark == "plan_path":
                # PlanPath: 检查是否到达目标位置
                if hasattr(state, 'pos') and hasattr(state, 'goal') and state.pos == state.goal:
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)  # 确保成功时有正奖励
            elif self.benchmark == "eight_queens":
                # EightQueens: 检查是否正确放置了所有皇后
                if hasattr(state, '_is_solved') and state._is_solved():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
            elif self.benchmark == "blocksworld":
                # Blocksworld: 检查是否达到目标配置
                if hasattr(state, '_is_goal_reached') and state._is_goal_reached():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
            elif self.benchmark == "sudoku4x4":
                # Sudoku4x4: 检查是否正确解决数独
                if hasattr(state, '_is_solved') and state._is_solved():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
        
        if self.agent_reward is None:
            self.agent_reward = 0.0
        

    def reset(self):
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.done = False
        self.is_pass = False