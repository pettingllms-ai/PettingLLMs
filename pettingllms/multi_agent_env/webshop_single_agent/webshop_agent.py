import copy
import logging
import re
from typing import Any, List
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


def truncate_text(text: str, length: int = 500) -> str:
    """Truncate text to specified length while preserving context."""
    if isinstance(text, str):
        pass
    else:
        text = str(text)
    
    if len(text) <= length:
        return text
    
    return text[:length // 2] + "...(truncated)..." + text[-length // 2:]


class WebShopAgent(Agent):
    """
    Agent specialized for navigating and making purchases in WebShop environment.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the WebShop Agent.
        
        Args:
            rollout_idx: Rollout index for logging
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)
       

    def update_from_env(self, env_data: Env):
        """
        Update agent state based on environment data and generate appropriate prompt.
        
        Args:
            env_data: Environment data containing current state
        """
        # Save environment data
        self.env_data = env_data
        
        # Extract state information
        state = getattr(env_data, "state", None)
        if state is None:
            raise ValueError("Environment state is None")
        
        def as_text(value: Any) -> str:
            """Convert value to text representation."""
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join([str(v) for v in value])
            return str(value)
        
        # Extract current state information
        goal = getattr(state, "goal", "")
        current_observation = getattr(state, "current_observation", "")
        available_actions = getattr(state, "available_actions", [])
        action_history = getattr(state, "action_history", [])
        observation_history = getattr(state, "observation_history", [])
        step_count = getattr(state, "step_count", 0)
        max_steps = getattr(state, "max_steps", 10)
        
        # Truncate long observations for better prompt efficiency
        current_obs_truncated = truncate_text(current_observation, 800)
        
        # Build action history context (last 3 actions)
        recent_actions = action_history[-3:] if len(action_history) > 3 else action_history
        action_context = ""
        if recent_actions:
            action_context = f"\n\nRecent actions taken:\n"
            for i, action in enumerate(recent_actions, 1):
                action_context += f"{len(action_history) - len(recent_actions) + i}. {action}\n"
        
        # Create shopping instructions
        shopping_instructions = [
            "You are a shopping assistant helping to find and purchase products online.",
            "Your goal is to successfully purchase a product that matches the given description.",
            "You have a maximum of 10 actions to complete the purchase.",
            "",
            "Key strategies:",
            "- Use search to find products matching the description",
            "- Don't be too specific in search terms (avoid size, color details)",
            "- Click on products to view details",
            "- Select appropriate size and color if needed",
            "- Click 'buy now' to complete the purchase",
            "- Prioritize products on the current page over navigating to new pages",
            "- If you have less than 3 actions left, purchase the best available option",
            "",
            "Action format:",
            "- search[product name] - to search for products",
            "- click[item] - to click on buttons, products, or links",
        ]
        
        # Build the prompt
        formatted_prompt = "\n".join(shopping_instructions) + "\n\n"
        formatted_prompt += f"SHOPPING GOAL: {goal}\n\n"
        formatted_prompt += f"CURRENT STEP: {step_count}/{max_steps}\n\n"
        formatted_prompt += f"CURRENT PAGE:\n{current_obs_truncated}\n"
        formatted_prompt += action_context
        formatted_prompt += f"\nAVAILABLE ACTIONS:\n{', '.join(available_actions)}\n\n"
        formatted_prompt += "Choose your next action from the available actions. "
        formatted_prompt += "Respond with only the action you want to take (e.g., 'search[laptop]' or 'click[buy now]')."
        
        self.current_prompt = {"text": formatted_prompt, "image": None}
        
        logger.debug(f"WebShop agent prompt generated for step {step_count}")

    def update_from_model(self, response: str):
        """
        Parse model response and extract the action to take.
        
        Args:
            response: Raw response from the language model
            
        Returns:
            Parsed action string
        """
        # Clean the response
        response = response.strip()
        
        # Try to extract action from various formats
        action = ""
        
        # Look for action patterns
        patterns = [
            r'^(search\[[^\]]+\])$',  # search[content]
            r'^(click\[[^\]]+\])$',   # click[content]
            r'(search\[[^\]]+\])',    # search anywhere in text
            r'(click\[[^\]]+\])',     # click anywhere in text
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                action = matches[0].strip()
                break
        
        # If no pattern matched, try to find any text that looks like an action
        if not action:
            # Look for lines that might be actions
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('search[') or line.startswith('click['):
                    action = line
                    break
        
        # Final fallback - if still no action found, use the first line
        if not action:
            first_line = response.split('\n')[0].strip()
            if first_line:
                action = first_line
            else:
                action = "search[product]"  # Default fallback action
        
        # Update the agent's current action
        self.current_action = action
        
        logger.debug(f"WebShop agent parsed action: '{action}' from response: '{response[:100]}...'")
        
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Execute the agent's action in the environment.
        
        Args:
            env_data: Environment data
            env_worker: Optional environment worker for parallel execution
        """
        # The action has been set by update_from_model
        action = self.current_action
        
        if not action:
            logger.warning("No action available for WebShop agent step")
            return
        
        logger.info(f"WebShop agent executing action: '{action}'")
        
        # The environment step is handled by the environment itself
        # This method is called after the environment step to perform any
        # additional agent-specific processing if needed
        
        return action

    def reset(self):
        """Reset the agent's internal state."""
        super().reset()
        self.current_prompt = {"text": None, "image": None}
        self.current_action = None
        logger.debug("WebShop agent reset")
