import copy
import logging
from typing import Any, Dict, Optional
import base64

from pettingllms.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


def encode_image(image_path: str) -> str:
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"


class VisualInfoAgent(BaseAgent):
    """
    A visual information agent that acts as a VLM to compare two images and provide modification suggestions.
    Input: Reference design image and current HTML rendered image
    Output: Difference analysis and modification suggestions
    """

    def __init__(self, max_suggestions=3):
        """
        Initialize VisualInfoAgent.
        
        Args:
            max_suggestions: Maximum number of suggestions
        """
        self.system_prompt = """You are a professional visual design analyst. Your task is to compare two webpage design images:
1. Reference design image (what the user wants)
2. Current implementation image (existing HTML rendering result)

Please carefully analyze the differences between these two images and provide specific modification suggestions. Focus on:
- Layout differences (position, alignment, spacing)
- Color differences (background color, text color, theme color)
- Font and text style differences
- Image and icon differences
- Component size and proportion differences
- Navigation bar, button and other interactive element differences

Please prioritize by importance and provide the most critical modification suggestions. Each suggestion should be specific and clear, making it easy for code implementation."""
        
        self._trajectory = Trajectory()
        self.messages = []
        self.max_suggestions = max_suggestions
        self.current_observation = None

    def analyze_visual_differences(self, reference_image_path: str, current_image_path: str) -> str:
        """
        Analyze visual differences between two images
        
        Args:
            reference_image_path: Reference design image path
            current_image_path: Current implementation image path
            
        Returns:
            Formatted difference analysis and modification suggestions
        """
        # Encode images
        ref_image_b64 = encode_image(reference_image_path)
        current_image_b64 = encode_image(current_image_path)
        
        # Build visual comparison message
        visual_analysis_prompt = f"""Please compare the following two webpage design images and provide modification suggestions:

1. The first image is the reference design (user expected design)
2. The second image is the current implementation (existing HTML rendering result)

Please analyze the main differences and provide specific modification suggestions:

Reference design image:
<img src="{ref_image_b64}" />

Current implementation image:
<img src="{current_image_b64}" />

Please provide analysis in the following format:

## Main Difference Analysis
1. [Difference description 1]
2. [Difference description 2]
3. [Difference description 3]

## Modification Suggestions
1. [Specific suggestion 1]
2. [Specific suggestion 2]
3. [Specific suggestion 3]

Please ensure each suggestion is specific and executable, including specific CSS properties or HTML structure modification guidance."""

        return visual_analysis_prompt

    def format_suggestions(self, analysis_result: str) -> str:
        """
        Format suggestions into standard output format
        
        Args:
            analysis_result: VLM analysis result
            
        Returns:
            Formatted suggestions
        """
        formatted_suggestions = f"""## Visual Analysis Result

{analysis_result}

## Priority Suggestions
Based on the above analysis, it is recommended to make modifications in the following priority order:
1. First fix the most obvious layout differences
2. Adjust colors and font styles
3. Optimize the position and size of detail elements

Please modify the HTML and CSS code according to these suggestions."""

        return formatted_suggestions

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update agent state from environment
        
        Args:
            observation: Observation containing two image paths
            reward: Reward value
            done: Whether completed
            info: Additional information
        """
        if isinstance(observation, dict):
            # Expect observation to contain reference and current image paths
            if "reference_image" in observation and "current_image" in observation:
                reference_image = observation["reference_image"]
                current_image = observation["current_image"]
                
                # Generate visual analysis prompt
                visual_prompt = self.analyze_visual_differences(reference_image, current_image)
                formatted_observation = visual_prompt
                
            elif "task_description" in observation:
                # Initial task description
                formatted_observation = f"Task description: {observation['task_description']}\n\nPlease wait for reference design image and current implementation image for comparison."
            else:
                formatted_observation = str(observation)
        else:
            formatted_observation = str(observation)

        if done:
            return

        self.messages.append({"role": "user", "content": formatted_observation})
        self.current_observation = formatted_observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update agent state based on model response
        
        Args:
            response: Model response
            
        Returns:
            Action containing formatted suggestions
        """
        # Format suggestions
        formatted_suggestions = self.format_suggestions(response)
        
        self.messages.append({"role": "assistant", "content": response})

        # Create new step
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=formatted_suggestions,
            model_response=response,
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)

        return Action(action=formatted_suggestions)

    def reset(self):
        """Reset agent state"""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return chat completion history"""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Return trajectory object"""
        return self._trajectory

    def get_current_state(self) -> Step | None:
        """Return current step state"""
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]

    def get_suggestions_summary(self) -> str:
        """Get suggestions summary"""
        if not self._trajectory.steps:
            return "No suggestions generated yet."
        
        latest_step = self._trajectory.steps[-1]
        return latest_step.action
