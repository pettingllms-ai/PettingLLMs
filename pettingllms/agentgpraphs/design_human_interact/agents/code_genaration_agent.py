import copy
import logging
import re
from typing import Any, Dict, Optional

from pettingllms.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


def extract_html_snippet(paragraph: str) -> tuple[str, str]:
    """
    Extract HTML code snippet from text
    
    Args:
        paragraph: Text containing HTML code
        
    Returns:
        tuple: (processed text, HTML code snippet)
    """
    # Replace image URLs to standard format
    paragraph = replace_urls(paragraph)
    
    # Match complete HTML content
    html_pattern = r"<html.*?>.*?</html>"
    match = re.search(html_pattern, paragraph, re.DOTALL)

    if match:
        return paragraph.replace(match.group(0), "[SEE RENDERED HTML]"), match.group(0)
    else:
        # Try to match body tag
        html_pattern = r"<body.*?>.*?</body>"
        match = re.search(html_pattern, paragraph, re.DOTALL)
        if match:
            return paragraph.replace(match.group(0), "[SEE RENDERED HTML]"), match.group(0)
        else:
            return paragraph, None


def replace_urls(text: str) -> str:
    """
    Replace image URLs to standard format
    
    Args:
        text: Text containing URLs
        
    Returns:
        Text after replacement
    """
    # Replace unsplash random image URLs
    pattern = r"https://source\.unsplash\.com/random/(\d+)x(\d+)/\?[\w=]+"
    
    def replace_match(match):
        width, height = match.groups()
        return f"https://picsum.photos/id/48/{width}/{height}"

    new_text = re.sub(pattern, replace_match, text)

    # Ensure all picsum images use id 48
    pattern = r"https://picsum\.photos/(\d+)/(\d+)"
    replacement = r"https://picsum.photos/id/48/\1/\2"
    new_text = re.sub(pattern, replacement, new_text)

    return new_text


class CodeGenerationAgent(BaseAgent):
    """
    HTML code generation agent that generates webpage code based on requirements and visual suggestions.
    Specialized for generating HTML and Tailwind CSS code.
    """

    def __init__(self, max_iterations=10):
        """
        Initialize CodeGenerationAgent.
        
        Args:
            max_iterations: Maximum number of iterations
        """
        self.system_prompt = """You are a professional frontend developer. Your task is to generate high-quality webpage code based on user requirements and visual suggestions.

Technical Requirements:
1. Build webpages using HTML and Tailwind CSS
2. Code must be wrapped in <html> tags
3. Use https://picsum.photos/id/48/width/height format for images
4. Keep id=48 to ensure image consistency
5. Write real and detailed business content

Workflow:
1. Understand user requirement description
2. Analyze modification suggestions from visual feedback
3. Generate or modify corresponding HTML/CSS code
4. Ensure code follows best practices and accessibility standards

Output Format:
- First briefly explain your design approach
- Then output "OUTPUT:" 
- Finally provide complete HTML code (wrapped in <html> tags)

Notes:
- Each response contains only one HTML code snippet
- Code should be complete and runnable
- Follow responsive design principles
- Ensure good user experience"""
        
        self._trajectory = Trajectory()
        self.messages = []
        self.max_iterations = max_iterations
        self.current_observation = None
        self.iteration_count = 0

    def format_requirements(self, task_description: str, visual_suggestions: str = None) -> str:
        """
        Format requirements and suggestions into code generation prompt
        
        Args:
            task_description: Task description
            visual_suggestions: Visual suggestions
            
        Returns:
            Formatted prompt
        """
        prompt = f"## Project Requirements\n{task_description}\n\n"
        
        if visual_suggestions:
            prompt += f"## Visual Feedback and Modification Suggestions\n{visual_suggestions}\n\n"
            prompt += "Please modify the existing code based on the above feedback to ensure the generated webpage matches the expected design.\n\n"
        else:
            prompt += "Please generate an initial webpage design based on the requirements.\n\n"
            
        prompt += """Please provide:
1. Brief design approach explanation
2. Complete HTML code (using Tailwind CSS)

Output Format:
Design Approach: [Your approach]

OUTPUT:
<html>
... your code ...
</html>"""

        return prompt

    def validate_html_code(self, html_code: str) -> tuple[bool, str]:
        """
        Validate basic format of HTML code
        
        Args:
            html_code: HTML code
            
        Returns:
            tuple: (is valid, error message)
        """
        if not html_code:
            return False, "HTML code is empty"
            
        if not html_code.strip().startswith('<html'):
            return False, "HTML code must start with <html> tag"
            
        if not html_code.strip().endswith('</html>'):
            return False, "HTML code must end with </html> tag"
            
        # Check basic structure
        if '<head>' not in html_code:
            return False, "Missing <head> tag"
            
        if '<body>' not in html_code:
            return False, "Missing <body> tag"
            
        # Check Tailwind CSS CDN
        if 'tailwindcss' not in html_code.lower():
            return False, "Recommend including Tailwind CSS CDN link"
            
        return True, "HTML code format is correct"

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update agent state from environment
        
        Args:
            observation: Observation containing task description or visual suggestions
            reward: Reward value
            done: Whether completed
            info: Additional information
        """
        if isinstance(observation, dict):
            if "task_description" in observation:
                # Initial task description
                task_desc = observation["task_description"]
                visual_suggestions = observation.get("visual_suggestions", None)
                formatted_observation = self.format_requirements(task_desc, visual_suggestions)
                
            elif "visual_suggestions" in observation:
                # Visual suggestions update
                visual_suggestions = observation["visual_suggestions"]
                task_desc = observation.get("task_description", "Continue optimizing webpage design based on suggestions")
                formatted_observation = self.format_requirements(task_desc, visual_suggestions)
                
            elif "feedback" in observation:
                # General feedback
                formatted_observation = f"Feedback: {observation['feedback']}\n\nPlease adjust the code based on the feedback."
                
            else:
                formatted_observation = str(observation)
        else:
            formatted_observation = str(observation)

        if done:
            return

        # Increment iteration count
        self.iteration_count += 1
        
        self.messages.append({"role": "user", "content": formatted_observation})
        self.current_observation = formatted_observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update agent state based on model response
        
        Args:
            response: Model response
            
        Returns:
            Action containing generated code
        """
        # Extract HTML code
        content = response
        action = response

        # Handle OUTPUT: format
        if "OUTPUT:" in response:
            parts = response.split("OUTPUT:")
            if len(parts) >= 2:
                thought = parts[0].strip()
                action = parts[1].strip()
                # Record complete response but use extracted code as action
                content = response
        
        # Extract HTML code snippet
        _, html_snippet = extract_html_snippet(action)
        
        if html_snippet:
            # Validate HTML code
            is_valid, validation_msg = self.validate_html_code(html_snippet)
            
            if is_valid:
                final_action = html_snippet
            else:
                final_action = f"Code validation failed: {validation_msg}\n\nOriginal code:\n{html_snippet}"
        else:
            final_action = action

        self.messages.append({"role": "assistant", "content": content})

        # Create new step
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=final_action,
            model_response=response,
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)

        return Action(action=final_action)

    def reset(self):
        """Reset agent state"""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None
        self.iteration_count = 0

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

    def get_generated_html(self) -> str:
        """Get generated HTML code"""
        if not self._trajectory.steps:
            return "No code generated yet."
        
        latest_step = self._trajectory.steps[-1]
        return latest_step.action

    def has_reached_max_iterations(self) -> bool:
        """Check if maximum iterations reached"""
        return self.iteration_count >= self.max_iterations

    def get_iteration_summary(self) -> str:
        """Get iteration summary"""
        return f"Current iteration: {self.iteration_count}/{self.max_iterations}"
