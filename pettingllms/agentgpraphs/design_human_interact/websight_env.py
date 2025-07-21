"""
Environment for web design tasks that handles both code generation and visual analysis.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING

from datasets import load_dataset  # type: ignore
from tqdm import tqdm
from pettingllms.environments.base.multi_turn_env import MultiTurnEnvironment
from sweet_rl.sweet_rl.utils.webpage_utils import render_full_html, get_driver, replace_urls

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from torch import Tensor
    from PIL.Image import Image as PILImage

try:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("⚠️ Some required packages are missing. Please run 'pip install torch Pillow transformers'")
    torch = None
    Image = None
    CLIPModel = None
    CLIPProcessor = None

class WebEnv(MultiTurnEnvironment):
    """
    Environment for web design tasks with dual-agent interaction.
    """

    def __init__(self, task: Optional[Dict] = None, max_turns: int = 3, temp_path: Optional[str] = None):
        """
        Initialize the web design environment.

        Args:
            task: Dictionary containing the task information
            max_turns: Maximum number of turns before terminating
            temp_path: Path for temporary files
        """
        super().__init__(task=task, max_turns=max_turns)
        self.temp_path = temp_path
        self.current_html: Optional[str] = None
        self.driver = None
        self.reference_image: Optional[str] = None
        self.current_image: Optional[str] = None
        self.clip_model: Optional[Any] = None
        self.clip_processor: Optional[Any] = None
        
        # Initialize CLIP model and processor
        if torch is not None and CLIPModel is not None and CLIPProcessor is not None:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("✅ CLIP model initialized successfully")
            except Exception as e:
                print(f"⚠️ CLIP model initialization failed: {e}")
                self.clip_model = None
                self.clip_processor = None
        else:
            print("⚠️ CLIP dependencies not available")
            self.clip_model = None
            self.clip_processor = None
        
        # Initialize webdriver
        try:
            self.driver = get_driver()
            print("✅ WebDriver initialized successfully")
        except Exception as e:
            print(f"⚠️ WebDriver initialization failed: {e}")
            self.driver = None

    def get_reward_and_next_obs(self, task: Dict, action: str) -> Tuple[float, Dict]:
        """
        Compute reward and next observation. This is a placeholder implementation
        as the actual reward is handled in the step function.
        
        Args:
            task: Task dictionary
            action: Action string
            
        Returns:
            Tuple of (reward, next_observation)
        """
        return 0.0, {}

    def load_dataset(self, num_samples: int = 100, version: str = "v0.2") -> List[Dict]:
        """
        Load samples from WebSight dataset
        
        Args:
            num_samples: Number of samples to load
            version: Dataset version
            
        Returns:
            List of formatted samples
        """
        print(f"🔄 Loading {num_samples} samples from HuggingFaceM4/WebSight...")
        
        try:
            ds = load_dataset("HuggingFaceM4/WebSight", version)["train"]
            
            samples = []
            for i in tqdm(range(min(num_samples, len(ds)))):
                sample = {
                    "task_id": f"websight_{i}",
                    "problem_description": ds[i]["llm_generated_idea"],
                    "ground_truth": replace_urls(ds[i]["text"]),
                    "original_index": i
                }
                samples.append(sample)
                
            print(f"✅ Successfully loaded {len(samples)} samples")
            return samples
            
        except Exception as e:
            print(f"❌ Error loading WebSight dataset: {e}")
            return []

    def reset(self, task=None, seed=None):
        """Reset the environment and return initial observations."""
        import random
        import os

        if seed is not None:
            random.seed(seed)

        if task is not None:
            self.task = task

        assert self.task is not None, "Task must be set before reset"

        self.done = False
        self.current_turn = 0
        self.history = []
        self.current_html = None
        
        # Render ground truth image
        if self.driver:
            try:
                self.reference_image = render_full_html(
                    self.driver,
                    self.task["ground_truth"],
                    self.temp_path,
                    env_id=self.task["task_id"]  # type: ignore
                )
            except Exception as e:
                print(f"❌ Ground truth rendering failed: {e}")
                self.reference_image = None

        return {
            "task_description": self.task["problem_description"],
            "reference_image": self.reference_image
        }, {}

    def render_html(self, html_code: str, iteration: int = 0) -> Optional[str]:
        """
        Render HTML code to image
        
        Args:
            html_code: HTML code to render
            iteration: Current iteration number
            
        Returns:
            Path to rendered image file
        """
        if not self.driver:
            print("⚠️ WebDriver not available, skipping render")
            return None
            
        try:
            image_path = render_full_html(
                self.driver,
                html_code,
                self.temp_path,
                env_id=f"{self.task['task_id']}_{iteration}"  # type: ignore
            )
            return image_path
        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            return None

    def step(self, role: str, action: str):
        """
        Take a step in the environment based on the role and action.

        Args:
            role: Either 'code' or 'visual' indicating the agent role
            action: Response string from the agent

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append({"role": role, "action": action})

        # Handle code generation role
        if role == "code":
            self.current_html = action
            self.current_image = self.render_html(action, self.current_turn)
            
            # Calculate CLIP-based reward if both images are available
            reward = 0.0
            if (self.current_image and self.reference_image and 
                self.clip_model is not None and self.clip_processor is not None and 
                Image is not None and torch is not None):
                try:
                    # Load and process images
                    current_img = Image.open(self.current_image).convert("RGB")
                    reference_img = Image.open(self.reference_image).convert("RGB")
                    
                    # Process images through CLIP
                    inputs1 = self.clip_processor(images=current_img, return_tensors="pt", padding=True).to("cuda")
                    inputs2 = self.clip_processor(images=reference_img, return_tensors="pt", padding=True).to("cuda")
                    
                    # Get image embeddings
                    with torch.no_grad():  # type: ignore
                        image_features1 = self.clip_model.get_image_features(**inputs1)
                        image_features2 = self.clip_model.get_image_features(**inputs2)
                    
                    # Normalize embeddings
                    image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
                    image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)
                    
                    # Calculate cosine similarity as reward
                    reward = float(torch.sum(image_features1 * image_features2, dim=-1).cpu().numpy())  # type: ignore
                except Exception as e:
                    print(f"⚠️ Error calculating CLIP reward: {e}")
                    reward = 0.0
            
            next_obs = {
                "reference_image": self.reference_image,
                "current_image": self.current_image
            }
            
        # Handle visual analysis role
        elif role == "visual":
            suggestions = action
            next_obs = {
                "task_description": self.task["problem_description"],  # type: ignore
                "visual_suggestions": suggestions
            }
            reward = 0.0  # Visual analysis role doesn't get immediate reward
            
        else:
            raise ValueError(f"Invalid role: {role}")

        # Increment turn counter
        self.current_turn += 1
        
        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True

        return next_obs, reward, self.done, self.task

    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
                print("✅ WebDriver closed successfully")
            except Exception as e:
                print(f"⚠️ Error closing WebDriver: {e}")

    @staticmethod
    def from_dict(env_args: dict) -> "WebEnv":
        return WebEnv(
            task=env_args["task"],
            max_turns=env_args.get("max_turns", 3),
            temp_path=env_args.get("temp_path")
        )
