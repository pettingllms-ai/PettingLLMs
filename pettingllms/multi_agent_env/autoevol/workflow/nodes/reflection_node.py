"""Reflection node that enables agents to self-critique and refine outputs."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional
from workflow.core import WorkflowNode, Context, Message, MessageType


class ReflectionNode(WorkflowNode):
    """Reflection node that implements self-refinement through reflection.
    
    Process:
    1. Agent generates initial response
    2. Agent reflects on its response and identifies issues
    3. Agent refines response based on reflection
    4. Repeat for N iterations
    """
    
    def __init__(
        self,
        name: str,
        agent: WorkflowNode,
        num_iterations: int = 2,
        reflection_prompt: Optional[str] = None,
        **kwargs
    ):
        """Initialize the reflection node.
        
        Args:
            name: Node name
            agent: Agent node to use for generation and reflection
            num_iterations: Number of reflection iterations
            reflection_prompt: Custom reflection prompt template
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        
        self.agent = agent
        self.num_iterations = num_iterations
        self.reflection_prompt = reflection_prompt or (
            "Review your previous response:\n{response}\n\n"
            "Reflect on:\n"
            "1. Are there any errors or inaccuracies?\n"
            "2. Is the response complete and well-structured?\n"
            "3. Could the answer be improved?\n\n"
            "Provide a critique of your response."
        )
        self.refinement_prompt = (
            "Original question: {question}\n\n"
            "Your previous response:\n{response}\n\n"
            "Your reflection:\n{reflection}\n\n"
            "Based on your reflection, provide an improved response."
        )
    
    def _reflect(self, context: Context, question: str, response: str) -> str:
        """Generate reflection on a response.
        
        Args:
            context: Workflow context
            question: Original question
            response: Response to reflect on
            
        Returns:
            Reflection text
        """
        reflection_context = Context()
        reflection_context.state = context.state.copy()
        
        reflection_input = Message(
            content=self.reflection_prompt.format(response=response),
            message_type=MessageType.USER_INPUT
        )
        reflection_context.add_message(reflection_input)
        
        result = self.agent(reflection_context)
        
        if result.message_type == MessageType.ERROR:
            self.logger.warning(f"Reflection failed: {result.content}")
            return "Unable to generate reflection due to error."
        
        return result.content
    
    def _refine(self, context: Context, question: str, response: str, reflection: str) -> str:
        """Refine response based on reflection.
        
        Args:
            context: Workflow context
            question: Original question
            response: Previous response
            reflection: Reflection on the response
            
        Returns:
            Refined response
        """
        refinement_context = Context()
        refinement_context.state = context.state.copy()
        
        refinement_input = Message(
            content=self.refinement_prompt.format(
                question=question,
                response=response,
                reflection=reflection
            ),
            message_type=MessageType.USER_INPUT
        )
        refinement_context.add_message(refinement_input)
        
        result = self.agent(refinement_context)
        
        if result.message_type == MessageType.ERROR:
            self.logger.warning(f"Refinement failed: {result.content}")
            return response  # Return previous response if refinement fails
        
        return result.content
    
    def process(self, context: Context) -> Message:
        """Process context through reflection and refinement.
        
        Args:
            context: Workflow context
            
        Returns:
            Message containing refined response
        """
        self.logger.info(f"Starting reflection with {self.num_iterations} iterations")
        
        # Get original question
        original_input = context.get_latest_message()
        if not original_input:
            return Message(
                content={"error": "No input message found"},
                message_type=MessageType.ERROR
            )
        
        question = original_input.content
        
        # Generate initial response
        self.logger.info("Generating initial response")
        initial_context = Context()
        initial_context.state = context.state.copy()
        initial_context.add_message(original_input)
        
        initial_result = self.agent(initial_context)
        
        if initial_result.message_type == MessageType.ERROR:
            return initial_result
        
        current_response = initial_result.content
        
        # Store history for metadata
        history = [{
            "iteration": 0,
            "response": current_response,
            "reflection": None
        }]
        
        # Reflection iterations
        for iteration in range(self.num_iterations):
            self.logger.info(f"Reflection iteration {iteration + 1}/{self.num_iterations}")
            
            # Reflect on current response
            reflection = self._reflect(context, question, current_response)
            
            self.logger.info(f"Generated reflection: {reflection[:100]}...")
            
            # Refine based on reflection
            refined_response = self._refine(context, question, current_response, reflection)
            
            self.logger.info(f"Generated refined response: {refined_response[:100]}...")
            
            # Update current response
            current_response = refined_response
            
            # Store in history
            history.append({
                "iteration": iteration + 1,
                "response": current_response,
                "reflection": reflection
            })
        
        return Message(
            content=current_response,
            message_type=MessageType.AGENT_RESPONSE,
            metadata={
                "num_iterations": self.num_iterations,
                "history": history
            }
        )

