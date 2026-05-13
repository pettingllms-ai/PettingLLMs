"""Workflow orchestration for composing and executing node sequences."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Any, Optional, Callable
from workflow.core import WorkflowNode, Context, Message, MessageType
import logging


class Workflow:
    """Workflow orchestrator for composing and executing node sequences.

    Supports:
    - Sequential execution
    - Conditional branching
    - Loops
    - Error handling
    - Result tracking
    """

    def __init__(
        self,
        name: str = "workflow",
        ai_client: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        server_address: Optional[str] = None,
        ppo_trainer_config: Optional[Any] = None,
        enable_thinking: bool = False
    ):
        """Initialize workflow.

        Args:
            name: Workflow name
            ai_client: Optional AI client for making LLM requests (can be set later or per-node)
            tokenizer: Optional tokenizer for DataProto conversion
            server_address: Optional vLLM server address for verl integration
            ppo_trainer_config: Optional PPO trainer config for verl integration
            enable_thinking: Whether to enable thinking mode in chat templates
        """
        self.name = name
        self.nodes: List[WorkflowNode] = []
        self.logger = self._setup_logger()

        # verl integration support
        self.ai_client = ai_client
        self.tokenizer = tokenizer
        self.server_address = server_address
        self.ppo_trainer_config = ppo_trainer_config
        self.enable_thinking = enable_thinking

        # DataProto collection for training
        self.dataproto_list: List[Any] = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this workflow."""
        logger = logging.getLogger(f"Workflow.{self.name}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                f'[%(asctime)s] Workflow.{self.name} - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def add_node(self, node: WorkflowNode) -> 'Workflow':
        """Add a node to the workflow.
        
        Args:
            node: Node to add
            
        Returns:
            Self for chaining
        """
        self.nodes.append(node)
        return self
    
    def add_nodes(self, nodes: List[WorkflowNode]) -> 'Workflow':
        """Add multiple nodes to the workflow.
        
        Args:
            nodes: List of nodes to add
            
        Returns:
            Self for chaining
        """
        self.nodes.extend(nodes)
        return self
    
    def run(self, input_message: str, initial_state: Optional[Dict[str, Any]] = None) -> Message:
        """Run the workflow.

        Args:
            input_message: Initial input message
            initial_state: Initial workflow state

        Returns:
            Final message from workflow execution
        """
        # Initialize context
        context = Context()
        if initial_state:
            context.state = initial_state.copy()

        # Store workflow reference in context so nodes can access workflow properties
        context.state['workflow'] = self

        # Add initial message
        initial_msg = Message(
            content=input_message,
            message_type=MessageType.USER_INPUT
        )
        context.add_message(initial_msg)
        
        self.logger.info(f"Starting workflow with {len(self.nodes)} nodes")
        self.logger.info(f"Input: {input_message[:100]}...")
        
        # Execute nodes sequentially
        for i, node in enumerate(self.nodes):
            self.logger.info(f"Executing node {i+1}/{len(self.nodes)}: {node.name}")
            
            result = node(context)
            
            # Check for errors
            if result.message_type == MessageType.ERROR:
                self.logger.error(f"Node {node.name} returned error: {result.content}")
                return result
            
            self.logger.info(f"Node {node.name} completed successfully")
        
        # Return final result
        final_result = context.get_latest_message()
        self.logger.info("Workflow completed successfully")
        
        return final_result
    
    def run_with_context(self, context: Context) -> Message:
        """Run workflow with existing context.
        
        Args:
            context: Existing context to use
            
        Returns:
            Final message from workflow execution
        """
        self.logger.info(f"Starting workflow with {len(self.nodes)} nodes (existing context)")
        
        # Execute nodes sequentially
        for i, node in enumerate(self.nodes):
            self.logger.info(f"Executing node {i+1}/{len(self.nodes)}: {node.name}")
            
            result = node(context)
            
            # Check for errors
            if result.message_type == MessageType.ERROR:
                self.logger.error(f"Node {node.name} returned error: {result.content}")
                return result
            
            self.logger.info(f"Node {node.name} completed successfully")
        
        # Return final result
        final_result = context.get_latest_message()
        self.logger.info("Workflow completed successfully")
        
        return final_result


class ConditionalWorkflow(Workflow):
    """Workflow with conditional execution support.
    
    Allows nodes to be executed conditionally based on context state.
    """
    
    def __init__(self, name: str = "conditional_workflow"):
        """Initialize conditional workflow.
        
        Args:
            name: Workflow name
        """
        super().__init__(name)
        self.conditions: List[Optional[Callable[[Context], bool]]] = []
    
    def add_node(
        self,
        node: WorkflowNode,
        condition: Optional[Callable[[Context], bool]] = None
    ) -> 'ConditionalWorkflow':
        """Add a node with optional condition.
        
        Args:
            node: Node to add
            condition: Optional condition function. Node executes only if returns True
            
        Returns:
            Self for chaining
        """
        self.nodes.append(node)
        self.conditions.append(condition)
        return self
    
    def run(self, input_message: str, initial_state: Optional[Dict[str, Any]] = None) -> Message:
        """Run the conditional workflow.
        
        Args:
            input_message: Initial input message
            initial_state: Initial workflow state
            
        Returns:
            Final message from workflow execution
        """
        # Initialize context
        context = Context()
        if initial_state:
            context.state = initial_state.copy()
        
        # Add initial message
        initial_msg = Message(
            content=input_message,
            message_type=MessageType.USER_INPUT
        )
        context.add_message(initial_msg)
        
        self.logger.info(f"Starting conditional workflow with {len(self.nodes)} nodes")
        self.logger.info(f"Input: {input_message[:100]}...")
        
        # Execute nodes conditionally
        for i, (node, condition) in enumerate(zip(self.nodes, self.conditions)):
            # Check condition
            if condition is not None:
                try:
                    should_execute = condition(context)
                    if not should_execute:
                        self.logger.info(f"Skipping node {i+1}/{len(self.nodes)}: {node.name} (condition not met)")
                        continue
                except Exception as e:
                    self.logger.error(f"Error evaluating condition for node {node.name}: {e}")
                    continue
            
            self.logger.info(f"Executing node {i+1}/{len(self.nodes)}: {node.name}")
            
            result = node(context)
            
            # Check for errors
            if result.message_type == MessageType.ERROR:
                self.logger.error(f"Node {node.name} returned error: {result.content}")
                return result
            
            self.logger.info(f"Node {node.name} completed successfully")
        
        # Return final result
        final_result = context.get_latest_message()
        self.logger.info("Conditional workflow completed successfully")
        
        return final_result


class LoopWorkflow(Workflow):
    """Workflow that can loop nodes until a condition is met.
    
    Useful for iterative refinement and retry logic.
    """
    
    def __init__(
        self,
        name: str = "loop_workflow",
        max_iterations: int = 10
    ):
        """Initialize loop workflow.
        
        Args:
            name: Workflow name
            max_iterations: Maximum number of loop iterations
        """
        super().__init__(name)
        self.max_iterations = max_iterations
        self.loop_condition: Optional[Callable[[Context], bool]] = None
    
    def set_loop_condition(self, condition: Callable[[Context], bool]) -> 'LoopWorkflow':
        """Set the condition for continuing the loop.
        
        Args:
            condition: Function that returns True to continue loop, False to stop
            
        Returns:
            Self for chaining
        """
        self.loop_condition = condition
        return self
    
    def run(self, input_message: str, initial_state: Optional[Dict[str, Any]] = None) -> Message:
        """Run the loop workflow.
        
        Args:
            input_message: Initial input message
            initial_state: Initial workflow state
            
        Returns:
            Final message from workflow execution
        """
        if self.loop_condition is None:
            raise ValueError("Loop condition must be set before running LoopWorkflow")
        
        # Initialize context
        context = Context()
        if initial_state:
            context.state = initial_state.copy()
        
        # Add initial message
        initial_msg = Message(
            content=input_message,
            message_type=MessageType.USER_INPUT
        )
        context.add_message(initial_msg)
        
        self.logger.info(f"Starting loop workflow (max {self.max_iterations} iterations)")
        self.logger.info(f"Input: {input_message[:100]}...")
        
        # Loop execution
        for iteration in range(self.max_iterations):
            self.logger.info(f"Loop iteration {iteration + 1}/{self.max_iterations}")
            
            # Execute all nodes in sequence
            for i, node in enumerate(self.nodes):
                self.logger.info(f"Executing node {i+1}/{len(self.nodes)}: {node.name}")
                
                result = node(context)
                
                # Check for errors
                if result.message_type == MessageType.ERROR:
                    self.logger.error(f"Node {node.name} returned error: {result.content}")
                    return result
                
                self.logger.info(f"Node {node.name} completed successfully")
            
            # Check loop condition
            try:
                should_continue = self.loop_condition(context)
                if not should_continue:
                    self.logger.info(f"Loop condition not met, stopping after {iteration + 1} iterations")
                    break
            except Exception as e:
                self.logger.error(f"Error evaluating loop condition: {e}")
                break
        
        # Return final result
        final_result = context.get_latest_message()
        self.logger.info("Loop workflow completed")
        
        return final_result

