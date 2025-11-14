import logging
from typing import Any
import os
import json
import shutil
import logging
from pettingllms.multi_agent_env.pychecker_rl.pychecker_utils import (
    extract_code_from_response,
    create_task_folder
)
from pettingllms.multi_agent_env.pychecker_rl.agents.prompt import CMB_SYSTEM_PROMPT, CMB_GENERATION_PROMPT, SEQ_SYSTEM_PROMPT, SEQ_GENERATION_PROMPT, CMB_CHECKER_TAIL, SEQ_CHECKER_TAIL, CMB_PythonHeader, SEQ_PythonHeader
from pettingllms.multi_agent_env.pychecker_rl.pychecker_worker import _await_ray_object_ref
from pettingllms.multi_agent_env.base.agent import Agent

logger = logging.getLogger(__name__)
class PyCheckerAgent(Agent):
    """
    Agent specialized for generating Python GoldenDUT code for hardware verification
    """
    def __init__(self, env_idx: int = 0, agent_sample_idx: int = 0, **kwargs):
        """
        Initialize PyChecker agent
        
        Args:
            env_idx: Environment index (used as rollout_idx)
            agent_sample_idx: Agent sample index
            **kwargs: Additional keyword arguments (benchmark, etc.)
        """
        self.env_idx = env_idx
        self.rollout_idx = env_idx  # Use env_idx as rollout_idx
        self.agent_sample_idx = agent_sample_idx
        self.task_folder = None
        self.stimulus_json_path = None
        self.current_action = ""
        self.current_prompt = None
        self.current_response = None
        self.env_data = None
        self.agent_reward = 0.0
        self.reward_history = []
        
        # Set any additional keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        logger.info(f"PyCheckerAgent initialized: env_idx={env_idx}, rollout_idx={self.rollout_idx}, agent_sample_idx={agent_sample_idx}")

    def reset(self, env_data):
        """
        Reset agent state for new episode
        
        Args:
            env_data: Environment data object containing state
        """
        if hasattr(env_data, 'state'):
            self.task_folder = env_data.state.task_folder
            self.stimulus_json_path = env_data.state.stimulus_json_path
            logger.info(f"PyCheckerAgent reset: task_folder={self.task_folder}")
        else:
            logger.warning("env_data has no state attribute")

    def update_from_env(self, turn_idx: int, env_data):
        """
        Update agent state from environment and generate prompt for Python GoldenDUT generation
        
        Args:
            turn_idx: Current turn index
            env_data: Environment data object
        """
        self.env_data = env_data
        state = getattr(env_data, "state", None)
        
        if state is not None:
            self.task_folder = getattr(state, "task_folder", None)
            self.stimulus_json_path = getattr(state, "stimulus_json_path", None)
            
            # Generate prompt using create_prompt method
            formatted_prompt = self.create_prompt(env_data)
            self.current_prompt = {"text": formatted_prompt, "image": None}
            
            logger.info(f"PyCheckerAgent updated from env: turn_idx={turn_idx}, task_folder={self.task_folder}")
        else:
            logger.warning("env_data has no state attribute")

    def create_context(self, env_data) -> dict:
        """
        Create context for agent prompt

        Args:
            env_data: Environment data object

        Returns:
            Dictionary containing context information
        """
        context = {}
        if hasattr(env_data, 'state'):
            state = env_data.state
            context['problem_input'] = state.problem_input
            context['spec'] = state.spec
            context['circuit_type'] = state.circuit_type
            context['stimulus_json_path'] = state.stimulus_json_path

            # Add history if available
            if hasattr(state, 'generated_python_code_list') and state.generated_python_code_list:
                context['previous_attempts'] = state.generated_python_code_list

        return context

    def create_prompt(self, env_data) -> str:
        """
        Create prompt for agent based on circuit type

        Args:
            env_data: Environment data object

        Returns:
            Formatted prompt string
        """
        context = self.create_context(env_data)
        circuit_type = context.get('circuit_type', 'CMB')
        problem_input = context.get('problem_input', '')
        spec = context.get('spec', '')

        if circuit_type == "CMB":
            system_prompt = CMB_SYSTEM_PROMPT
            generation_prompt = CMB_GENERATION_PROMPT
        else:
            system_prompt = SEQ_SYSTEM_PROMPT
            generation_prompt = SEQ_GENERATION_PROMPT

        # Format the generation prompt with description and module_header
        formatted_generation_prompt = generation_prompt.format(
            description=problem_input,
            module_header=spec
        )

        prompt = f"{system_prompt}\n\n{formatted_generation_prompt}"

        # Add previous attempts if any
        previous_attempts = context.get('previous_attempts', [])
        if previous_attempts:
            prompt += "\n\nPrevious attempts (that failed):\n"
            for i, attempt in enumerate(previous_attempts[-3:]):  # Show last 3 attempts
                prompt += f"\n--- Attempt {i+1} ---\n{attempt}\n"

        return prompt

    def update_from_model(self, response: str):
        """
        Parse the LLM response and extract Python GoldenDUT code
        
        Args:
            response: LLM response string
            
        Returns:
            Extracted Python code
        """
        self.current_response = response
        self.current_action = extract_code_from_response(response, code_type="python")
        logger.info(f"PyCheckerAgent extracted code: {len(self.current_action)} characters")
        return self.current_action

    async def step(self, env_data, env_worker: Any = None):
        """
        Process the generated Python code and evaluate it through full workflow

        Workflow:
        1. Prepare task_folder with required files
        2. Execute Python GoldenDUT with stimulus.json
        3. Run Verilog simulation
        4. Calculate reward

        Args:
            env_data: Environment data object
            env_worker: Optional worker for async execution
        """
        generated_code = self.current_action
        env_data.state.generated_python_code = generated_code
        task_folder = env_data.state.task_folder

        # Initialize info dict
        info = {
            'code_extracted': False,
            'code_runs': False,
            'sim_success': False,
            'output_matches': False,
            'error_message': ''
        }

        reward = 0.0
        self.agent_reward = 0.0

        # Check if code was extracted
        if not generated_code or len(generated_code.strip()) < 10:
            info['error_message'] = 'No GoldenDUT code extracted from LLM response'
            logger.warning(f"GoldenDUT code extraction failed for rollout {self.rollout_idx}")
            self.agent_reward = 0.0

            # Update state with agent-specific fields
            env_data.state.pychecker_extracted = info['code_extracted']
            env_data.state.pychecker_runs = info['code_runs']
            env_data.state.pychecker_sim_success = info['sim_success']
            env_data.state.pychecker_matches = info['output_matches']
            env_data.state.pychecker_error_message = info['error_message']

            logger.info(f"PyCheckerAgent step completed: reward={reward}, info={info}")
            return

        info['code_extracted'] = True

        # Update history
        if generated_code:
            env_data.state.generated_python_code_list.append(generated_code)

        circuit_type = env_data.state.circuit_type
        if circuit_type == "CMB":
            python_header = CMB_PythonHeader
            tail = CMB_CHECKER_TAIL
        else:
            python_header = SEQ_PythonHeader
            tail = SEQ_CHECKER_TAIL

        # Write golden_dut.py with header and tail
        full_python_code = python_header + "\n" + generated_code + "\n" + tail
        golden_dut_path = os.path.join(task_folder, "golden_dut.py")
        with open(golden_dut_path, "w") as f:
            f.write(full_python_code)

        # Write circuit_info.json for workflow functions
        circuit_info = {
            "circuit_type": circuit_type,
            "verilog_dut": env_data.state.golden_output
        }
        circuit_info_path = os.path.join(task_folder, "circuit_info.json")
        with open(circuit_info_path, "w") as f:
            json.dump(circuit_info, f, indent=2)
        # Use the Ray actor handle directly (align with code worker structure)
        ray_actor = env_worker
        
        # Step 1: Execute Python golden_dut.py to generate testbench
        python_success, python_error = await _await_ray_object_ref(
            ray_actor.run_python_file.remote(
                python_file_path="golden_dut.py",
                working_directory=task_folder,
                timeout=60.0
            ), 90.0
        )
        
        if not python_success:
            info['error_message'] = f"Python execution failed: {python_error}"
            logger.warning(f"Python execution failed: {python_error}")
            reward = 0.0
            self.agent_reward = 0.0
        else:
            info['code_runs'] = True

            # Write top.v file (Verilog DUT)
            top_v_path = os.path.join(task_folder, "top.v")
            spec_path = os.path.join(task_folder, "spec.txt")
            with open(spec_path, "w") as f:
                f.write(env_data.state.problem_input)
            with open(top_v_path, "w") as f:
                verilog_code = env_data.state.golden_output
                # Ensure file ends with newline (POSIX requirement for Verilator)
                if verilog_code and not verilog_code.endswith('\n'):
                    verilog_code += '\n'
                f.write(verilog_code)

            # Step 2: Run Verilog simulation
            sim_success, sim_error, sim_results = await _await_ray_object_ref(
                ray_actor.run_verilog_simulation.remote(
                    task_folder=task_folder,
                    circuit_type=circuit_type,
                    timeout=120.0
                ), 180.0
            )

            # Calculate reward based on simulation results
            if not sim_success:
                info['error_message'] = sim_error
                logger.warning(f"Simulation failed: {sim_error}")
                # Python file was generated but simulation failed
                reward = 0.3
                self.agent_reward = 0.3
            else:
                # Simulation ran successfully
                # Check simulation results for logging
                compile_success = sim_results.get('compile_success', False)
                all_tests_passed = sim_results.get('all_tests_passed', False)
                

                if all_tests_passed:
                    # All tests passed
                    info['output_matches'] = True
                    reward = 1.0
                    self.agent_reward = 1.0
                    logger.info(f"All tests passed for rollout {self.rollout_idx}, reward={reward}")
                elif compile_success:
                    # Simulation ran but some tests failed - python file generated
                    info['error_message'] = "Some tests failed"
                    reward = 0.3
                    self.agent_reward = 0.3
                    logger.info(f"Some tests failed for rollout {self.rollout_idx}, reward={reward}")
                    info['sim_success'] = True

                    
                else:
                    # Simulation failed to run properly - python file generated
                    info['error_message'] = "Simulation failed to run properly"
                    reward = 0.3
                    self.agent_reward = 0.3
                    logger.warning(f"Simulation failed to run properly for rollout {self.rollout_idx}, reward={reward}")

              
        
        # Update state with agent-specific fields
        env_data.state.pychecker_extracted = info['code_extracted']
        env_data.state.pychecker_runs = info['code_runs']
        env_data.state.pychecker_sim_success = info['sim_success']
        env_data.state.pychecker_matches = info['output_matches']
        env_data.state.pychecker_error_message = info['error_message']
        
        logger.info(f"PyCheckerAgent step completed: reward={reward}, info={info}")

    def calculate_reward(self, env_data):
        """
        Calculate reward (already done in step)
        """
        self.agent_reward = self.agent_reward
        self.reward_history.append(self.agent_reward)
        self.success = (self.agent_reward >= 1.0)
        
        

    def get_observation(self, env_data):
        """
        Get observation for next turn
        
        Args:
            env_data: Environment data object
            
        Returns:
            Observation string or None
        """
        if hasattr(env_data, 'state') and hasattr(env_data.state, 'error_message'):
            return env_data.state.error_message
        return None


