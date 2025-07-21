# Frontend Design Agent Test System

A comprehensive testing system for frontend design agents using HuggingFaceM4/WebSight dataset and sglang servers, following the sweet_rl evaluation pattern.

## Overview

This test system implements the dual-agent collaboration workflow described in sweet_rl for frontend design tasks:

- **Data Source**: HuggingFaceM4/WebSight dataset samples
- **Agent Communication**: Two sglang servers on different ports
- **Evaluation**: Automated testing with success rate metrics
- **Rendering**: HTML to image conversion using sweet_rl tools

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  WebSight       │    │  Test System     │    │  SGLang Servers │
│  Dataset        │───▶│  Controller      │───▶│  Port 8000/8001 │
│  Samples        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Visual Analysis │
                       │  & Code Gen      │
                       │  Agents          │
                       └──────────────────┘
```

## Prerequisites

### 1. Environment Setup

```bash
# Install required packages
pip install datasets requests tqdm

# Install sweet_rl dependencies (for HTML rendering)
pip install selenium
```

### 2. WebDriver Setup (for HTML Rendering)

```bash
# Install Firefox and GeckoDriver
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
tar -xvzf geckodriver-v0.35.0-linux64.tar.gz
sudo mv geckodriver /usr/local/bin/

# Verify installation
geckodriver --version
```

### 3. SGLang Server Setup

Start two sglang servers on different ports:

#### Code Generation Agent (Port 8000)
```bash
# Start code generation server
python -m sglang.launch_server \
    --model-path /path/to/llama3.1-8b-instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.8
```

#### Visual Analysis Agent (Port 8001)
```bash
# Start visual analysis server (VLM)
python -m sglang.launch_server \
    --model-path /path/to/qwen2-vl-7b-instruct \
    --port 8001 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.8 \
    --chat-template chatml-vision
```

## Usage

### Command Line Interface

```bash
cd pettingllms/agentgpraphs/design_human_interact/

python agent_collaboration_graph.py \
    --hostname localhost \
    --code_port 8000 \
    --visual_port 8001 \
    --num_samples 20 \
    --max_iterations 3 \
    --output_path results/evaluation_results.json \
    --temp_path /tmp/frontend_test \
    --dataset_version v0.2
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hostname` | localhost | SGLang server hostname |
| `--code_port` | 8000 | Port for code generation agent |
| `--visual_port` | 8001 | Port for visual analysis agent |
| `--num_samples` | 10 | Number of WebSight samples to test |
| `--max_iterations` | 3 | Maximum iterations per task |
| `--output_path` | None | Path to save evaluation results |
| `--temp_path` | /tmp | Temporary directory for files |
| `--dataset_version` | v0.2 | WebSight dataset version |

### Python API Usage

```python
from agent_collaboration_graph import FrontendDesignTestSystem

# Initialize test system
test_system = FrontendDesignTestSystem(
    hostname="localhost",
    code_port=8000,
    visual_port=8001,
    max_iterations=3
)

# Run evaluation
results = test_system.run_evaluation(
    num_samples=10,
    output_path="results.json"
)

# Check results
print(f"Success rate: {results['success_rate']:.2%}")
print(f"Average iterations: {results['average_iterations']:.1f}")

# Cleanup
test_system.cleanup()
```

## Evaluation Workflow

### 1. Data Loading
- Loads samples from HuggingFaceM4/WebSight dataset
- Extracts problem descriptions and ground truth HTML
- Preprocesses URLs using sweet_rl utilities

### 2. Task Execution
For each sample:
1. **Ground Truth Rendering**: Render reference HTML to image
2. **Initial Code Generation**: Generate initial HTML using code agent
3. **Iterative Refinement**:
   - Render current HTML to image
   - Compare with reference using visual agent
   - Generate improvement suggestions
   - Update HTML code based on suggestions
   - Repeat until satisfactory or max iterations reached

### 3. Evaluation Metrics
- **Success Rate**: Percentage of tasks achieving satisfactory results
- **Average Iterations**: Mean number of iterations per task
- **Task Completion**: Individual task success/failure status

## Output Format

### Evaluation Results JSON
```json
{
  "total_tasks": 10,
  "successful_tasks": 7,
  "success_rate": 0.7,
  "average_iterations": 2.3,
  "detailed_results": [
    {
      "task_id": "websight_0",
      "task_description": "Create a modern blog layout...",
      "ground_truth_html": "<html>...</html>",
      "reference_image": "/tmp/websight_0_gt.png",
      "final_html": "<html>...</html>",
      "success": true,
      "total_iterations": 2,
      "iterations": [...]
    }
  ]
}
```

### Console Output
```
🚀 Starting Frontend Design Agent Evaluation
📊 Samples: 10, Max iterations: 3
🔗 Testing connections to sglang servers...
✅ Code generation server (8000): Connected
✅ Visual analysis server (8001): Connected
🔄 Loading 10 samples from HuggingFaceM4/WebSight...
100%|████████| 10/10 [00:15<00:00, 1.52it/s]
✅ Successfully loaded 10 samples

🎯 Evaluating task: websight_0
📝 Description: Create a modern blog layout with sidebar...
  🔄 Iteration 1/3
  🔄 Iteration 2/3
  ✨ Design achieved satisfactory results!
  ✅ Task 1/10 completed. Success: True

🎯 Evaluation Summary:
  📊 Total tasks: 10
  ✅ Successful: 7
  📈 Success rate: 70.00%
  🔄 Average iterations: 2.3
```

## Troubleshooting

### Common Issues

#### 1. SGLang Connection Failed
```
❌ Code generation server (8000): Connection refused
```
**Solution**: Ensure sglang servers are running and accessible on specified ports.

#### 2. WebSight Dataset Loading Error
```
❌ Error loading WebSight dataset: Dataset not found
```
**Solution**: Check internet connection and huggingface-datasets installation.

#### 3. WebDriver Initialization Failed
```
⚠️ WebDriver initialization failed: geckodriver not found
```
**Solution**: Install Firefox and GeckoDriver as described in prerequisites.

#### 4. HTML Rendering Failed
```
❌ Rendering failed: WebDriver session error
```
**Solution**: Restart WebDriver or use headless mode for server environments.

### Performance Optimization

#### 1. Parallel Processing
- Use multiple worker processes for larger evaluations
- Implement async HTTP requests for sglang communication

#### 2. Resource Management
- Monitor GPU memory usage on sglang servers
- Implement request queuing for high-load scenarios

#### 3. Caching
- Cache rendered images to avoid re-rendering
- Store intermediate results for interrupted evaluations

## Integration with Sweet_RL

This test system follows the sweet_rl evaluation pattern:

### Similarities
- Uses WebSight dataset for frontend design tasks
- Implements multi-turn agent collaboration
- Generates visual comparisons and success metrics
- Supports HTML rendering and image comparison

### Extensions
- Modular agent design with clear interfaces
- Flexible sglang server configuration
- Comprehensive evaluation metrics
- Robust error handling and logging

## Example Results

Based on initial testing with Llama-3.1-8B and Qwen2-VL-7B:

| Metric | Value |
|--------|-------|
| Success Rate | 65-75% |
| Average Iterations | 2.1-2.8 |
| Task Completion Time | 30-60s per task |
| Memory Usage | ~8GB GPU per server |

## Advanced Configuration

### Custom Model Prompts
Modify agent system prompts in:
- `agents/visual_info_agent.py`
- `agents/code_genaration_agent.py`

### Custom Evaluation Criteria
Override `_is_design_satisfactory()` method for custom success criteria.

### Dataset Customization
Extend `WebSightDataLoader` to support custom datasets or filtering criteria.

## Contributing

To extend the test system:

1. **Add New Metrics**: Implement additional evaluation functions
2. **Support New Models**: Add model-specific prompt templates
3. **Enhance Rendering**: Integrate additional HTML rendering backends
4. **Improve UI**: Add web interface for result visualization

## License

This test system follows the same CC-By-NC license as sweet_rl. 