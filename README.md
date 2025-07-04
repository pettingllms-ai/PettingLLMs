# Multi-Agent LLM Training Framework

## 📖 About The Project

This project is dedicated to exploring the training of Large Language Models (LLMs) in multi-agent environments. Our core approach involves leveraging agent symmetry and self-play techniques to train a single, robust policy for agents in symmetric games. We utilize state-of-the-art models and reinforcement learning algorithms to achieve intelligent and coordinated agent behavior.

## ✨ Core Components

Here is an overview of the current supported features and components of our framework.

* **Symmetry Strategy**
    * [] **Agent Symmetry:** Using self-play to train a single policy for symmetric games.

* **Training Environments**
    * [] Tic-Tac
    * [] Hanabi

* **Supported Models**
    * [] LLM-QWen-7b
    * [] LLM-QWen-0.5b

* **Data Modalities**
    * [] VLM (Vision Language Model)
    * [] LLM (Large Language Model)

* **Reinforcement Learning Algorithms**
    * [] PPO (Proximal Policy Optimization)

## 🗺️ Roadmap

We have an active development roadmap to enhance the capabilities of this framework. Our future goals include:

* [ ] **Model Expansion:**
    * Integrate a wider variety of open-source LLMs.
    * Develop and test novel model architectures specifically for multi-agent tasks.

* [ ] **Modality Enhancement:**
    * Expand to include other sensory modalities like audio or structured data.
    * Improve the fusion techniques between different modalities (e.g., VLM and LLM).

* [ ] **Algorithm Diversification:**
    * Implement and test other multi-agent RL algorithms (e.g., MADDPG, QMIX).
    * Explore curriculum learning and automated environment generation.

* [ ] **Environment Suite Growth:**
    * Add more complex and diverse game environments.
    * Develop environments for co-operative and competitive non-symmetric scenarios.

* [ ] **Evaluation and Benchmarking:**
    * Establish a comprehensive benchmark for evaluating multi-agent LLM performance.
    * Conduct extensive experiments and publish findings.
