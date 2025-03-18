<h1 align="center">
OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problem with Reasoning Large Language Model
</h1>
<p align="center"> <a href="https://arxiv.org/abs/2503.10009" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://github.com/bwz96sco/or_llm_agent"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>  </p>

![](assets/dynamic.gif?autoplay=1)

<br>

## Abstract
Operations Research (OR) has been widely applied in various fields such as resource allocation, production planning, and supply chain management. However, addressing real-world OR problems requires OR experts to perform mathematical modeling and programmers to develop solution algorithms. This traditional method, heavily reliant on experts, is costly and has long development cycles, severely limiting the widespread adoption of OR techniques. Few have considered using Artificial Intelligence (AI) to replace professionals to achieve fully automated solutions for OR problems. We proposed OR-LLM-Agent, the AI agent that enables end-to-end automation for solving real-world OR problems. OR-LLM-Agent leverages the Chain-of-Thought (CoT) reasoning capabilities of Large Language Models (LLMs) to translate natural language problem descriptions into formal mathematical models and automatically generate Gurobi solver code. In OR-LLM-Agent, OR-CodeAgent is designed to automate code execution and repair within a sandbox environment, facilitating the derivation of the final solution. Due to the lack of dedicated benchmark datasets for evaluating the automated solving of OR problems, we construct a benchmark dataset comprising 83 real-world OR problems described in natural language. We conduct comparative experiments with state-of-the-art (SOTA) reasoning LLMs, including GPT-o3-mini, DeepSeek-R1, and Gemini 2.0 Flash Thinking. The OR-LLM-Agent achieved the highest pass rate of 100% and the highest solution accuracy of 85%, demonstrating the feasibility of automated OR problem-solving.
<br><br>

## Introduction 📖

Companies have traditionally relied on OR experts to develop specialized models for complex optimization problems, ensuring rigorous and tailored solutions. However, this approach is costly and slow, and even well-defined models face implementation challenges, as effective use of solvers like Gurobi and CPLEX demands advanced programming and debugging skills.

<img src="./assets/pic1_2.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

We propose OR-LLM-Agent, a reasoning LLM-based framework for fully automated OR optimization. OR-LLM-Agent converts a natural language problem description into a mathematical model, generates and executes the solution code, and thus facilitates an end-to-end automation pipeline from OR problem description to the solution. OR-LLM-Agent comprises modules for user problem description input, LLM mathematical modeling, LLM code generation, and OR-CodeAgent. The LLM mathematical modeling module constructs linear programming models for the OR problem. The LLM code generation module produces OR solver code based on the mathematical model. OR-CodeAgent ensures automated code execution and repair to obtain the final solution. The framework of OR-LLM-Agent is shown in following figure.

<img src="./assets/pic2_1.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

<br><br>

## Installation
### Prerequisites
- Python 3.8+

### Installation Steps
```bash
# Clone the repository
git https://github.com/bwz96sco/or_llm_agent.git
cd or_llm_agent

# Install package
pip install -r requirements.txt
```

### Getting Started
```bash
# Start evaluate current dataset
python or_llm_eval.py --agent

#if you do not sepcify agent argument, you can use the LLM to solve problems directly
python or_llm_eval.py

#you can also specify the model
python or_llm_eval.py --agent --model gpt-4o-mini-2024-07-18

#use async methods to run tasks parallelly
python or_llm_eval_async.py --agent
```
Make sure to setup your OpenAI API key in `.env` file!

```bash
#setup a .env file
cp .env.example .env
```

<br><br><br>


## Create Dataset

We have already set a dataset under the data/datasets directory for you, you can create your own dataset by the save_json.py, you need to paste your questions  and answers in the file manually:

```python
#save dataset to a json file
python data/save_json.py
```

You can also get stastic graph for datasets:

```bash
#get the length distribution
python data/question_length.py
```



<br><br>


## Authors
- **Bowen Zhang**¹ * (bowen016@e.ntu.edu.sg)
- **Pengcheng Luo**²³ * 

¹ Nanyang Technological University, Singapore.  
² Ningbo Artificial Intelligence Institute, Shanghai Jiao Tong University, Ningbo, PR, China.  
³ Department of Automation, Shanghai Jiao Tong University, Shanghai, PR China <br>
\* Equal contribution
<br><br>

---
<p align="center">
Made at Shanghai Jiao Tong University and Nanyang Technological University
</p>
