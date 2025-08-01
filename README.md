
<div align="center">
<h1 align="center">
OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problem with Reasoning Large Language Model
</h1>

[Chinese Version 中文版本](./README_CN.md)

<p align="center"> 
<a href="https://arxiv.org/abs/2503.10009" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://github.com/bwz96sco/or_llm_agent"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>  
<a href="https://youtu.be/O_0jd940nGk?si=c3JBRga1pJfI21wL" target="_blank">
<img src="https://img.shields.io/badge/YouTube-Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube">
</a>
<a href="https://huggingface.co/datasets/SJTU/BWOR" target="_blank">
<img src="https://img.shields.io/badge/HuggingFace-DataSet-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
</a>
</p>

![](assets/dynamic.gif?autoplay=1)
</div>



<br>

## Abstract
With the rise of artificial intelligence (AI), applying large language models (LLMs) to mathematical problem-solving has attracted increasing attention. Most existing approaches attempt to improve Operations Research (OR) optimization problem-solving through prompt engineering or fine-tuning strategies for LLMs. However, these methods are fundamentally constrained by the limited capabilities of non-reasoning LLMs. To overcome these limitations, we propose OR-LLM-Agent, an AI agent framework built on reasoning LLMs for automated OR problem solving. The framework decomposes the task into three sequential stages: mathematical modeling, code generation, and debugging. Each task is handled by a dedicated sub-agent, which enables more targeted reasoning. We also construct BWOR, an OR dataset for evaluating LLM performance on OR tasks. Our analysis shows that in the benchmarks NL4OPT, MAMO, and IndustryOR, reasoning LLMs sometimes underperform their non-reasoning counterparts within the same model family. In contrast, BWOR provides a more consistent and discriminative assessment of model capabilities. Experimental results demonstrate that OR-LLM-Agent utilizing DeepSeek-R1 in its framework outperforms advanced methods, including GPT-o3, Gemini 2.5 Pro, DeepSeek-R1, and ORLM, by at least 7\% in accuracy. These results demonstrate the effectiveness of task decomposition for OR problem solving.
<br><br>

## Introduction 📖

Traditional OR models deliver tailored solutions but are costly, slow, and difficult to implement due to complex solver expertise requirements.

<img src="./assets/pic1_2.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

OR-LLM-Agent is an LLM-based framework that fully automates OR optimization by converting natural language problem descriptions into models, generating solver code, and executing it to deliver solutions.

<img src="./assets/pic2_1.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

<br><br>

## Installation
### Prerequisites
- Python 3.8+
- Gurobi Optimizer

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
# Start to evaluate Default dataset directly with o3-mini model
python or_llm_eval_async_resilient.py

#You can also set some arguments(--math to enable math model agent, --debug to enable debugging agent, --model to specify model, --data_path to specify the path of dataset)
python run_openrouter.py --math --debug --model deepseek/deepseek-r1-0528 --data_path data/datasets/IndustryOR.json

#You can use the preset bash script to run evaluation in batch
chmod +x run_eval_batch_agent.sh
bash ./run_eval_batch_agent.sh
```
Make sure to setup your OpenAI API key in `.env` file!

```bash
#setup a .env file
cp .env.example .env
```

You need to set the OPENAI_API_KEY and OPENAI_API_BASE url(if you want to use OpenAI compatible service). You will also need to set the CLAUDE_API_KEY if you want to use claude model. If you want to use the DeepSeek model, I recommend you to use the Volcengine(you can get a tutorial from https://www.volcengine.com/docs/82379/1449737), set the OPENAI_API_KEY to Api Key provided by volcengine and set  OPENAI_API_BASE to
```
https://ark.cn-beijing.volces.com/api/v3
```

<br><br><br>

## Citation
```latex
@article{zhang2025or,
  title={Or-llm-agent: Automating modeling and solving of operations research optimization problem with reasoning large language model},
  author={Zhang, Bowen and Luo, Pengcheng},
  journal={arXiv preprint arXiv:2503.10009},
  year={2025}
}
```
---
<p align="center">
Made at Shanghai Jiao Tong University and Nanyang Technological University
</p>
