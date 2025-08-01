<div align="center">
<h1 align="center">
OR-LLM-Agent: 基于推理大型语言模型的运筹优化问题建模与求解自动化框架
</h1>

[英文版本 English Version](./README.md)

<p align="center"> 
<a href="https://arxiv.org/abs/2503.10009" target="_blank"><img src="https://img.shields.io/badge/arXiv-论文-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://github.com/bwz96sco/or_llm_agent"><img src="https://img.shields.io/badge/GitHub-代码-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>  
<a href="https://youtu.be/O_0jd940nGk?si=c3JBRga1pJfI21wL" target="_blank">
<img src="https://img.shields.io/badge/YouTube-视频-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube">
</a>
<a href="https://huggingface.co/datasets/SJTU/BWOR" target="_blank">
<img src="https://img.shields.io/badge/HuggingFace-数据集-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
</a>
</p>

![](assets/dynamic.gif?autoplay=1)
</div>

<br>

## 摘要
随着人工智能（AI）的兴起，将大型语言模型（LLM）应用于数学问题求解正受到越来越多的关注。现有的大多数方法尝试通过提示工程或微调策略提升 LLM 在运筹优化（OR）问题求解中的表现。然而，这些方法受限于非推理型 LLM 的能力瓶颈。为突破这一限制，我们提出 **OR-LLM-Agent**，一个基于推理型 LLM 的 AI 智能体框架，用于自动化 OR 问题求解。该框架将任务分解为三个顺序阶段：数学建模、代码生成和调试。每个子任务均由专门的子智能体完成，从而实现更有针对性的推理。我们同时构建了 **BWOR** 数据集，用于评估 LLM 在 OR 任务上的表现。实验分析表明，在 NL4OPT、MAMO 和 IndustryOR 等基准中，推理型 LLM 在某些情况下甚至不及同族的非推理型模型。相比之下，BWOR 能够提供更稳定且更具区分度的评估结果。实验结果显示，OR-LLM-Agent（结合 DeepSeek-R1）相比包括 GPT-o3、Gemini 2.5 Pro、DeepSeek-R1 和 ORLM 在内的先进方法，在准确率上至少提升 7%。这证明了任务分解在 OR 问题求解中的有效性。
<br><br>

## 引言 📖

传统 OR 模型虽能提供定制化的解决方案，但代价高昂、执行缓慢，并且由于需要复杂的求解器专业知识，难以实施。

<img src="./assets/pic1_2.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

**OR-LLM-Agent** 是一个基于 LLM 的框架，可完全自动化 OR 优化过程：它能够将自然语言问题描述转换为数学模型，生成求解器代码，并执行求解以输出结果。

<img src="./assets/pic2_1.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

<br><br>

## 安装说明
### 依赖环境
- Python 3.8+
- Gurobi Optimizer

### 安装步骤
```bash
# 克隆仓库
git https://github.com/bwz96sco/or_llm_agent.git
cd or_llm_agent

# 安装依赖
pip install -r requirements.txt
```

### 快速开始
```bash
# 使用 o3-mini 模型直接评估默认数据集
python or_llm_eval_async_resilient.py

# 你也可以指定参数 (--math 启用数学建模智能体, --debug 启用调试智能体, --model 指定模型, --data_path 指定数据集路径)
python run_openrouter.py --math --debug --model deepseek/deepseek-r1-0528 --data_path data/datasets/IndustryOR.json

# 使用预设脚本批量运行评估
chmod +x run_eval_batch_agent.sh
bash ./run_eval_batch_agent.sh
```
请确保在 `.env` 文件中配置 OpenAI API key！

```bash
# 创建 .env 文件
cp .env.example .env
```

你需要在 `.env` 文件中设置 **OPENAI_API_KEY** 和 **OPENAI_API_BASE**（如果你使用 OpenAI 兼容服务）。  
若使用 **Claude 模型**，需额外设置 **CLAUDE_API_KEY**。  
若使用 **DeepSeek 模型**，推荐通过 **火山引擎（Volcengine）**，可参考 [官方教程](https://www.volcengine.com/docs/82379/1449737)。  
将 OPENAI_API_KEY 设置为火山引擎提供的 API Key，并将 OPENAI_API_BASE 设置为：
```
https://ark.cn-beijing.volces.com/api/v3
```

<br><br><br>

## 引用
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
由上海交通大学与南洋理工大学联合完成
</p>
