
<div align="center">
<h1 align="center">
OR-LLM-Agent：基于大型语言模型推理的运筹优化问题建模与求解自动化
</h1>

[英文版本 English Version](./README.md)

<p align="center"> <a href="https://arxiv.org/abs/2503.10009" target="_blank"><img src="https://img.shields.io/badge/arXiv-论文-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://github.com/bwz96sco/or_llm_agent"><img src="https://img.shields.io/badge/GitHub-代码-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>  </p>

![](assets/dynamic.gif?autoplay=1)
</div>



<br>

## 摘要
运筹学（Operations Research, OR）已被广泛应用于资源分配、生产计划、供应链管理等多个领域。然而，解决实际的运筹优化问题通常需要运筹专家进行数学建模，并由程序员开发求解算法。这种高度依赖专家的传统方法成本高、开发周期长，严重限制了运筹技术的普及应用。当前鲜有尝试使用人工智能（AI）替代专家实现运筹问题的全自动求解。为此，我们提出了 OR-LLM-Agent —— 一个可以实现真实运筹问题端到端自动求解的 AI 智能体。OR-LLM-Agent 利用大型语言模型（LLMs）的链式思维（Chain-of-Thought, CoT）推理能力，将自然语言描述的问题自动转化为数学模型，并自动生成 Gurobi 求解器代码。在 OR-LLM-Agent 中，我们设计了 OR-CodeAgent，用于在沙箱环境中自动执行与修复代码，从而求解最终结果。由于缺乏专门用于评估运筹自动化求解的基准数据集，我们构建了一个包含 83 个真实自然语言描述的运筹问题的基准集。我们与当前先进的推理型 LLM（如 GPT-o3-mini、DeepSeek-R1 和 Gemini 2.0 Flash Thinking）进行了对比实验。OR-LLM-Agent 实现了 100% 的最高通过率和 85% 的最高解答准确率，验证了自动化求解运筹问题的可行性。

<br><br>

## 简介 📖

传统上，企业依赖运筹专家为复杂优化问题开发专门模型，以确保解法的严谨性和针对性。然而，这种方式不仅成本高、速度慢，即便构建了良好模型，也常常在实现阶段面临困难——例如使用 Gurobi、CPLEX 等求解器时，需要具备高级编程和调试能力。

<img src="./assets/pic1_2.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

我们提出了 OR-LLM-Agent —— 一个基于推理型大型语言模型的运筹优化自动化框架。它可将自然语言描述的问题转化为数学模型，生成并执行求解代码，形成从问题描述到求解结果的全自动端到端流程。OR-LLM-Agent 包含以下模块：用户问题描述输入、LLM 数学建模、LLM 代码生成、以及 OR-CodeAgent。其中，LLM 数学建模模块构建线性规划模型；LLM 代码生成模块基于模型生成求解代码；OR-CodeAgent 实现代码的自动执行与修复，输出最终结果。整体框架如下图所示：

<img src="./assets/pic2_1.PNG" alt="or-llm-agent" width="1000" height="auto" div align=center>

<br><br>

## 安装方法
### 环境要求
- Python 3.8+
- Gurobi优化器

### 安装步骤
```bash
# 克隆代码库
git https://github.com/bwz96sco/or_llm_agent.git
cd or_llm_agent

# 安装依赖
pip install -r requirements.txt
```

### 快速开始
```bash
# 启动评估现有数据集
python or_llm_eval.py --agent

# 如果不指定 agent 参数，模型将直接用于求解
python or_llm_eval.py

# 你也可以指定使用的模型
python or_llm_eval.py --agent --model gpt-4o-mini-2024-07-18

# 使用异步方式并行运行任务
python or_llm_eval_async.py --agent
```

请确保在 `.env` 文件中设置你的 OpenAI API 密钥！

```bash
# 创建 .env 文件
cp .env.example .env
```

你需要设置 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`（如果使用 OpenAI 兼容服务）。如果希望使用 Claude 模型，则还需设置 `CLAUDE_API_KEY`。若使用 DeepSeek 模型，推荐通过火山引擎（Volcengine）接入（教程见 https://www.volcengine.com/docs/82379/1449737），设置 `OPENAI_API_KEY` 为其提供的 Api Key，并设置 `OPENAI_API_BASE` 为 `https://ark.cn-beijing.volces.com/api/v3`。

<br><br><br>

## 创建数据集

我们已在 `data/datasets` 目录下提供了一个数据集示例。你可以通过 `save_json.py` 脚本手动粘贴问题和答案，创建自己的数据集：

```python
# 将数据保存为 json 格式
python data/save_json.py
```

你也可以为数据集生成统计图：

```bash
# 获取问题长度分布
python data/question_length.py
```

<br><br>
## 设置 MCP 服务器与客户端

<div align="center">
<img src="assets/MCP.gif" alt="MCP Demo" width="800" height="auto">
</div>

我们还添加了一个模型上下文协议（Model Context Protocol，简称 MCP）服务器，以便更好地使用此工具。根据 Claude MCP 官网的官方文档，我们推荐使用 `uv` 包管理器来搭建 MCP 服务器。

```bash
# 创建虚拟环境并激活
uv venv
source .venv/bin/activate

# 安装依赖包
uv add -r requirements.txt
```

为了在 MCP 客户端中使用此功能，我们以 Claude 桌面客户端为例，首先你需要在 `claude_desktop_config.json` 中添加 MCP 路径：

```python
{
    "mcpServers": {
        "Optimization": {
            "command": "/{UV 安装文件夹的绝对路径}/uv",
            "args": [
                "--directory",
                "/{OR-LLM-AGENT 文件夹的绝对路径}",
                "run",
                "mcp_server.py.py"
            ]
        }
    }
}
```

然后你就可以打开 Claude 桌面客户端，检查锤子图标中是否出现了`get_operation_research_problem_answer` 项。

<img src="./assets/mcp_client.png" alt="mcp_client" width="1000" height="auto" div align=center>

<br><br>

## 作者
- **张博文**¹ * (bowen016@e.ntu.edu.sg)
- **罗鹏程**²³ * 

¹ 新加坡南洋理工大学  
² 上海交通大学宁波人工智能研究院  
³ 上海交通大学自动化系  
\* 共同第一作者

<br><br>

---
<p align="center">
项目由上海交通大学与新加坡南洋理工大学联合完成
</p>
