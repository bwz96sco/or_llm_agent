
<div align="center">
<h1 align="center">
OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problem with Reasoning Large Language Model
</h1>

[Chinese Version 中文版本](./README_MCP_CN.md)

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

## Set up MCP Server & Client

<div align="center">
<img src="../assets/MCP.gif" alt="MCP Demo" width="800" height="auto">
</div>

We also add a Model Context Protocol(MCP) server to facilitate the utilization of this tool. According to the official document from claude MCP website, we recommend using the `uv` package manager to set up the MCP server.

```bash
# Create virtual environment and activate it
uv venv
source .venv/bin/activate

#install package
uv add -r requirements.txt
```

For using in the MCP client, here we use the Claude desktop Client as an example, first you need to add MCP path to the `claude_desktop_config.json`:

```python
{
    "mcpServers": {
        "Optimization": {
            "command": "/{ABSOLUTE PATH TO UV INSTALLED FOLDER}/uv",
            "args": [
                "--directory",
                "/{ABSOLUTE PATH TO OR-LLM-AGENT FOLDER}",
                "run",
                "mcp_server.py.py"
            ]
        }
    }
}
```

Then you can open the Claude desktop Client, check if there is a `get_operation_research_problem_answer` in the hammer icon.	

<img src="../assets/mcp_client.png" alt="mcp_client" width="1000" height="auto" div align=center>

<br><br>

---
<p align="center">
Made at Shanghai Jiao Tong University and Nanyang Technological University
</p>
