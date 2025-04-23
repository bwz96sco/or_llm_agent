from mcp.server.fastmcp import FastMCP
from or_llm_eval import or_llm_agent
import io
from contextlib import redirect_stdout
# Initialize FastMCP server
mcp = FastMCP("or_llm_agent")

@mcp.tool()
def get_operation_research_problem_answer(user_question: str) -> str:
    """
    Use the agent to solve the optimization problem. Agent will generate python gurobi code and run it to get the result.

    Args:
        user_question: The user's question

    Returns:
        The result of the optimization problem, including math model, gurobi code and result.
    """
    buffer2 = io.StringIO()
    with redirect_stdout(buffer2):
        or_llm_agent(user_question)
    return buffer2.getvalue()

if __name__ == "__main__":
    mcp.run()
