from typing import Annotated

from pydantic import Field

from agent.knowledge import GB1KnowledgeGraph


def register_knowledge_tools(mcp, known):
    """根据已知实验构建图谱并注册查询工具"""

    knowledge = GB1KnowledgeGraph(known)

    @mcp.tool(structured_output=True)
    def query_mutation_knowledge(
        mutation: str,
        limit: Annotated[int, Field(ge=1, le=20)] = 3,
    ) -> dict[str, object]:
        """查询单个替换的单点证据、组合背景及性质变化，如 A20C

        返回变体 ID 和来源，组合 fitness 属于完整变体
        limit 分别限制最高、最低组合背景的返回数量
        """
        return knowledge.query_mutation(mutation, limit)

    @mcp.tool(structured_output=True)
    def query_position_knowledge(
        position: Annotated[int, Field(ge=2, le=56)],
    ) -> dict[str, object]:
        """查询位点的 19 种替换和已知单点证据，缺失标签保持为空"""
        return knowledge.query_position(position)

    @mcp.tool(structured_output=True)
    def compare_residue_properties(original: str, target: str) -> dict[str, object]:
        """比较标准氨基酸的疏水性、极性、大小和电荷编码，返回来源

        性质差值不代表 GB1 fitness 的变化
        """
        return knowledge.compare_properties(original, target)
