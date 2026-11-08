import re

import networkx as nx

from agent.amino_acids import PROPERTIES, SOURCES
from src.prepare_data import WT_SEQUENCE


class GB1KnowledgeGraph:
    """位点、突变、残基性质和变体观测的有向图"""

    def __init__(self, known):
        """使用已知实验数据构建图谱"""

        self.graph = nx.DiGraph(
            protein="GB1", wild_type=WT_SEQUENCE,
            assay="Olson 2014 GB1 binding enrichment",
            score_type="measured_log_enrichment",
            fitness_definition="ln((selection+0.5)/(input+0.5)) - WT 同式值",
        )
        for aa, properties in PROPERTIES.items():
            self.graph.add_node(f"aa:{aa}", kind="amino_acid", residue=aa, **properties)
        for position, original in enumerate(WT_SEQUENCE, start=1):
            position_id = f"position:{position}"
            self.graph.add_node(position_id, kind="position", position=position)
            self.graph.add_edge(position_id, f"aa:{original}", relation="wild_type")
            if position == 1:
                continue
            for target in PROPERTIES:
                if target == original:
                    continue
                mutation = f"{original}{position}{target}"
                mutation_id = f"mutation:{mutation}"
                self.graph.add_node(mutation_id, kind="mutation", mutation=mutation)
                self.graph.add_edge(position_id, mutation_id, relation="allows_mutation")
                self.graph.add_edge(mutation_id, f"aa:{original}", relation="from_residue")
                self.graph.add_edge(mutation_id, f"aa:{target}", relation="to_residue")

        # fitness 记录在完整变体节点上
        for row in known.itertuples(index=False):
            variant_id = f"variant:{row.id}"
            self.graph.add_node(
                variant_id, kind="variant", candidate_id=int(row.id),
                sequence=row.sequence, mutations=row.mutations,
                n_mutations=int(row.n_mutations), measured_fitness=float(row.fitness),
                source=row.source,
            )
            if row.mutations != "WT":
                for mutation in row.mutations.split(":"):
                    self.graph.add_edge(
                        variant_id, f"mutation:{mutation}", relation="contains_mutation",
                    )

    def compare_properties(self, original, target):
        """返回残基性质及替换前后的差值"""

        if original not in PROPERTIES or target not in PROPERTIES:
            raise ValueError("残基必须为大写标准氨基酸单字母")
        before = self.graph.nodes[f"aa:{original}"]
        after = self.graph.nodes[f"aa:{target}"]
        return {
            "original": original, "target": target,
            "properties": {
                name: {
                    "before": before[name], "after": after[name],
                    "delta": round(after[name] - before[name], 4),
                    "source": SOURCES[name],
                }
                for name in SOURCES
            },
            "limitation": "性质相近可支持保守替换假设，性质变化较大可支持激进替换假设；"
                          "这些尺度不提供 GB1 有益或有害的结论，也不是结构或保守性证据",
        }

    def query_mutation(self, mutation, limit=3):
        """分别查询突变的单点证据与组合证据"""

        if not re.fullmatch(r"[A-Z][0-9]+[A-Z]", mutation):
            raise ValueError("请提供单个突变，例如 A20C")
        mutation_id = f"mutation:{mutation}"
        if mutation_id not in self.graph:
            raise ValueError("突变不符合 GB1 野生型、2—56 位范围或标准氨基酸要求")
        if not 1 <= limit <= 20:
            raise ValueError("limit 应在 1—20 之间")
        observations = [
            {"evidence_id": node, **self.graph.nodes[node]}
            for node in self.graph.predecessors(mutation_id)
            if self.graph.nodes[node]["kind"] == "variant"
        ]
        singles = [row for row in observations if row["n_mutations"] == 1]
        combinations = sorted(
            (row for row in observations if row["n_mutations"] > 1),
            key=lambda row: (-row["measured_fitness"], row["candidate_id"]),
        )
        return {
            "mutation": mutation,
            "assay": dict(self.graph.graph),
            "single_mutant_observations": singles,
            "combination_count": len(combinations),
            "highest_known_combinations": [
                self.combination_evidence(row) for row in combinations[:limit]
            ],
            "lowest_known_combinations": [
                self.combination_evidence(row) for row in reversed(combinations[-limit:])
            ],
            "residue_properties": self.compare_properties(mutation[0], mutation[-1]),
            "limitation": "仅包含训练集与本组已揭示反馈；空单点列表表示尚无允许访问的单点证据。"
                          "组合分数属于完整序列，不能拆为单点效应；单点收益不能保证组合收益。"
                          "最高和最低背景是描述性例子，不是因果结论或显著性检验",
        }

    def combination_evidence(self, observation):
        """查询单点对照，齐全时计算组合的加性偏离"""

        components = []
        for mutation in observation["mutations"].split(":"):
            singles = [
                {"evidence_id": node, **self.graph.nodes[node]}
                for node in self.graph.predecessors(f"mutation:{mutation}")
                if self.graph.nodes[node]["kind"] == "variant"
                and self.graph.nodes[node]["n_mutations"] == 1
            ]
            components.append({"mutation": mutation, "single_mutant_observations": singles})
        # WT=0，仅在单点对照齐全时计算加性偏离
        complete = all(len(row["single_mutant_observations"]) == 1 for row in components)
        expected = sum(
            row["single_mutant_observations"][0]["measured_fitness"] for row in components
        ) if complete else None
        return {
            **observation, "component_evidence": components,
            "additive_expected_fitness": expected,
            "additive_deviation": (
                observation["measured_fitness"] - expected if complete else None
            ),
            "interpretation": "偏离仅为 WT=0 对数富集尺度上的算术差值，未进行统计显著性检验；"
                              "null 表示缺少完整单点对照，不代表偏离为零",
        }

    def query_position(self, position):
        """汇总位点各替换的已知证据"""

        if not 2 <= position <= 56:
            raise ValueError("可变位点范围为 2—56")
        substitutions = []
        for node in self.graph.successors(f"position:{position}"):
            if self.graph.nodes[node]["kind"] != "mutation":
                continue
            observations = [
                self.graph.nodes[parent]
                for parent in self.graph.predecessors(node)
                if self.graph.nodes[parent]["kind"] == "variant"
            ]
            substitutions.append({
                "mutation": self.graph.nodes[node]["mutation"],
                "known_variant_count": len(observations),
                "single_mutant_observations": [
                    {"evidence_id": f"variant:{row['candidate_id']}", **row}
                    for row in observations if row["n_mutations"] == 1
                ],
            })
        return {
            "position": position, "wild_type": WT_SEQUENCE[position - 1],
            "substitutions": substitutions,
            "score_type": self.graph.graph["score_type"],
            "limitation": "观测数量反映当前已知数据覆盖范围，不代表某替换更优；缺失分数不是 0",
        }
