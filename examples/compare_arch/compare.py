import os
from langgraph_compare import compare

# Compares all experiments that have already been run.
# Run each architecture script first via its VS Code launch config,
# then execute this script to generate the comparison report.
experiment_names = [
    "main",
    "tavily",
    "supervision",
    "hierarchical",
    "network",
    "mcp_math_100",
    "mcp_docs_100",
]

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Only include experiments whose metrics report was actually generated.
available = [
    name for name in experiment_names
    if os.path.exists(os.path.join(repo_root, "experiments", name, "reports", "metrics_report.json"))
]

if len(available) < 2:
    print(f"[ERROR] Need at least 2 experiments to compare, found: {available}")
else:
    print(f"Comparing: {available}\n")
    compare(available)
