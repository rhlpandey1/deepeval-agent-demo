
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from agent_instrumented import support_agent as _support_agent
from judge_factory import get_judge
from dataset import GOLDEN_DATA


# Use the configured evaluation provider (Claude/OpenAI)
judge = get_judge()


# -------------------------------------------------------
# Traced support agent
# -------------------------------------------------------

@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()

    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)

        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)

    return _support_agent(user_input)


# -------------------------------------------------------
# Define evaluation metrics
# -------------------------------------------------------

task_completion = TaskCompletionMetric(
    threshold=0.7,
    model=judge
)

tool_correctness = ToolCorrectnessMetric(
    model=judge
)


# -------------------------------------------------------
# Build evaluation dataset from central dataset.py
# -------------------------------------------------------

dataSet = EvaluationDataset(
    goldens=[
        Golden(
            input=item["input"],
            expected_tools=[
                ToolCall(name=name)
                for name in item["expected_tools"]
            ],
            expected_output=item["expected_output"],
        )
        for item in GOLDEN_DATA
    ]
)


# -------------------------------------------------------
# Execute evaluations
# -------------------------------------------------------

def run_evaluations():
    for golden in dataSet.evals_iterator(
        metrics=[task_completion, tool_correctness]
    ):
        support_agent(golden.input)


# -------------------------------------------------------
# Pytest execution
# -------------------------------------------------------

def test_tracing_components():
    run_evaluations()


# -------------------------------------------------------
# Direct Python execution
# -------------------------------------------------------

if __name__ == "__main__":
    run_evaluations()
