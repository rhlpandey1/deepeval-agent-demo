
import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
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

    if golden and golden.expected_output:
        update_current_trace(
            expected_output=golden.expected_output
        )

    return _support_agent(user_input)


# -------------------------------------------------------
# Define custom correctness metric
# -------------------------------------------------------

correctness = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output conveys the same factual information "
        "as the expected output. Minor wording differences are acceptable; "
        "missing or wrong facts are not."
    ),
    model=judge,
    threshold=0.79,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
)


# -------------------------------------------------------
# Build evaluation dataset from central dataset.py
# -------------------------------------------------------

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input=item["input"],
            expected_output=item["expected_output"],
        )
        for item in GOLDEN_DATA
    ]
)


# -------------------------------------------------------
# Execute evaluations
# -------------------------------------------------------

def run_evaluations():
    for golden in dataset.evals_iterator(metrics=[correctness]):
        support_agent(golden.input)


# -------------------------------------------------------
# Pytest execution
# -------------------------------------------------------

def test_custom_metric_evaluation():
    run_evaluations()


# -------------------------------------------------------
# Direct Python execution
# -------------------------------------------------------

if __name__ == "__main__":
    run_evaluations()
