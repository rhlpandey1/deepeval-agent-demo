
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    PromptAlignmentMetric,
    StepEfficiencyMetric,
    AnswerRelevancyMetric
)
from deepeval.tracing import observe

from dataset import GOLDEN_DATA
from agent_instrumented import support_agent as _support_agent
from judge_factory import get_judge


# ---------------------------------------------------------------------------
# LLM Judge - Provider selected through environment configuration
# ---------------------------------------------------------------------------

judge = get_judge()


# ---------------------------------------------------------------------------
# Agent wrapper with DeepEval tracing
# ---------------------------------------------------------------------------

@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge
)

step_efficiency = StepEfficiencyMetric(
    threshold=0.5,
    model=judge
)

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."
    ],
    threshold=0.7,
    model=judge
)


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input=item["input"],
            expected_output=item["expected_output"]
        )
        for item in GOLDEN_DATA
    ]
)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluations():

    metrics = [
        prompt_alignment,
        step_efficiency,
        answer_relevancy
    ]

    for golden in dataset.evals_iterator(metrics=metrics):
        support_agent(golden.input)


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------

def test_multiple_metrics_evaluation():
    run_evaluations()


if __name__ == "__main__":
    run_evaluations()
