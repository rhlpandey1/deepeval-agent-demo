
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.contextvars import get_current_golden
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.tracing import observe, update_current_trace

from rag_agent import rag_support_agent as _rag_support_agent
from judge_factory import get_judge
from dataset import GOLDEN_DATA


def build_agent():

    @observe(name="rag_support_agent")
    def rag_support_agent(user_input: str) -> str:

        golden = get_current_golden()

        if golden and golden.expected_output:
            update_current_trace(
                expected_output=golden.expected_output
            )

        return _rag_support_agent(user_input)

    return rag_support_agent


def run_evaluations():

    # Use configured evaluation provider
    judge = get_judge()

    rag_support_agent = build_agent()

    # -------------------------------------------------------
    # Build RAG evaluation dataset from central dataset.py
    # -------------------------------------------------------

    rag_inputs = {
        "What is the return policy for electronics?",
        "How long does express shipping take and what does it cost?",
    }

    dataset = EvaluationDataset(
        goldens=[
            Golden(
                input=item["input"],
                expected_output=item["expected_output"],
            )
            for item in GOLDEN_DATA
            if item["input"] in rag_inputs
        ]
    )

    # -------------------------------------------------------
    # Define RAG evaluation metrics
    # -------------------------------------------------------

    metrics = [
        ContextualPrecisionMetric(
            threshold=0.7,
            model=judge
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model=judge,
            include_reason=True
        ),
        AnswerRelevancyMetric(
            threshold=0.7,
            model=judge,
            include_reason=True
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=judge,
            include_reason=True
        ),
    ]

    # -------------------------------------------------------
    # Execute evaluations
    # -------------------------------------------------------

    for golden in dataset.evals_iterator(metrics=metrics):
        rag_support_agent(golden.input)


def test_rag_agent():
    run_evaluations()


if __name__ == "__main__":
    run_evaluations()
