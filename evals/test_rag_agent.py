import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

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
from claude_judge import ClaudeJudge


def build_agent():
    @observe(name="rag_support_agent")
    def rag_support_agent(user_input: str) -> str:
        golden = get_current_golden()
        if golden and golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
        return _rag_support_agent(user_input)
    return rag_support_agent


def run_evaluations():
    judge = ClaudeJudge()
    rag_support_agent = build_agent()

    dataset = EvaluationDataset(
        goldens=[
            Golden(
                input="What is the return policy for electronics?",
                expected_output=(
                    "Electronics can be returned within 15 days of delivery if unopened "
                    "and in original packaging. Refunds take 5–7 business days."
                ),
            ),
            Golden(
                input="How long does express shipping take and what does it cost?",
                expected_output=(
                    "Express shipping takes 1–2 business days and costs $15. "
                    "Orders placed before 2 PM are dispatched the same day."
                ),
            ),
        ]
    )

    metrics = [
        ContextualPrecisionMetric(threshold=0.7, model=judge),
        ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True),
    ]

    for golden in dataset.evals_iterator(metrics=metrics):
        rag_support_agent(golden.input)


def test_rag_agent():
    run_evaluations()


if __name__ == "__main__":
    run_evaluations()
