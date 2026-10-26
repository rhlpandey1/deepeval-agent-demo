
import os
import sys

# Add project root to Python's module search path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from agent_instrumented import support_agent

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from claude_judge import ClaudeJudge

judge = ClaudeJudge()



actual_output = support_agent("Where is my order ORD-1042?")

test_case = LLMTestCase(
    input="Where is my order ORD-1042?",
    actual_output=actual_output
)

evaluate(
    test_cases=[test_case],
    metrics=[
        TaskCompletionMetric(
            threshold=0.6,
            model=judge
        )
    ]
)
