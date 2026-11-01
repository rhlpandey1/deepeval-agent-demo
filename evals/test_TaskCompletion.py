
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


def run_evaluations():

    # Initialize Claude as the evaluation judge
    judge = ClaudeJudge()

    # -------------------------------------------------------
    # Execute the agent
    # -------------------------------------------------------

    user_input = "Where is my order ORD-1042?"

    actual_output = support_agent(user_input)

    print(f"Input: {user_input}")
    print(f"Actual Output: {actual_output}")

    # -------------------------------------------------------
    # Create test case
    # -------------------------------------------------------

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output
    )

    # -------------------------------------------------------
    # Define evaluation metric using Claude
    # -------------------------------------------------------

    task_completion_metric = TaskCompletionMetric(
        threshold=0.6,
        model=judge
    )

    # -------------------------------------------------------
    # Execute evaluation
    # -------------------------------------------------------

    evaluate(
        test_cases=[test_case],
        metrics=[task_completion_metric]
    )


# -----------------------------------------------------------
# Pytest execution
# -----------------------------------------------------------

def test_task_completion():
    run_evaluations()


# -----------------------------------------------------------
# Direct Python execution
# -----------------------------------------------------------

if __name__ == "__main__":
    run_evaluations()
