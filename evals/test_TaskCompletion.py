
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

from judge_factory import get_judge
from dataset import GOLDEN_DATA


def run_evaluations():

    # Initialize the configured evaluation judge
    judge = get_judge()

    # -------------------------------------------------------
    # Get test input and expected output from shared dataset
    # -------------------------------------------------------

    golden = next(
        item for item in GOLDEN_DATA
        if item["input"] == "Where is my order ORD-1042?"
    )

    user_input = golden["input"]

    # -------------------------------------------------------
    # Execute the agent
    # -------------------------------------------------------

    actual_output = support_agent(user_input)

    print(f"Input: {user_input}")
    print(f"Expected Output: {golden['expected_output']}")
    print(f"Actual Output: {actual_output}")

    # -------------------------------------------------------
    # Create test case
    # -------------------------------------------------------

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        expected_output=golden["expected_output"]
    )

    # -------------------------------------------------------
    # Define evaluation metric
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
