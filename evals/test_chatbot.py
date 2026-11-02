
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.evaluate import evaluate
from deepeval.metrics import (
    TurnRelevancyMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
)
from deepeval.test_case import ConversationalTestCase, Turn

from chatbot import chat
from judge_factory import get_judge


def run_evaluations():

    # Use the configured evaluation provider (Claude/OpenAI)
    judge = get_judge()

    # -------------------------------------------------------
    # Run multi-turn chatbot conversation
    # -------------------------------------------------------

    turns = []
    history = []

    user_messages = [
        "Hi! I placed an order last week, the order ID is ORD-1042.",
        "Is it going to arrive on time?",
        "What was the ETA you just mentioned?",
        "Can I upgrade to express shipping?",
    ]

    for user_msg in user_messages:

        reply, history, tools_called = chat(user_msg, history)

        turns.append(
            Turn(
                role="user",
                content=user_msg
            )
        )

        turns.append(
            Turn(
                role="assistant",
                content=reply
            )
        )

        if tools_called:
            for tool in tools_called:
                print(
                    f"[Tool] {tool['name']} "
                    f"({tool['args']}) -> {tool['result']}"
                )

    # -------------------------------------------------------
    # Create conversational test case
    # -------------------------------------------------------

    test_case = ConversationalTestCase(
        turns=turns
    )

    # -------------------------------------------------------
    # Define metrics using configured judge
    # -------------------------------------------------------

    turn_relevancy_metric = TurnRelevancyMetric(
        threshold=0.5,
        model=judge
    )

    retention_metric = KnowledgeRetentionMetric(
        threshold=0.5,
        model=judge
    )

    completeness_metric = ConversationCompletenessMetric(
        threshold=0.5,
        model=judge
    )

    metrics = [
        turn_relevancy_metric,
        retention_metric,
        completeness_metric,
    ]

    # -------------------------------------------------------
    # Execute evaluation
    # -------------------------------------------------------

    evaluate(
        test_cases=[test_case],
        metrics=metrics
    )


# -----------------------------------------------------------
# Pytest execution
# -----------------------------------------------------------

def test_chatbot_evaluation():
    run_evaluations()


# -----------------------------------------------------------
# Direct Python execution
# -----------------------------------------------------------

if __name__ == "__main__":
    run_evaluations()
