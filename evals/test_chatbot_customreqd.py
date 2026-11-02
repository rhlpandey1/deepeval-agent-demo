
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deepeval.evaluate import evaluate
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import (
    Turn,
    ConversationalTestCase,
    MultiTurnParams,
)

from chatbot import chat
from judge_factory import get_judge


def run_evaluations():

    # Use configured evaluation provider (Claude/OpenAI)
    judge = get_judge()

    turns = []
    history = []

    # -------------------------------------------------------
    # Run multi-turn chatbot conversation
    # -------------------------------------------------------

    user_messages = [
        "Hi! I placed an order last week, the order ID is ORD-1042.",
        "Is it going to arrive on time?",
        "What was the ETA you just mentioned?",
        "Can I upgrade to express shipping?",
    ]

    for user_msg in user_messages:

        reply, history, _ = chat(user_msg, history)

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

    # -------------------------------------------------------
    # Define conversational correctness metric
    # -------------------------------------------------------

    correctness = ConversationalGEval(
        name="Correctness",
        criteria=(
            "Did the chatbot fully resolve the customer's issue? "
            "It should use tools when needed and provide accurate answers."
        ),
        model=judge,
        threshold=0.80,
        evaluation_params=[
            MultiTurnParams.ROLE,
            MultiTurnParams.CONTENT,
        ],
    )

    # -------------------------------------------------------
    # Create conversational test case
    # -------------------------------------------------------

    test_case = ConversationalTestCase(
        turns=turns
    )

    # -------------------------------------------------------
    # Execute evaluation
    # -------------------------------------------------------

    evaluate(
        test_cases=[test_case],
        metrics=[correctness]
    )


# -------------------------------------------------------
# Pytest execution
# -------------------------------------------------------

def test_chatbot_custom_evaluation():
    run_evaluations()


# -------------------------------------------------------
# Direct Python execution
# -------------------------------------------------------

if __name__ == "__main__":
    run_evaluations()
