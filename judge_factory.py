
import os

from pathlib import Path
from dotenv import load_dotenv
from deepeval.models import OpenAIModel

from claude_judge import ClaudeJudge


# Load project-level environment configuration
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def get_judge():
    """
    Creates an evaluation judge based on EVAL_MODEL_PROVIDER.

    Supported providers:
        - claude
        - openai
    """

    provider = os.getenv(
        "EVAL_MODEL_PROVIDER", "claude"
    ).strip().lower()

    if provider == "claude":

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is missing. "
                "Please configure it in your .env file."
            )

        print("Evaluation judge: Claude")
        return ClaudeJudge()

    elif provider == "openai":

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing. "
                "Please configure it in your .env file."
            )

        model_name = os.getenv(
            "OPENAI_JUDGE_MODEL", "gpt-4o"
        )

        print(f"Evaluation judge: OpenAI ({model_name})")

        return OpenAIModel(
            model=model_name,
            api_key=api_key
        )

    else:
        raise ValueError(
            f"Unsupported evaluation provider: {provider}. "
            "Supported providers: claude, openai."
        )
