
from deepeval.models import DeepEvalBaseLLM
from langchain_anthropic import ChatAnthropic


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return "claude-haiku-4-5"
