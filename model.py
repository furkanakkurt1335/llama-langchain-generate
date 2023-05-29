from typing import List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from utils.prompter import Prompter
from generate import evaluate

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return evaluate(prompt)

llm = CustomLLM()

prompter = Prompter()

instruction = "Write a poem about a llama."
print("Initial instruction: ", instruction)
input = "Llamas are cute."
print("Initial input: ", input)

prompt = prompter.generate_prompt(instruction, input)
print("Generated Prompt:", prompt)

response = llm(prompt)
print("Response: ", response)
