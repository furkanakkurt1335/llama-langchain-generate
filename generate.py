import sys, torch, argparse
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from typing import List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from utils.prompter import Prompter

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--lora_weights", type=str)
args = parser.parse_args()

base_model = args.base_model
if args.lora_weights:
    from peft import PeftModel
    lora_weights = args.lora_weights
else:
    lora_weights = None

load_8bit = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = LlamaTokenizer.from_pretrained(base_model)

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model, device_map={"": device}, low_cpu_mem_usage=True
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def evaluate(
    prompt,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output

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
