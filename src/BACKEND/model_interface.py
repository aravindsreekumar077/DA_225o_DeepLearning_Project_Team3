
#Importing the model
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig
import torch
from SLM.src.agentic import Agent 
from SLM.src.config import get_pretrained_config,get_default_config


class ModelInterfaceT5:
    def __init__(self):
        self.peft_model_id="src/MODELS/flan-t5-math-lora-out-saved"
        self.config = PeftConfig.from_pretrained(self.peft_model_id)
        self.base_model = T5ForConditionalGeneration.from_pretrained(self.config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.base_model, self.peft_model_id)
        self.tokenizer = T5Tokenizer.from_pretrained(self.peft_model_id)
        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.tokenizer=T5Tokenizer.from_pretrained(self.peft_model_id)
        

    def infer(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids=inputs.input_ids, max_length=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": answer}
    
class ModelInterfacePhi4:
    def __init__(self):
        # self.config = get_pretrained_config(repo_id="unsloth/Phi-4-mini-instruct-GGUF",
        #                     filename="src/SLM/models/Phi-4-mini-instruct-Q5_K_M.gguf")
        self.config= get_default_config()
        self.config.model.use_prompt=False
        self.agent=Agent(self.config)
        self.system_message=  """
You are a helpful AI assistant with access to tool. Before acting on a step, ALWAYS list out your upcoming actions in less than or equal to 3 parts. When you need to use a tool, list what you're gonna do, with what values and then call:
1. Output a JSON object, with 'name' and 'parameters'. For example:
   {"name": "calculator", "parameters": {"expression": "3.14 * a **2"}}
   {"name": "get_date", "parameters": {}}
   {"name": "python_shell", "parameters": {"code": "<code using standard libraries>"}}
   {"name": "get_weather_details", "parameters": {"location": "<location>"}}
2. Wait for the tool result, once you get it, say something like "Now that we have the `result_value`, we can proceed with the next step."
3. Continue your response using the tool result

Available tools:
- calculator: Evaluates simple mathematical expressions (requires "expression" parameter). The calculator supports the following functions:
  "sin", "cos", "tan", "log", "exp", "sqrt", "floor", "ceil", "asin",
  "acos", "atan", "degrees", "radians", "pi", "e", "pow".
- get_date: Returns current date in YYYYMMDD format (requires empty {} parameter)
- python_shell: Executes Python code and returns its result (requires "code" parameter)
- get_weather_details: Fetches weather details for a location (requires "location" parameter)

Tool results will be provided in the format: [Tool <name> returned: <result> | $result_N], always use $result_N to refer to the result of the Nth tool call. You cannot refer to subvalues within $result_N, always use the whole value.

If you encounter errors, or are asked something which the tools dont support switch to the python_shell tool and always use standard libraries, double-check variable names, return types and print outputs.

Never run the same tool twice with the same values or pass empty parameters except for get_date.

Never stimulate the response or caclulate it manually, always use the tools to get the result, if the tool response doesn't come fix your json call and try again.

Once all steps in the prompt are completed, summarize write an end message and print a tick mark (✓) at the end of your step to indicate completion.

The minute you type "USER" the system will crash, be cautious.
"""
    def infer(self, query: str) -> str:
        response=""
        for tok in self.agent.chat(self.system_message,query):
            response += tok
        return {"response": response}
    
# if __name__ == "__main__":
#     model_interface = ModelInterfaceT5()
#     query = "translate English to French: How are you?"
#     response = model_interface.infer(query)
#     print(response)  # Should print the answer to the query