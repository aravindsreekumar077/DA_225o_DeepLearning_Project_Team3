
#Importing the model
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig
import torch


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
    
# if __name__ == "__main__":
#     model_interface = ModelInterfaceT5()
#     query = "translate English to French: How are you?"
#     response = model_interface.infer(query)
#     print(response)  # Should print the answer to the query