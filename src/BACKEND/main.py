##SLAM-Backend##
from pydantic import BaseModel
from TOOLS.OCR import get_ocr_text
from fastapi import FastAPI, UploadFile, File, HTTPException


app = FastAPI()

'''
#Importing the model
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig
import torch

peft_model_id = "/kaggle/input/t5/pytorch/default/1/flan-t5-math-lora-saved"

# 1. Load LoRA config
config = PeftConfig.from_pretrained(peft_model_id)

# 2. Load base model (T5, for example)
base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)

# 3. Load adapter weights into base model
model = PeftModel.from_pretrained(base_model, peft_model_id)

# 4. Load tokenizer
tokenizer = T5Tokenizer.from_pretrained(peft_model_id)
model = model.to(device)
'''




class Query(BaseModel):
    input_text: str

@app.get("/ping")
def ping():
    return {"response": "Hi , SLAM backend is up and running"}


@app.post("/OCR")
async def get_ocr(image: UploadFile = File(...)):
    image=await(image.read())
    return get_ocr_text(image)

@app.post("/calculator")
def calculate():
    return {"response": "Hi , Placeholder for calculator"}


@app.post("/json_formatter")
def json_format():
    return {"response": "Hi , Placeholder for json_formatter"}

@app.post("/translator")
def translator(text: str):
    return {"response": f"Hi , Placeholder for translator-{text}"}

#Inference code for model
'''
@app.post("/infer")
async def generate_output(query: Query):


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    question = "what is ninety eight times fifteen?"
    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=inputs.input_ids, max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": answer}
'''