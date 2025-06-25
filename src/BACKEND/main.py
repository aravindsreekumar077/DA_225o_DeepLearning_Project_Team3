##SLAM-Backend##
from pydantic import BaseModel
from TOOLS.OCR import get_ocr_text
from TOOLS.calculator import evaluate_expression
from fastapi import FastAPI, UploadFile, File, HTTPException
from model_interface import ModelInterfaceT5    

app = FastAPI()

model_interface = ModelInterfaceT5()

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
def calculate(query : str):
    response = evaluate_expression(query)
    return {"response": response} 

@app.post("/json_formatter")
def json_format(query: str = None):
    return {"response": query}

@app.post("/translator")
def translator(text: str):
    return {"response": f"Hi , Placeholder for translator-{text}"}

@app.post("/infer_t5")
async def infer_t5(query: Query):
    try:
        response = model_interface.infer(query.input_text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
