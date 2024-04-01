from model import model_pipeline
from typing import Union
from fastapi import FastAPI, UploadFile
import io
from PIL import Image

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/ask")
def ask_question(text: str, image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    result = model_pipeline(text, image)
    return {"answer": result}