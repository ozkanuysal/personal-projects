from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = "How many cat in the picture?"

encoding = processor(image,text,return_tensors="pt")
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()

print("Prediction:", model.config.id2label[idx])
